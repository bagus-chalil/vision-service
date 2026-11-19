"""
Formula coding (batch + EXP/MFG) per SKU: katalog token + parser.

Keputusan user 2026-11-19: AI mengenali FORMULA coding, di mana pun letaknya (emboss crimp,
inkjet di badan tube, kotak label, tutup) - TIDAK ada asumsi "depan = batch, belakang = EXP".
Pembagian tugas:
  - Ollama (llm_extract.analyze_coding_formula) hanya MENYALIN baris kode apa adanya.
  - Modul ini (Python murni, deterministik) mencari teks yang cocok formula di salinan itu.
  - Formula dikirim dari luar (ipc_app, per SKU) - sama seperti field_type, Vision Service
    tidak pernah menebak formula dari foto. Formula TIDAK dimasukkan ke prompt.
Prinsip lama tetap: nilai tidak pernah diedit/ditebak; tidak cocok/ambigu = flag (REVIEW),
bukan koreksi. Yang toleran HANYA pemisah antar elemen (spasi, /, ., :, -, _, baris baru),
karena di lapangan coding bisa "menyatu atau tidak" (EXP301029 vs EXP 301029).

Sintaks template (satu "segmen" = satu blok coding; formula = 1..n segmen):
  {TOKEN}         token wajib, field default menurut token (lihat TOKENS)
  {TOKEN@field}   token untuk field tertentu (mis. {MM@mfg})
  teks biasa      label literal, tidak peka huruf besar/kecil (EXP, MFG, BATCH NO, NOT, EX)
  [ ... ]         bagian opsional; alternatif dipisah |  -> [EXP DATE|EXP]
  ( ... )         alternatif wajib                     -> (EXP|ED)
  Pemisah antar elemen selalu opsional & bebas: "{DD}{MM}{YY}" cocok dgn 181129, 18 11 29,
  18/11/29. Ujung segmen tidak boleh menempel huruf/angka lain (gotcha #15: 310826QHB8 tidak
  boleh menghasilkan EXP 310826; barcode 13 digit tidak menghasilkan tanggal).

Contoh (dari Format_Kodifikasi_IPC + data operator):
  "{BATCH3} EXP {DD}{MM}{YY} [{INIT}]"                OKB EXP 231029 CW
  "EXP {MMM} {YY}"                                     EXP NOV 29 (tube Scarlett)
  "EX{MM}{YY} {BATCH3}"                                EX0927 AJZ (sachet)
  "{BATCH3} {MM@mfg}/{YYYY@mfg} {MM}/{YYYY}"           AHB / 08/2026 / 04/2029 (tanpa label)
  "[EXP] {MM}{YYYY}"                                   EXP 112028 (MMYYYY, bukan DDMMYY!)
"""

import calendar
import re

# --------------------------------------------------------------------------
# Katalog token
# --------------------------------------------------------------------------
# kind: date part (DD/MM/YY/YYYY/MMM) -> divalidasi kalender; text -> apa adanya.
TOKENS = {
    "BATCH3": {"re": r"[A-Z]{3}", "field": "batch", "kind": "text",
               "desc": "Batch PPIC 3 huruf besar (OKB, PLB, AHB)"},
    "INIT": {"re": r"[A-Za-z]{2}", "field": "init", "kind": "text",
             "desc": "Inisial Line Leader & IPC, 2 huruf (huruf kecil juga muncul: bH, cW)"},
    "DD": {"re": r"[0-9]{2}", "field": "exp", "kind": "day", "desc": "Hari 01-31"},
    "MM": {"re": r"[0-9]{2}", "field": "exp", "kind": "month", "desc": "Bulan 01-12"},
    "MMM": {"re": r"[A-Za-z]{3}", "field": "exp", "kind": "month_name",
            "desc": "Nama bulan 3 huruf (JAN..DEC, juga MEI/AGU/AGS/OKT/DES)"},
    "YY": {"re": r"[0-9]{2}", "field": "exp", "kind": "year2", "desc": "Tahun 2 digit (20YY)"},
    "YYYY": {"re": r"20[0-9]{2}", "field": "exp", "kind": "year4", "desc": "Tahun 4 digit 20xx"},
}
# Token generik berpanjangan: L<n> huruf besar, D<n> angka, X<n> huruf besar/angka.
# Field default None (tidak dilaporkan) kecuali diberi @field, mis. {X7@batch} untuk 26Y070X.
GENERIC_TOKEN = re.compile(r"^([LDX])([1-9][0-9]?)$")
GENERIC_RE = {"L": "[A-Z]{%d}", "D": "[0-9]{%d}", "X": "[A-Z0-9]{%d}"}

MONTH_NAMES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "MEI": 5, "JUN": 6, "JUL": 7,
    "AUG": 8, "AGU": 8, "AGS": 8, "AGT": 8, "SEP": 9, "OCT": 10, "OKT": 10, "NOV": 11,
    "DEC": 12, "DES": 12,
}
# Kata 3 huruf yang bukan batch (label/bulan). Batch PPIC yang kebetulan sama dengan kata ini
# -> NOT_FOUND -> REVIEW (arah aman), bukan tebakan.
RESERVED_WORDS = {"EXP", "MFG", "MFD", "LOT", "NOT", "NET", "USE", "BPOM", "ED"} | set(MONTH_NAMES)

# Label yang, bila tepat di depan kandidat TANPA label, berarti kandidat itu milik field lain.
CONFLICT_LABELS = {
    "exp": ("MFG DATE", "MFD DATE", "MFG", "MFD", "PROD", "PRODUCTION DATE", "MANUFACTURED", "NA", "NOT"),
    "mfg": ("EXP DATE", "EXP", "ED", "USE BEFORE", "BEST BEFORE", "BB", "NA", "NOT"),
    "batch": ("EXP", "MFG", "MFD", "NA", "NOT"),
}
DATE_FIELDS_KINDS = ("day", "month", "month_name", "year2", "year4")
REQUIRED_FIELDS = ("batch", "exp", "mfg")  # field yang menentukan status; init dll informatif

SEP = r"[\s/.:\-_,]*"
_SEP_CHARS = " \t\r\n/.:-_,"


class FormulaError(ValueError):
    """Template formula tidak valid (salah sintaks / token tak dikenal)."""


# --------------------------------------------------------------------------
# Kompilasi template -> regex
# --------------------------------------------------------------------------


def _token_spec(name):
    name = name.strip()
    if name in TOKENS:
        return TOKENS[name]
    m = GENERIC_TOKEN.match(name)
    if m:
        return {"re": GENERIC_RE[m.group(1)] % int(m.group(2)), "field": None, "kind": "text"}
    raise FormulaError(f"token tidak dikenal: {{{name}}}")


def _split_top(text, sep="|"):
    """Pisah di `sep` yang tidak berada di dalam {} [] ()."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _closing(text, i, open_ch, close_ch):
    depth = 0
    for j in range(i, len(text)):
        if text[j] == open_ch:
            depth += 1
        elif text[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j
    raise FormulaError(f"'{open_ch}' tanpa pasangan '{close_ch}' di: {text}")


class _Compiler:
    def __init__(self):
        self.groups = {}  # nama grup regex -> info
        self.n = 0

    def _name(self, info):
        self.n += 1
        name = f"g{self.n}"
        self.groups[name] = info
        return name

    def elements(self, text):
        """Template -> list regex elemen (tanpa pemisah; pemisah disisipkan di antaranya)."""
        els, i = [], 0
        while i < len(text):
            ch = text[i]
            if ch in _SEP_CHARS:
                i += 1
                continue
            if ch == "{":
                j = _closing(text, i, "{", "}")
                body = text[i + 1:j]
                tok, _, field = body.partition("@")
                spec = _token_spec(tok)
                info = {"type": "token", "token": tok.strip(), "kind": spec["kind"],
                        "field": (field.strip() or spec["field"])}
                els.append(f"(?P<{self._name(info)}>{spec['re']})")
                i = j + 1
            elif ch in "[(":
                close = "]" if ch == "[" else ")"
                j = _closing(text, i, ch, close)
                alts = [a for a in _split_top(text[i + 1:j]) if a.strip()]
                if not alts:
                    raise FormulaError(f"grup kosong di: {text}")
                # alternatif terpanjang dulu (EXP DATE sebelum EXP)
                alts.sort(key=lambda a: -len(a))
                compiled = "|".join(self.sequence(a) for a in alts)
                els.append(f"(?:{compiled})" + ("?" if ch == "[" else ""))
                i = j + 1
            elif ch in "]})|":
                raise FormulaError(f"'{ch}' tak terduga di: {text}")
            else:
                j = i
                while j < len(text) and text[j] not in _SEP_CHARS and text[j] not in "{[(]})|":
                    j += 1
                lit = text[i:j]
                info = {"type": "literal", "text": lit}
                els.append(f"(?P<{self._name(info)}>(?i:{re.escape(lit)}))")
                i = j
        return els

    def sequence(self, text):
        els = self.elements(text)
        if not els:
            raise FormulaError(f"segmen kosong: {text!r}")
        # Elemen opsional diberi pemisah di dalam grup opsionalnya, supaya pemisah tidak
        # "menggantung" bila elemen itu absen. Cukup: SEP di antara semua elemen (SEP boleh kosong).
        return SEP.join(els)


class CompiledSegment:
    def __init__(self, template):
        if not isinstance(template, str) or not template.strip():
            raise FormulaError("segmen formula harus string tidak kosong")
        self.template = template
        comp = _Compiler()
        body = comp.sequence(template)
        self.groups = comp.groups
        fields = {g["field"] for g in comp.groups.values() if g["type"] == "token" and g["field"]}
        if not fields:
            raise FormulaError(f"segmen tanpa field: {template!r}")
        self.fields = fields
        # Lookahead supaya pencarian overlapping (setiap posisi awal dicoba).
        self.regex = re.compile(r"(?=(?<![A-Za-z0-9])(?P<all>" + body + r")(?![A-Za-z0-9]))")


def compile_formula(formula):
    """formula: string (1 segmen) atau list string. -> list CompiledSegment."""
    segs = [formula] if isinstance(formula, str) else list(formula or [])
    if not segs:
        raise FormulaError("formula kosong")
    return [CompiledSegment(s) for s in segs]


# --------------------------------------------------------------------------
# Pencocokan
# --------------------------------------------------------------------------


def _date_from_parts(parts):
    """parts: {kind: raw}. -> (date dict | None, granularity, error | None). Tanpa menebak:
    kombinasi tak valid (bulan 13, 31 Februari) = error, bukan dibetulkan."""
    year = month = day = None
    if "year4" in parts:
        year = int(parts["year4"])
    elif "year2" in parts:
        year = 2000 + int(parts["year2"])
    if "month" in parts:
        month = int(parts["month"])
    elif "month_name" in parts:
        month = MONTH_NAMES.get(parts["month_name"].upper())
        if month is None:
            return None, None, f"nama bulan tidak dikenal: {parts['month_name']}"
    if "day" in parts:
        day = int(parts["day"])
    if month is not None and not 1 <= month <= 12:
        return None, None, f"bulan di luar 01-12: {month:02d}"
    if day is not None:
        if not 1 <= day <= 31:
            return None, None, f"hari di luar 01-31: {day:02d}"
        if month is not None and year is not None and day > calendar.monthrange(year, month)[1]:
            return None, None, f"tanggal tidak ada: {day:02d}/{month:02d}/{year}"
    if year is None or month is None:
        return None, None, "tanggal tidak lengkap (butuh minimal bulan + tahun)"
    return {"year": year, "month": month, "day": day}, ("day" if day is not None else "month"), None


def _preceding_label(text, start, labels):
    before = text[max(0, start - 24):start].rstrip(_SEP_CHARS).upper()
    for lab in sorted(labels, key=len, reverse=True):
        if re.search(r"(?<![A-Z0-9])" + re.escape(lab) + r"$", before):
            return lab
    return None


def _match_fields(seg, m, text):
    """Satu match regex -> {field: info} + alasan tolak (None bila diterima)."""
    found = {}
    first = None  # elemen paling awal yang ikut cocok
    for name, info in seg.groups.items():
        s = m.start(name)
        if s < 0:
            continue
        if first is None or s < first[0]:
            first = (s, info)
        if info["type"] != "token" or not info["field"]:
            continue
        f = found.setdefault(info["field"], {"spans": [], "parts": {}})
        f["spans"].append((s, m.end(name)))
        f["parts"].setdefault(info["kind"] if info["kind"] in DATE_FIELDS_KINDS else "text", "")
        key = info["kind"] if info["kind"] in DATE_FIELDS_KINDS else "text"
        f["parts"][key] += m.group(name)

    out, reject = {}, None
    for field, f in found.items():
        a = min(s for s, _ in f["spans"])
        b = max(e for _, e in f["spans"])
        entry = {"value": text[a:b], "start": a}
        if any(k in DATE_FIELDS_KINDS for k in f["parts"]):
            date, gran, err = _date_from_parts(f["parts"])
            entry.update(parts={k: v for k, v in f["parts"].items() if k != "text"},
                         date=date, granularity=gran)
            entry["key"] = (date["year"], date["month"], date["day"]) if date else None
            if err:
                reject = reject or f"{field}: {err}"
        else:
            entry["key"] = "".join(m.group(n) for n, i in seg.groups.items()
                                   if i["type"] == "token" and i["field"] == field and m.start(n) >= 0)
            if field == "batch" and entry["key"].upper() in RESERVED_WORDS:
                reject = reject or f"batch: '{entry['key']}' kata label/bulan, bukan batch"
        out[field] = entry

    # Segmen tanpa label di depannya: tolak bila didahului label milik field lain
    # (mis. "MFD 140624" tidak boleh jadi EXP; gotcha #13/#18).
    if first is not None and first[1]["type"] == "token" and first[1]["field"] in CONFLICT_LABELS:
        lab = _preceding_label(text, first[0], CONFLICT_LABELS[first[1]["field"]])
        if lab:
            reject = reject or f"{first[1]['field']}: didahului label '{lab}'"
    return out, reject


def find_candidates(text, segments):
    """Semua kecocokan (overlapping) tiap segmen di teks. -> (accepted, rejected)."""
    accepted, rejected, seen = [], [], set()
    for idx, seg in enumerate(segments):
        for m in seg.regex.finditer(text):
            span = (m.start("all"), m.end("all"))
            if (idx, span) in seen:
                continue
            seen.add((idx, span))
            fields, reject = _match_fields(seg, m, text)
            cand = {"segment": idx, "text": m.group("all"), "span": span, "fields": fields}
            (rejected if reject else accepted).append({**cand, "reason": reject} if reject else cand)
    return accepted, rejected


def _public_entry(e):
    return {k: v for k, v in e.items() if k not in ("start", "key")}


def extract(lines, formula):
    """Inti parser. lines: list[str] hasil salinan (atau satu string). formula: lihat compile_formula.

    Per field (batch/exp/mfg/...):
      FOUND      tepat satu nilai berbeda di semua kandidat yang diterima
      AMBIGUOUS  >1 nilai berbeda (mis. dua tanggal tanpa label) -> REVIEW, tidak memilih
      NOT_FOUND  tidak ada kandidat yang lolos
    Nilai = substring persis dari teks (termasuk pemisah apa adanya); `key` untuk pembanding
    (tanggal: (tahun, bulan, hari|None); teks: token digabung tanpa pemisah).
    """
    segments = formula if formula and isinstance(formula, list) and isinstance(formula[0], CompiledSegment) \
        else compile_formula(formula)
    text = lines if isinstance(lines, str) else "\n".join(l for l in (lines or []) if l)
    accepted, rejected = find_candidates(text, segments)
    wanted = set().union(*(s.fields for s in segments))
    fields = {}
    for field in sorted(wanted):
        by_key = {}
        for c in accepted:
            e = c["fields"].get(field)
            if e is not None:
                by_key.setdefault(e["key"], (e, c))
        if not by_key:
            fields[field] = {"status": "NOT_FOUND", "value": None}
        elif len(by_key) == 1:
            e, c = next(iter(by_key.values()))
            fields[field] = {"status": "FOUND", **_public_entry(e), "matched_text": c["text"],
                             "key": e["key"]}
        else:
            fields[field] = {
                "status": "AMBIGUOUS", "value": None,
                "candidates": [{**_public_entry(e), "matched_text": c["text"]} for e, c in by_key.values()],
            }
    return {
        "fields": fields,
        "required_fields": [f for f in REQUIRED_FIELDS if f in wanted],
        "rejected": [{"text": r["text"], "reason": r["reason"]} for r in rejected],
    }


def key_of(field_result):
    """Pembanding antar view/foto: key internal, atau None bila bukan FOUND."""
    if not field_result or field_result.get("status") != "FOUND":
        return None
    k = field_result.get("key")
    return tuple(k) if isinstance(k, list) else k


def strip_keys(result):
    """Salinan hasil extract() tanpa `key` internal (untuk respons JSON)."""
    out = dict(result)
    out["fields"] = {f: {k: v for k, v in r.items() if k != "key"} for f, r in result["fields"].items()}
    return out


# --------------------------------------------------------------------------
# Gabungan beberapa foto (tube: operator foto 2x, tiap foto dicari batch & EXP)
# --------------------------------------------------------------------------


def merge_photos(per_photo):
    """per_photo: list of {field: {"status": OK|LOW|..., "key":..., "view_keys": [...]}} (status
    per field sudah lewat konsensus view). Aturan, per field:
      - foto-foto OK harus bernilai sama; beda -> CONFLICT
      - foto yang TIDAK OK tapi membaca sesuatu: bertentangan bila TIDAK ADA satu view pun yang
        membaca nilai OK itu (foto itu konsisten melihat hal lain) -> CONFLICT. Satu view
        halusinasi di antara view yang membaca nilai OK tidak memblokir (uji 2026-11-19: foto
        batch PLB, satu view 640 px membaca 'FLR').
      - selain itu, ada foto OK -> OK; tidak ada -> LOW_CONFIDENCE
    Nilai tidak pernah digabung per karakter."""
    fields = sorted(set().union(*(p.keys() for p in per_photo))) if per_photo else []
    merged = {}
    for f in fields:
        rows = [(i, p.get(f) or {}) for i, p in enumerate(per_photo)]
        ok = [(i, r) for i, r in rows if r.get("status") == "OK"]
        ok_keys = {r.get("key") for _, r in ok}
        conflict = len(ok_keys) > 1
        if len(ok_keys) == 1:
            want = next(iter(ok_keys))
            for _, r in rows:
                vk = [k for k in (r.get("view_keys") or [r.get("key")]) if k is not None]
                if r.get("status") != "OK" and vk and want not in vk:
                    conflict = True
        elif not ok:
            conflict = len({r.get("key") for _, r in rows if r.get("key") is not None}) > 1
        if conflict:
            merged[f] = {"status": "CONFLICT", "value": None,
                         "per_photo": [{"photo": i, "status": r.get("status"), "value": r.get("value")} for i, r in rows]}
        elif ok:
            merged[f] = {**ok[0][1], "status": "OK", "photos": [j for j, _ in ok]}
        else:
            best = next((r for _, r in rows if r.get("value") is not None), {})
            merged[f] = {**best, "status": "LOW_CONFIDENCE", "photos": []}
    return merged


# --------------------------------------------------------------------------
# Usulan formula dari CONTOH coding (alat bantu QC / evaluasi; hasil WAJIB dikonfirmasi)
# --------------------------------------------------------------------------

_LABEL_WORDS = {"EXP", "MFG", "MFD", "ED", "BN", "BATCH", "NO", "DATE", "EX", "NOT", "LOT", "USE", "BEFORE"}
_NOTE_RE = re.compile(r"\([^)]*\)")


def _date_candidates(digits):
    """Tafsiran tanggal yang MUNGKIN untuk satu deret angka (tanpa memilih)."""
    n, out = len(digits), []
    if n == 6:
        d, m, y = int(digits[:2]), int(digits[2:4]), int(digits[4:])
        if 1 <= d <= 31 and 1 <= m <= 12:
            out.append(("{DD}{MM}{YY}", (2000 + y, m, d)))
        m2, y2 = int(digits[:2]), int(digits[2:])
        if 1 <= m2 <= 12 and 2000 <= y2 <= 2099:
            out.append(("{MM}{YYYY}", (y2, m2, None)))
    elif n == 8:
        d, m, y = int(digits[:2]), int(digits[2:4]), int(digits[4:])
        if 1 <= d <= 31 and 1 <= m <= 12 and 2000 <= y <= 2099:
            out.append(("{DD}{MM}{YYYY}", (y, m, d)))
    elif n == 4:
        m, y = int(digits[:2]), int(digits[2:])
        if 1 <= m <= 12:
            out.append(("{MM}{YY}", (2000 + y, m, None)))
    return out


def formula_from_example(example, exp_date=None):
    """Usulkan template dari contoh coding (mis. teks IM "AFA EXP 040627 XX NOT241102388K"
    atau isian operator). exp_date (date/datetime/'YYYY-MM-DD') hanya dipakai untuk memilih
    tafsiran 6 digit DDMMYY vs MMYYYY. -> {"segments": [...], "notes": [...]} atau
    {"segments": None, "notes": [...]} bila tidak bisa diusulkan tanpa menebak."""
    notes = []
    ref = None
    if exp_date:
        s = str(exp_date)[:10]
        ref = (int(s[:4]), int(s[5:7]), int(s[8:10]))
    text = _NOTE_RE.sub(" ", example.replace("\\n", "\n").replace("\r", ""))
    raw_lines = [l for l in re.split(r"\n|\s/\s|\s/|/\s", text)]
    segments = []
    date_fields_seen = 0
    for line in raw_lines:
        toks = re.findall(r"[A-Za-z]+|[0-9]+", line)
        out, i, has_field, label = [], 0, False, None
        while i < len(toks):
            t, up = toks[i], toks[i].upper()
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if up in _LABEL_WORDS:
                if up in ("EXP", "ED", "EX"):
                    label = "exp"
                elif up in ("MFG", "MFD"):
                    label = "mfg"
                out.append(up)
                i += 1
                continue
            if up in MONTH_NAMES and nxt and nxt.isdigit() and len(nxt) in (2, 4):
                field = label or ("mfg" if date_fields_seen == 0 and _two_dates(example) else "exp")
                tag = "" if field == "exp" else "@" + field
                out.append("{MMM%s} {%s%s}" % (tag, "YY" if len(nxt) == 2 else "YYYY", tag))
                has_field, date_fields_seen, i = True, date_fields_seen + 1, i + 2
                continue
            if t.isdigit() and re.fullmatch(r"[0-9]{2}[A-Z][0-9]{3}[A-Z]", "".join(toks[i:i + 4])) \
                    and re.search(re.escape("".join(toks[i:i + 4])), line):  # 26Y070X, menyatu
                out.append("{X7@batch}")
                has_field, i = True, i + 4
                continue
            if t.isdigit():
                field = label or ("mfg" if date_fields_seen == 0 and _two_dates(example) else "exp")
                tag = "" if field == "exp" else "@" + field
                if len(t) == 2 and nxt and nxt.isdigit() and len(nxt) == 4:  # 10 2029 / 04/2027
                    out.append("{MM%s} {YYYY%s}" % (tag, tag))
                    has_field, date_fields_seen, i = True, date_fields_seen + 1, i + 2
                    continue
                cands = _date_candidates(t)
                if field == "exp" and ref and len(cands) > 1:
                    cands = [c for c in cands if c[1][:2] == ref[:2] and c[1][2] in (None, ref[2])] or cands
                if len(cands) == 1:
                    out.append(re.sub(r"\{(\w+)\}", lambda mm: "{%s%s}" % (mm.group(1), tag), cands[0][0]))
                    has_field, date_fields_seen = True, date_fields_seen + 1
                elif len(cands) > 1:
                    notes.append(f"'{t}' bisa {' atau '.join(c[0] for c in cands)} - perlu konfirmasi QC")
                    return {"segments": None, "notes": notes}
                elif out and out[-1] == "NOT":
                    out[-1] = "NOT{D%d}" % len(t)
                elif len(t) >= 8:
                    notes.append(f"deret angka '{t}' diabaikan (bukan tanggal)")
                i += 1
                continue
            if len(t) == 3 and t.isupper() and up not in RESERVED_WORDS:
                out.append("{BATCH3}")
                has_field = True
            elif len(t) == 2 and out and re.search(r"(Y|M|MMM)\}$", out[-1]):
                out.append("[{INIT}]")
            elif len(t) == 1 and out and out[-1].startswith("NOT{D"):
                out[-1] += "{L1}"
            i += 1
        if has_field:
            segments.append(" ".join(out))
    if not segments:
        notes.append("tidak ada batch/tanggal yang dikenali")
        return {"segments": None, "notes": notes}
    return {"segments": segments, "notes": notes}


def _two_dates(example):
    """True bila contoh punya 2 tanggal TANPA label sama sekali (pola Lovetec: MFG lalu EXP)."""
    up = example.upper()
    if re.search(r"\b(EXP|ED|EX|MFG|MFD)\b", up) or re.search(r"\bEX[0-9]", up):
        return False
    return len(re.findall(r"[0-9]{2}\s*/\s*20[0-9]{2}|[0-9]{6,8}", up)) >= 2


# --------------------------------------------------------------------------
# Mode AUTO: gabungan semua formula katalog (bila formula SKU belum ada)
# --------------------------------------------------------------------------
# Permintaan user 2026-11-19: operator tidak boleh disuruh memilih formula. Di production
# ipc_app mengirim formula SKU; selama SKU belum punya formula (baru 6 IM terdokumentasi),
# pakai gabungan katalog. Ini BUKAN klasifikasi foto: tidak ada formula yang "dipilih" - semua
# dicoba sekaligus dan aturan AMBIGUOUS yang sama berlaku (dua tafsiran beda nilai -> REVIEW).
# Penyaringan lebih longgar daripada formula SKU -> respons diberi formula_source "auto".

AUTO = "auto"
CATALOG_PATH = __import__("pathlib").Path(__file__).with_name("coding_formulas.json")


def load_catalog(path=None):
    import json

    data = json.loads((path or CATALOG_PATH).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def auto_segments(catalog=None):
    """Semua segmen unik di katalog, urutan tetap."""
    seen, out = set(), []
    for item in (catalog or load_catalog()).values():
        for seg in item["segments"]:
            if seg not in seen:
                seen.add(seg)
                out.append(seg)
    return out
