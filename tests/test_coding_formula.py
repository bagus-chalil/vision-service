"""Unit test parser formula coding (tanpa GPU/Ollama). Jalankan: python tests/test_coding_formula.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import coding_formula as cf  # noqa: E402


def f(lines, formula):
    return cf.extract(lines, formula)["fields"]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --- contoh Format_Kodifikasi_IPC ---------------------------------------------------------
r = f(["AFA EXP 040627 XX NOT241102388K"], "{BATCH3} EXP {DD}{MM}{YY} {INIT} NOT{D9}{L1}")
check(r["batch"]["value"] == "AFA" and r["exp"]["date"] == {"year": 2027, "month": 6, "day": 4}, r)
check(r["exp"]["granularity"] == "day", r)

r = f(["AHB", "EXP AUG 29"], ["{BATCH3}", "EXP {MMM} {YY}"])
check(r["batch"]["value"] == "AHB" and r["exp"]["date"] == {"year": 2029, "month": 8, "day": None}, r)
check(r["exp"]["granularity"] == "month", r)

r = f(["EX0927 AJZ"], "EX{MM}{YY} {BATCH3}")
check(r["exp"]["value"] == "0927" and r["exp"]["date"]["month"] == 9 and r["batch"]["value"] == "AJZ", r)

r = f(["AHB", "08/2026", "04/2029"], "{BATCH3} {MM@mfg}/{YYYY@mfg} {MM}/{YYYY}")
check(r["mfg"]["date"]["year"] == 2026 and r["exp"]["date"] == {"year": 2029, "month": 4, "day": None}, r)

r = f(["MFG 04/2024 ADZ", "EXP 04/2027 XX"], ["MFG {MM@mfg}/{YYYY@mfg} {BATCH3}", "EXP {MM}/{YYYY} [{INIT}]"])
check(r["batch"]["value"] == "ADZ" and r["exp"]["date"]["year"] == 2027 and r["mfg"]["date"]["year"] == 2024, r)

# --- dua foto user 2026-11-19 (kode di badan tube, bukan di crimp) -------------------------
form = ["[BATCH NO|BATCH|BN] {BATCH3}", "[EXP DATE|EXP] {DD}{MM}{YY} [{INIT}]"]
r = f(["BATCH NO PLB", "EXP DATE 181129 TQ"], form)
check(r["batch"]["value"] == "PLB" and r["exp"]["date"] == {"year": 2029, "month": 11, "day": 18}, r)
r = f(["BATCH NO PL8", "EXP DATE 181129 TQ"], form)  # B terbaca 8 -> tidak dikoreksi
check(r["batch"]["status"] == "NOT_FOUND", r)
r = f(["8 997240 890787", "EXP NOV 29"], ["{BATCH3}", "EXP {MMM} {YY}"])  # barcode bukan tanggal
check(r["exp"]["date"] == {"year": 2029, "month": 11, "day": None} and r["batch"]["status"] == "NOT_FOUND", r)

# --- menyatu atau tidak --------------------------------------------------------------------
for s in ("EXP301029", "EXP 301029", "EXP. 30 10 29", "EXP:30/10/29", "EXP\n301029"):
    r = f([s], "EXP {DD}{MM}{YY}")
    check(r["exp"]["status"] == "FOUND" and r["exp"]["date"]["day"] == 30, (s, r))
r = f(["HKB / EXP 271029UI"], "{BATCH3} EXP {DD}{MM}{YY} [{INIT}]")
check(r["batch"]["value"] == "HKB" and r["exp"]["date"]["day"] == 27, r)

# --- DDMMYYYY dan MMYYYY ------------------------------------------------------------------
r = f(["EXP 29122026"], "EXP {DD}{MM}{YYYY}")
check(r["exp"]["date"] == {"year": 2026, "month": 12, "day": 29}, r)
r = f(["EXP 112028"], "EXP {MM}{YYYY}")
check(r["exp"]["date"] == {"year": 2028, "month": 11, "day": None}, r)
r = f(["EXP 112028"], "EXP {DD}{MM}{YY}")  # tafsiran salah -> ditolak, bukan dibetulkan
check(r["exp"]["status"] == "NOT_FOUND", r)

# --- jebakan lama --------------------------------------------------------------------------
# gotcha #15: 6 digit pertama kode 10-char bukan EXP
check(f(["310826QHB8"], "[EXP] {DD}{MM}{YY}")["exp"]["status"] == "NOT_FOUND", "gotcha15")
# gotcha #13/#18: MFD tanpa label EXP tidak boleh jadi EXP
r = f(["MFD 140624 8 QFZ", "EXP 140627"], "[EXP] {DD}{MM}{YY}")
check(r["exp"]["status"] == "FOUND" and r["exp"]["date"]["year"] == 2027, r)
r = f(["MFD 140624 8 QFZ"], "[EXP] {DD}{MM}{YY}")
check(r["exp"]["status"] == "NOT_FOUND", r)
# dua tanggal tanpa label & formula cuma EXP -> ambigu, tidak memilih
r = f(["08/2026", "04/2029"], "{MM}/{YYYY}")
check(r["exp"]["status"] == "AMBIGUOUS" and len(r["exp"]["candidates"]) == 2, r)
# tanggal tidak ada
check(f(["EXP 310229"], "EXP {DD}{MM}{YY}")["exp"]["status"] == "NOT_FOUND", "31 Feb")
# kata label/bulan bukan batch
check(f(["EXP NOV 29"], "{BATCH3}")["batch"]["status"] == "NOT_FOUND", "reserved")
# bacaan sama dari dua tempat -> tetap FOUND
r = f(["OKB EXP 231029 CW", "OKB"], ["{BATCH3} EXP {DD}{MM}{YY} [{INIT}]", "{BATCH3}"])
check(r["batch"]["status"] == "FOUND", r)
# batch panjang
check(f(["Depan 26Y070X"], "{X7@batch}")["batch"]["value"] == "26Y070X", "X7")

# --- formula salah -------------------------------------------------------------------------
for bad in ("{FOO}", "EXP [", "EXP"):
    try:
        cf.compile_formula(bad)
    except cf.FormulaError:
        pass
    else:
        raise AssertionError(f"harus FormulaError: {bad}")

# --- gabungan 2 foto -----------------------------------------------------------------------
a = {"batch": {"status": "OK", "key": "PLB", "value": "PLB"}, "exp": {"status": "LOW_CONFIDENCE", "key": None}}
b = {"batch": {"status": "LOW_CONFIDENCE", "key": None}, "exp": {"status": "OK", "key": (2029, 11, 18), "value": "181129"}}
m = cf.merge_photos([a, b])
check(m["batch"]["status"] == "OK" and m["exp"]["status"] == "OK" and m["exp"]["photos"] == [1], m)
c = {"batch": {"status": "LOW_CONFIDENCE", "key": "PLR", "view_keys": ["PLR", "PLR", None, "PLR"]},
     "exp": {"status": "OK", "key": (2029, 11, 18)}}
check(cf.merge_photos([a, c])["batch"]["status"] == "CONFLICT", "foto lain konsisten baca nilai lain")
d = {"batch": {"status": "LOW_CONFIDENCE", "key": "FLR", "view_keys": [None, "FLR", "PLB", "PLB"]}}
check(cf.merge_photos([a, d])["batch"]["status"] == "OK", "satu view halusinasi tidak memblokir")
e = {"exp": {"status": "OK", "key": (2029, 11, 19)}}
check(cf.merge_photos([b, e])["exp"]["status"] == "CONFLICT", "dua foto OK beda")

# --- usulan formula dari contoh -------------------------------------------------------------
cases = {
    "AFA EXP 040627 XX NOT241102388K": ["{BATCH3} EXP {DD}{MM}{YY} [{INIT}] NOT{D9}{L1}"],
    "OKB / EXP 281029": ["{BATCH3}", "EXP {DD}{MM}{YY}"],
    "EXP NOV 29": ["EXP {MMM} {YY}"],
    "EX0927 AJZ": ["EX {MM}{YY} {BATCH3}"],
}
for ex, want in cases.items():
    got = cf.formula_from_example(ex)["segments"]
    check(got == want, (ex, got))
# MMYYYY 6 digit selalu berawal MM lalu "20" -> sebagai DDMMYY bulannya 20 (tak valid), jadi
# tidak pernah benar-benar ambigu; tetap tercatat di sini supaya tidak di-"perbaiki" jadi DDMMYY.
check(cf.formula_from_example("EXP 112028")["segments"] == ["EXP {MM}{YYYY}"], "MMYYYY")
check(cf.formula_from_example("EXP 291029")["segments"] == ["EXP {DD}{MM}{YY}"], "DDMMYY")
check(cf.formula_from_example("AHB / 08/2026 / 04/2029")["segments"]
      == ["{BATCH3}", "{MM@mfg} {YYYY@mfg}", "{MM} {YYYY}"], cf.formula_from_example("AHB / 08/2026 / 04/2029"))
check(cf.formula_from_example("Depan : 26Y070X\\nBelakang : EXP 112028")["segments"]
      == ["{X7@batch}", "EXP {MM}{YYYY}"], "X7 dari contoh")
# --- katalog contoh: setiap 'contoh' harus terurai lengkap oleh formulanya ------------------
import json  # noqa: E402

catalog = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "coding_formulas.json"), encoding="utf-8"))
for key, item in catalog.items():
    if key.startswith("_"):
        continue
    for ex in item["contoh"]:
        out = cf.extract(ex.replace(" / ", "\n"), item["segments"])
        missing = [fl for fl in out["required_fields"] if out["fields"][fl]["status"] != "FOUND"
                   and not (fl == "batch" and "EXP 29122026" in ex or ex == "EXP NOV 29")]
        check(not missing, (key, ex, missing, out["fields"]))
print("coding_formula OK")
