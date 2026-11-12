"""
LLM (Ollama, vision) extraction for document-type field_types: `label_bulk`
(tahap 1) and later `im_document` (tahap 2).

Design rules (see CLAUDE.md + docs/startup-ai-crosscheck-plan.md):
- Vision Service only READS. No PASS/FAIL, no date arithmetic, no month
  conversion - that all lives in ipc_app.
- Values are returned exactly as the model read them (`*_raw`). Never
  stripped/corrected here. Format checks are FLAGS only (`format_ok`).
- Unsure field = null + legible false. One retry only if the JSON is broken.
- Endpoint/model/thinking are env-configurable, nothing hardcoded:
    OLLAMA_URL            default http://127.0.0.1:11434
    OLLAMA_MODEL          default qwen3.8:27b
    OLLAMA_THINK          default false  (false | low | medium | high)
    OLLAMA_TIMEOUT        default 120    (seconds, per attempt)
    LABEL_VIEW_SIDES      default 1024,640,1600,2048 (image sizes = views; all must agree)
    IM_VIEW_SIDES / CODING_VIEW_SIDES  per-field override, fallback LABEL_VIEW_SIDES
    OLLAMA_NUM_CTX        default 8192

Plain import (sibling of main.py), same convention as tube_emboss_pipeline.py.
"""

import base64
import json
import os
import re
import time

import cv2
import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def _cfg():
    think_raw = os.environ.get("OLLAMA_THINK", "false").strip().lower()
    think = False if think_raw in ("", "false", "0", "off", "no") else think_raw
    return {
        "url": os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
        "model": os.environ.get("OLLAMA_MODEL", "qwen3.8:27b"),
        "think": think,
        "timeout": float(os.environ.get("OLLAMA_TIMEOUT", "120")),
        "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
    }


# --------------------------------------------------------------------------
# label_bulk: prompt + JSON schema
# --------------------------------------------------------------------------

LABEL_BULK_PROMPT = """You are reading a photo of a yellow "Label Bulk" sticker from a cosmetics factory.
Transcribe ONLY the three printed values below, exactly as they appear. Do not correct, reformat, translate or guess.

1. kode_bulk: the large printed number next to the label "Kode Bulk" (10 digits, e.g. 6587010023).
2. no_batch: the printed text in the row labelled "No Batch" (format like 26W137/MJB: digits+letter, a slash, then 3 letters).
3. mixing_date: the printed date in the row labelled "Mixing Date" (format DD/MM/YYYY).

Strict rules:
- Copy each value character by character, exactly as printed, including the "/" and any "0"/"O" you can see. Do not normalise.
- IGNORE everything else, especially: the handwritten "Exp. Date", handwritten notes/signatures/stamps, the white timestamp/weight sticker (e.g. "26/10/26 18:36:30 Weight: 298.8 KG"), the product name, Qty, No. Storage, Operator, PIC. The white sticker date is NOT the Mixing Date.
- If a value is not visible, cut off, blurred, covered, or you are not sure of every character, set that value to null and its legible flag to false. NEVER invent text that is not in the photo.
- Glare (bright white streaks), blur, smudges, folds or objects covering even ONE character of a value make the WHOLE value illegible: return null and legible=false for it. Do NOT fill the gap with a plausible character, and do NOT complete a value so that it matches the expected format. A wrong confident value is far worse than null.
- Judge each value separately: a clear Mixing Date does not make a glare-covered No Batch legible.
- legible.<field> is true only when you could read every character of that value with certainty.

Answer with JSON only, no explanation, matching exactly this shape:
{"kode_bulk": string|null, "no_batch": string|null, "mixing_date": string|null,
 "legible": {"kode_bulk": boolean, "no_batch": boolean, "mixing_date": boolean}}"""

LABEL_BULK_SCHEMA = {
    "type": "object",
    "properties": {
        "kode_bulk": {"type": ["string", "null"]},
        "no_batch": {"type": ["string", "null"]},
        "mixing_date": {"type": ["string", "null"]},
        "legible": {
            "type": "object",
            "properties": {
                "kode_bulk": {"type": "boolean"},
                "no_batch": {"type": "boolean"},
                "mixing_date": {"type": "boolean"},
            },
            "required": ["kode_bulk", "no_batch", "mixing_date"],
        },
    },
    "required": ["kode_bulk", "no_batch", "mixing_date", "legible"],
}

# Format checks are FLAGS only - a mismatch never edits the value.
NO_BATCH_FORMAT = re.compile(r"^[0-9]{2}[A-Za-z][0-9]{3}/[A-Za-z]{3}$")
MIXING_DATE_FORMAT = re.compile(r"^[0-9]{2}/[0-9]{2}/[0-9]{4}$")
KODE_BULK_FORMAT = re.compile(r"^[0-9]{10}$")


class LlmUnavailable(Exception):
    """Ollama unreachable / timeout / HTTP error (NOT a bad-JSON case)."""


class LlmBadJson(Exception):
    """Model answered but the answer is not usable JSON for the schema."""


# --------------------------------------------------------------------------
# Ollama call
# --------------------------------------------------------------------------


def _encode_image(image_bgr, max_side):
    h, w = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale != 1.0:  # up- AND down-scale: the size itself is the "view"
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _call_ollama(prompt, image_b64, schema, cfg):
    payload = {
        "model": cfg["model"],
        "stream": False,
        "think": cfg["think"],  # model default is medium -> must override
        "format": schema,  # structured output
        "options": {"temperature": 0, "seed": 0, "num_ctx": cfg["num_ctx"]},
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
    }
    try:
        resp = requests.post(
            f"{cfg['url']}/api/chat", json=payload, timeout=cfg["timeout"]
        )
        resp.raise_for_status()
        body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise LlmUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return body


def _parse_json(content):
    try:
        data = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise LlmBadJson(f"not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LlmBadJson("JSON root is not an object")
    return data


def _extract_with_retry(prompt, image_b64, schema, cfg, validate):
    """One attempt + exactly one retry if the JSON is broken/invalid.
    LlmUnavailable is NOT retried (it propagates)."""
    last_err = None
    attempts = 0
    for _ in range(2):
        attempts += 1
        body = _call_ollama(prompt, image_b64, schema, cfg)
        content = (body.get("message") or {}).get("content", "")
        try:
            data = _parse_json(content)
            validate(data)
            return data, attempts, body
        except LlmBadJson as exc:
            last_err = exc
    raise LlmBadJson(f"after {attempts} attempts: {last_err}")


# --------------------------------------------------------------------------
# label_bulk
# --------------------------------------------------------------------------


def _validate_label_bulk(data):
    for key in ("kode_bulk", "no_batch", "mixing_date"):
        if key not in data or not (data[key] is None or isinstance(data[key], str)):
            raise LlmBadJson(f"field {key} missing or wrong type")
    legible = data.get("legible")
    if not isinstance(legible, dict) or any(
        not isinstance(legible.get(k), bool)
        for k in ("kode_bulk", "no_batch", "mixing_date")
    ):
        raise LlmBadJson("legible missing or wrong type")


def _build_label_bulk_result(views, attempts, bodies, elapsed):
    """views = list of validated model answers, one per image size. The FIRST is
    the primary reading that gets returned; the rest only vote (agreement)."""
    keys = ("kode_bulk", "no_batch", "mixing_date")

    def norm(v):
        return v if (v is not None and v.strip()) else None

    primary = views[0]
    raw = {k: norm(primary[k]) for k in keys}
    legible = {k: bool(primary["legible"][k]) and raw[k] is not None for k in keys}

    # Multi-view consensus. Measured: on glare/blur photos the model completes
    # covered characters, says legible=true, and the guess CHANGES with the input
    # size; a genuinely clear value is identical at every size. So a value is only
    # trusted if EVERY view reads it identically (and each view called it legible).
    agree = {
        k: raw[k] is not None
        and all(norm(v[k]) == raw[k] and bool(v["legible"][k]) for v in views)
        for k in keys
    }

    format_ok = {
        "kode_bulk": bool(raw["kode_bulk"] and KODE_BULK_FORMAT.match(raw["kode_bulk"])),
        "no_batch": bool(raw["no_batch"] and NO_BATCH_FORMAT.match(raw["no_batch"])),
        "mixing_date": bool(
            raw["mixing_date"] and MIXING_DATE_FORMAT.match(raw["mixing_date"])
        ),
    }

    def trusted(k):
        return legible[k] and format_ok[k] and agree[k]

    critical = ("no_batch", "mixing_date")  # what ipc_app needs for the cross-check
    ok_fields = sum(1 for k in critical if trusted(k))
    if ok_fields == len(critical):
        status = "OK" if trusted("kode_bulk") else "LOW_CONFIDENCE"
        confidence = 0.95 if status == "OK" else 0.85
    else:
        status = "LOW_CONFIDENCE"
        confidence = 0.5 if ok_fields else 0.2

    return {
        "no_batch_raw": raw["no_batch"],
        "mixing_date_raw": raw["mixing_date"],
        "kode_bulk": raw["kode_bulk"],
        "legible": legible,
        "format_ok": format_ok,
        "view_agreement": agree,
        "views_read": [
            {k: v[k] for k in ("no_batch", "mixing_date", "kode_bulk")} for v in views
        ],
        # NOT a calibrated probability (LLMs have none): coarse routing signal
        # derived from legibility + format + multi-view agreement only.
        "confidence": confidence,
        "status": status,
        "engine_used": "ollama",
        "model": bodies[0].get("model"),
        "attempts": attempts,
        "elapsed_sec": round(elapsed, 2),
    }


def _error_result(field_type, status_detail, elapsed):
    return {
        "status": "ERROR",
        "error": status_detail,
        "confidence": 0.0,
        "engine_used": "ollama",
        "elapsed_sec": round(elapsed, 2),
        "field_type": field_type,
    }


def _view_sides(env_name="LABEL_VIEW_SIDES"):
    raw = os.environ.get(env_name) or os.environ.get("LABEL_VIEW_SIDES", "1024,640,1600,2048")
    return [int(x) for x in raw.split(",") if x.strip()]


# --------------------------------------------------------------------------
# im_document (tahap 2): nomor IM + umur simpan per kemasan
# --------------------------------------------------------------------------

IM_PROMPT = """You are reading a photo of a printed factory work instruction sheet ("IM", Instruksi Mutu) from a cosmetics factory. The photo may be tilted, blurry or partly cut off.
Transcribe ONLY these two things, exactly as printed. Do not correct, translate, reformat or guess.

1. im_number: the text after "No instruksi mutu:" in the yellow header box (e.g. "IM-KKI-LUXCRIME HYDRALIGHT CONCEALER W01 4.8G - V4").
2. shelf_life: in the section about the batch number format ("Format batch number ..."), each packaging row (e.g. Botol/Bottle, Unit Box, Tube, Mica, Bottom, Case) may carry a note in parentheses like "(3 Tahun dari Tanggal Mixing)" or "(1 Tahun 6 bulan dari Tanggal Mixing)". Return one entry per packaging row that has such a note:
   - packaging: the packaging label exactly as printed on that row.
   - raw_text: the note exactly as printed, including the parentheses.
   - years: the number printed before "Tahun" (integer), or null if no "Tahun" is printed.
   - months: the number printed before "bulan"/"Bulan" (integer), or null if no "bulan" is printed.
   Copy the numbers exactly as printed. Do NOT convert years to months, do NOT add, do NOT infer one packaging's value from another.

Strict rules:
- If a packaging row or its note is cut off, blurred, covered or you are not sure of every character, do NOT include that entry (or, for im_number, set it to null and legible.im_number=false). NEVER invent text that is not in the photo.
- If no shelf-life note is readable anywhere, return an empty shelf_life list and legible.shelf_life=false.
- legible.im_number / legible.shelf_life are true only when every character involved was read with certainty.

Answer with JSON only, matching exactly this shape:
{"im_number": string|null,
 "shelf_life": [{"packaging": string, "raw_text": string, "years": integer|null, "months": integer|null}],
 "legible": {"im_number": boolean, "shelf_life": boolean}}"""

IM_SCHEMA = {
    "type": "object",
    "properties": {
        "im_number": {"type": ["string", "null"]},
        "shelf_life": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "packaging": {"type": "string"},
                    "raw_text": {"type": "string"},
                    "years": {"type": ["integer", "null"]},
                    "months": {"type": ["integer", "null"]},
                },
                "required": ["packaging", "raw_text", "years", "months"],
            },
        },
        "legible": {
            "type": "object",
            "properties": {"im_number": {"type": "boolean"}, "shelf_life": {"type": "boolean"}},
            "required": ["im_number", "shelf_life"],
        },
    },
    "required": ["im_number", "shelf_life", "legible"],
}

# Flag only: IM number looks like "IM-<something>"; never edits the value.
IM_NUMBER_FORMAT = re.compile(r"^IM-\S.*$")
# raw_text must literally contain the claimed numbers (flag: model did not invent them)
_SHELF_YEARS = re.compile(r"(\d+)\s*Tahun", re.I)
_SHELF_MONTHS = re.compile(r"(\d+)\s*bulan", re.I)


def _validate_im(data):
    im = data.get("im_number")
    if "im_number" not in data or not (im is None or isinstance(im, str)):
        raise LlmBadJson("im_number missing or wrong type")
    sl = data.get("shelf_life")
    if not isinstance(sl, list):
        raise LlmBadJson("shelf_life missing or not a list")
    for e in sl:
        if not (
            isinstance(e, dict)
            and isinstance(e.get("packaging"), str)
            and isinstance(e.get("raw_text"), str)
            and (e.get("years") is None or isinstance(e.get("years"), int))
            and (e.get("months") is None or isinstance(e.get("months"), int))
        ):
            raise LlmBadJson("shelf_life entry malformed")
    legible = data.get("legible")
    if not isinstance(legible, dict) or any(
        not isinstance(legible.get(k), bool) for k in ("im_number", "shelf_life")
    ):
        raise LlmBadJson("legible missing or wrong type")


def _shelf_entry_ok(e):
    """Flag: numbers claimed must literally appear in raw_text (never edits)."""
    y = _SHELF_YEARS.search(e["raw_text"])
    m = _SHELF_MONTHS.search(e["raw_text"])
    return (
        (e["years"] is None) == (y is None)
        and (e["months"] is None) == (m is None)
        and (y is None or int(y.group(1)) == e["years"])
        and (m is None or int(m.group(1)) == e["months"])
        and (y is not None or m is not None)
    )


def _shelf_key(entries):
    """Order-independent, whitespace/case-insensitive signature for voting only."""
    return sorted(
        (re.sub(r"\s+", " ", e["packaging"]).strip().lower(), e["years"], e["months"])
        for e in entries
    )


def _build_im_result(views, attempts, bodies, elapsed):
    def norm(v):
        return v if (v is not None and v.strip()) else None

    primary = views[0]
    im_number = norm(primary["im_number"])
    shelf = primary["shelf_life"]
    legible = {
        "im_number": bool(primary["legible"]["im_number"]) and im_number is not None,
        "shelf_life": bool(primary["legible"]["shelf_life"]) and len(shelf) > 0,
    }
    # Umur simpan: yang menentukan EXP hanya NILAI (tahun/bulan), bukan label
    # kemasan dan bukan kelengkapan Botol/Unit Box (keputusan user 2026-11-11).
    # Dipercaya bila SEMUA view memberi himpunan nilai yang sama dan himpunan itu
    # TEPAT SATU nilai. Bila ada >1 nilai berbeda antar-kemasan (mis. Botol 3 Th,
    # Unit Box 2 Th) -> tidak dipercaya, ipc_app yang memutuskan (F4: REVIEW).
    def value_set(entries):
        return {(e["years"], e["months"]) for e in entries}

    # Keputusan user 2026-11-11: view yang mengembalikan daftar KOSONG = tidak
    # ikut memilih (abstain), bukan tidak sepakat. Syarat: minimal 3 view
    # (dan tidak lebih dari 1 view abstain) membaca, tiap view yang membaca
    # menyatakan legible, dan gabungan nilainya TEPAT SATU. View yang membaca
    # nilai BERBEDA tetap membatalkan (tidak pernah dianggap abstain).
    voting = [v for v in views if v["shelf_life"]]
    all_values = set().union(*(value_set(v["shelf_life"]) for v in voting)) if voting else set()
    agree = {
        "im_number": im_number is not None
        and all(norm(v["im_number"]) == im_number and v["legible"]["im_number"] for v in views),
        "shelf_life": len(voting) >= 3
        and len(views) - len(voting) <= 1
        and all(v["legible"]["shelf_life"] for v in voting)
        and len(all_values) == 1,
    }
    shelf_value = None
    if agree["shelf_life"]:
        years, months = next(iter(all_values))
        shelf_value = {"years": years, "months": months}  # apa adanya, tanpa konversi
        shelf = voting[0]["shelf_life"]  # entri dari view pertama yang membaca
        legible["shelf_life"] = True
    format_ok = {
        "im_number": bool(im_number and IM_NUMBER_FORMAT.match(im_number)),
        "shelf_life": len(shelf) > 0 and all(_shelf_entry_ok(e) for e in shelf),
    }
    trusted = {k: legible[k] and format_ok[k] and agree[k] for k in legible}
    if trusted["im_number"] and trusted["shelf_life"]:
        status, confidence = "OK", 0.95
    else:
        status = "LOW_CONFIDENCE"
        confidence = 0.5 if (trusted["im_number"] or trusted["shelf_life"]) else 0.2
    return {
        "im_number": im_number,
        "shelf_life": shelf,
        # nilai tunggal yang disepakati semua view (None bila tidak sepakat / >1 nilai)
        "shelf_life_value": shelf_value,
        "legible": legible,
        "format_ok": format_ok,
        "view_agreement": agree,
        "views_read": [{"im_number": v["im_number"], "shelf_life": v["shelf_life"]} for v in views],
        "confidence": confidence,
        "status": status,
        "engine_used": "ollama",
        "model": bodies[0].get("model"),
        "attempts": attempts,
        "elapsed_sec": round(elapsed, 2),
    }


def analyze_im_document(image_bgr):
    """im_document contract. Same multi-size consensus as label_bulk. Returns raw
    transcription only: years/months exactly as printed, NO conversion to months
    and no EXP math (that is ipc_app's job)."""
    cfg = _cfg()
    start = time.perf_counter()
    views, bodies, attempts = [], [], 0
    try:
        for side in _view_sides("IM_VIEW_SIDES"):
            image_b64 = _encode_image(image_bgr, side)
            data, n, body = _extract_with_retry(IM_PROMPT, image_b64, IM_SCHEMA, cfg, _validate_im)
            views.append(data)
            bodies.append(body)
            attempts += n
    except LlmUnavailable as exc:
        return _error_result("im_document", f"LLM_UNAVAILABLE: {exc}", time.perf_counter() - start)
    except LlmBadJson as exc:
        return _error_result("im_document", f"LLM_BAD_JSON: {exc}", time.perf_counter() - start)
    return _build_im_result(views, attempts, bodies, time.perf_counter() - start)


def analyze_label_bulk(image_bgr):
    """Returns the label_bulk contract dict. status: OK / LOW_CONFIDENCE /
    ERROR (LLM unavailable or JSON unusable after 1 retry - never a PASS).
    All inference runs on the Ollama server (GPU); nothing heavy runs locally."""
    cfg = _cfg()
    start = time.perf_counter()
    views, bodies, attempts = [], [], 0
    try:
        for side in _view_sides():
            image_b64 = _encode_image(image_bgr, side)
            data, n, body = _extract_with_retry(
                LABEL_BULK_PROMPT, image_b64, LABEL_BULK_SCHEMA, cfg, _validate_label_bulk
            )
            views.append(data)
            bodies.append(body)
            attempts += n
    except LlmUnavailable as exc:
        return _error_result("label_bulk", f"LLM_UNAVAILABLE: {exc}", time.perf_counter() - start)
    except LlmBadJson as exc:
        return _error_result("label_bulk", f"LLM_BAD_JSON: {exc}", time.perf_counter() - start)
    return _build_label_bulk_result(views, attempts, bodies, time.perf_counter() - start)


# --------------------------------------------------------------------------
# coding_print: foto coding NON-TUBE (batch + EXP tercetak di kotak/jar/tutup/
# botol/label). Tube emboss tetap lewat tube_emboss_pipeline.
# --------------------------------------------------------------------------

CODING_PROMPT = """You are reading a photo taken on a cosmetics production line. Somewhere on the product, its box, cap, jar or a sticker there may be a machine-printed production CODING (inkjet, laser, thermal-transfer or a printed label) that states the batch code and the expiry date. Typical layouts:
  "OKB EXP 231029 CW"   (3-letter batch, EXP, 6-digit date DDMMYY, 2-letter initials)
  "AKB" on one line and "EXP 191029 QW" on the next line
  "BATCH : BKB" / "EXP DATE : 301029 MH"
  "Batch No: EKB" / "Mfg Date: 201026" / "Exp Date: 201029"
  "AKB EXP 10 2029 GK"  (expiry as month + 4-digit year)

Transcribe exactly as printed, character by character. Do not correct, reformat or guess.
1. batch: the batch code only (the code after "BATCH"/"Batch No"/"BN", or the code printed directly before "EXP" on the same coding). Usually 3 letters.
2. exp: ONLY the expiry date characters that follow "EXP"/"EXP DATE"/"Exp Date", keeping spaces/dots/slashes exactly as printed (e.g. "231029", "10 2029"). Do NOT include the label "EXP" and do NOT include the trailing initials (e.g. "CW").
3. exp_suffix: the short code printed right after the expiry date on the same line (e.g. "CW"), or null if none.
4. packaging_type: the kind of product container that is the main subject of the photo:
   - "tube": a squeezable, flexible plastic/laminate tube whose end is a flat, pressed (crimp-sealed) strip, usually with a screw or flip cap at the other end.
   - "non_tube": anything else that is a product container: carton box, jar, bottle, cap/lid, lipstick or stick, compact, pump, pouch/sachet.
   - "unknown": no product container visible (only paper/forms, a blank/dark frame, a pallet), or you cannot tell.

Strict rules:
- Read ONLY machine-printed coding. IGNORE all handwriting (pen/marker on forms or paper), even if it looks like a coding.
- IGNORE manufacturing/production dates (Mfg Date, MFD, Prod), NA/BPOM numbers, barcodes, ingredient text and product names.
- If two or more printed codings with DIFFERENT values are visible, set both values to null and legible false.
- If there is no printed coding in the photo, return null for every value and legible false.
- Glare, blur, small size, cut-off, smudge or anything covering even ONE character makes the WHOLE value illegible: return null and legible=false. Never complete a value so it fits the expected format. A wrong confident value is far worse than null.
- legible.<field> is true only if you read every character of that value with certainty.

Answer with JSON only:
{"batch": string|null, "exp": string|null, "exp_suffix": string|null, "packaging_type": "tube"|"non_tube"|"unknown",
 "legible": {"batch": boolean, "exp": boolean}}"""

CODING_SCHEMA = {
    "type": "object",
    "properties": {
        "batch": {"type": ["string", "null"]},
        "exp": {"type": ["string", "null"]},
        "exp_suffix": {"type": ["string", "null"]},
        "packaging_type": {"type": "string", "enum": ["tube", "non_tube", "unknown"]},
        "legible": {
            "type": "object",
            "properties": {"batch": {"type": "boolean"}, "exp": {"type": "boolean"}},
            "required": ["batch", "exp"],
        },
    },
    "required": ["batch", "exp", "exp_suffix", "packaging_type", "legible"],
}

CODING_LOCATE_PROMPT = """Find the machine-printed production coding in this photo (the printed text with a batch code and "EXP" expiry date, e.g. "OKB EXP 231029 CW" or "BATCH : BKB / EXP DATE : 301029"). Ignore handwriting, barcodes, ingredient text.
Return its bounding box covering ALL lines of that coding, as integers normalised to 0-1000 of the image width/height: [x1, y1, x2, y2].
If there is no printed coding, set found to false.
JSON only: {"found": boolean, "box": [x1, y1, x2, y2] | null}"""

CODING_LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "box": {"type": ["array", "null"], "items": {"type": "integer"}},
    },
    "required": ["found", "box"],
}

# Flags only - a mismatch never edits the value.
CODING_BATCH_FORMAT = re.compile(r"^[A-Za-z]{3}$")
CODING_EXP_DDMMYY = re.compile(r"^(0[1-9]|[12][0-9]|3[01])(0[1-9]|1[0-2])([0-9]{2})$")
CODING_EXP_MM_YYYY = re.compile(r"^(0[1-9]|1[0-2])[ ./-](20[0-9]{2})$")
PACKAGING_TYPES = ("tube", "non_tube", "unknown")


def _validate_coding(data):
    for key in ("batch", "exp", "exp_suffix"):
        if key not in data or not (data[key] is None or isinstance(data[key], str)):
            raise LlmBadJson(f"field {key} missing or wrong type")
    if data.get("packaging_type") not in PACKAGING_TYPES:
        raise LlmBadJson("packaging_type missing or invalid")
    legible = data.get("legible")
    if not isinstance(legible, dict) or any(
        not isinstance(legible.get(k), bool) for k in ("batch", "exp")
    ):
        raise LlmBadJson("legible missing or wrong type")


def _validate_coding_locate(data):
    if not isinstance(data.get("found"), bool):
        raise LlmBadJson("found missing")
    box = data.get("box")
    if data["found"] and not (
        isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box)
    ):
        raise LlmBadJson("box malformed")


def coding_exp_format(exp_raw):
    """Classify the printed EXP string WITHOUT changing it. exp_parts are plain
    substrings of exp_raw (no conversion, no date math - that is ipc_app's job).
    Returns (exp_format, exp_parts): DDMMYY / MM_YYYY / UNKNOWN / None."""
    if not exp_raw:
        return None, None
    m = CODING_EXP_DDMMYY.match(exp_raw)
    if m:
        return "DDMMYY", {"day": m.group(1), "month": m.group(2), "year": m.group(3)}
    m = CODING_EXP_MM_YYYY.match(exp_raw)
    if m:
        return "MM_YYYY", {"day": None, "month": m.group(1), "year": m.group(2)}
    return "UNKNOWN", None


def _coding_consensus(views):
    """Same multi-size consensus as label_bulk: a value is trusted only when
    EVERY view reads it identically and calls it legible. Never majority vote -
    measured (2026-11-11): on a small print 3 of 4 views agreed on a WRONG digit."""

    def norm(v):
        return v.strip() if (v is not None and v.strip()) else None

    primary = views[0]
    raw = {k: norm(primary[k]) for k in ("batch", "exp")}
    legible = {k: bool(primary["legible"][k]) and raw[k] is not None for k in raw}
    agree = {
        k: raw[k] is not None
        and all(norm(v[k]) == raw[k] and bool(v["legible"][k]) for v in views)
        for k in raw
    }
    exp_format, exp_parts = coding_exp_format(raw["exp"])
    format_ok = {
        "batch": bool(raw["batch"] and CODING_BATCH_FORMAT.match(raw["batch"])),
        "exp": exp_format in ("DDMMYY", "MM_YYYY"),
    }
    trusted = {k: legible[k] and agree[k] and format_ok[k] for k in raw}
    kinds = [v["packaging_type"] for v in views]
    return {
        "batch_raw": raw["batch"],
        "exp_raw": raw["exp"],
        "exp_suffix": norm(primary["exp_suffix"]),
        "exp_format": exp_format,
        "exp_parts": exp_parts,
        # Flag only: never used to route/choose a pipeline (field_type comes from
        # the caller - CLAUDE.md design principle). Views disagree -> "unknown".
        "packaging_type": kinds[0] if len(set(kinds)) == 1 else "unknown",
        "legible": legible,
        "format_ok": format_ok,
        "view_agreement": agree,
        "trusted": trusted,
        "views_read": [
            {"batch": v["batch"], "exp": v["exp"], "packaging_type": v["packaging_type"]}
            for v in views
        ],
    }


def _read_coding_views(image_bgr, cfg):
    views, bodies, attempts = [], [], 0
    for side in _view_sides("CODING_VIEW_SIDES"):
        data, n, body = _extract_with_retry(
            CODING_PROMPT, _encode_image(image_bgr, side), CODING_SCHEMA, cfg, _validate_coding
        )
        views.append(data)
        bodies.append(body)
        attempts += n
    return views, bodies, attempts


def _crop_normalised_box(image_bgr, box, pad=0.6, min_frac=0.12):
    """Crop around a 0-1000 normalised box with generous padding (never tighter
    than min_frac of the frame) so a slightly-off box still contains the coding."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [max(0, min(1000, v)) for v in box]
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = (x2 - x1) / 1000 * w, (y2 - y1) / 1000 * h
    cx, cy = (x1 + x2) / 2000 * w, (y1 + y2) / 2000 * h
    half_w = max(bw * (1 + pad), w * min_frac) / 2
    half_h = max(bh * (1 + pad * 2), h * min_frac) / 2
    a, b = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    c, d = int(min(w, cx + half_w)), int(min(h, cy + half_h))
    crop = image_bgr[b:d, a:c]
    return crop if crop.size else None


def analyze_coding_print(image_bgr):
    """coding_print contract (non-tube batch + EXP). Raw transcription only:
    no PASS/FAIL, no date math, never strips/corrects. Two passes:
      1. whole photo, 4-size consensus.
      2. ONLY if pass 1 is not OK ("rescue"): the model points at the coding,
         it is cropped + re-read with the same 4-size consensus. The rescued value
         is accepted only if pass 1 already read exactly that value in at least one
         view (independent corroboration). A pass-1 OK is never touched, so the
         rescue can only turn LOW into OK, never change an OK value.
    Measured 2026-11-11 (76 foto, scratchpad prototype): 44 OK all correct,
    0 false OK; rescue added 5 of those."""
    cfg = _cfg()
    start = time.perf_counter()
    try:
        views, bodies, attempts = _read_coding_views(image_bgr, cfg)
        result = _coding_consensus(views)
        result["read_via"] = "full_frame"
        rescue_info = None
        if not all(result["trusted"].values()):
            loc, n, _ = _extract_with_retry(
                CODING_LOCATE_PROMPT,
                _encode_image(image_bgr, 1024),
                CODING_LOCATE_SCHEMA,
                cfg,
                _validate_coding_locate,
            )
            attempts += n
            crop = _crop_normalised_box(image_bgr, loc["box"]) if loc["found"] and loc["box"] else None
            rescue_info = {"located": crop is not None, "box": loc.get("box")}
            if crop is not None:
                c_views, _, n = _read_coding_views(crop, cfg)
                attempts += n
                rescued = _coding_consensus(c_views)
                pass1_values = {(v["batch"], v["exp"]) for v in views}
                corroborated = (rescued["batch_raw"], rescued["exp_raw"]) in pass1_values
                rescue_info.update(
                    status="OK" if all(rescued["trusted"].values()) else "LOW_CONFIDENCE",
                    batch_raw=rescued["batch_raw"],
                    exp_raw=rescued["exp_raw"],
                    corroborated_by_pass1=corroborated,
                    views_read=rescued["views_read"],
                )
                if all(rescued["trusted"].values()) and corroborated:
                    keep_pkg = result["packaging_type"]  # crop says nothing about the container
                    result = rescued
                    result["packaging_type"] = keep_pkg
                    result["read_via"] = "crop_rescue"
        result["rescue"] = rescue_info
    except LlmUnavailable as exc:
        return _error_result("coding_print", f"LLM_UNAVAILABLE: {exc}", time.perf_counter() - start)
    except LlmBadJson as exc:
        return _error_result("coding_print", f"LLM_BAD_JSON: {exc}", time.perf_counter() - start)

    if all(result["trusted"].values()):
        status, confidence = "OK", 0.95
    else:
        status = "LOW_CONFIDENCE"
        confidence = 0.5 if any(result["trusted"].values()) else 0.2
    result.update(
        # NOT a calibrated probability: routing signal from legibility + format +
        # multi-view agreement only.
        confidence=confidence,
        status=status,
        engine_used="ollama",
        model=bodies[0].get("model"),
        attempts=attempts,
        elapsed_sec=round(time.perf_counter() - start, 2),
    )
    return result
