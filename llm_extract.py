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
