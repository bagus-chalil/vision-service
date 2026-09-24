"""
Tube emboss code sub-pipeline (COSMAX IPC Vision Service).

Three stages, in order:
  1. YOLO11n localization (best.pt) -> crop to tube bbox -> crop again to the
     top ~15% of that crop (crimp seal area, where the date/batch code is
     embossed). If 0 tubes detected or confidence too low, stop before OCR.
  2. PaddleOCR recognition on the crimp crop via recognize_crimp_text_dual().
     Raw pass first; if its confidence is already below
     GEMINI_FALLBACK_THRESHOLD, stop there (LOW_CONFIDENCE either way, second
     pass would only add latency). Otherwise also run a sharpened/
     contrast-enhanced pass (preprocess_crimp_for_ocr()) and cross-validate:
     agreement -> trust it, disagreement -> confidence is capped so it always
     routes to LOW_CONFIDENCE rather than picking a side blindly. Reuses the
     PaddleOCR instance already loaded in main.py - do not construct a second
     one, model load is slow and doubles memory.
  3. Format validation against emboss_format_patterns.json. Same hard rule
     as the document-OCR pipeline: never strip/trim/guess-correct OCR text.
     A length or per-position charset mismatch always flags FORMAT_MISMATCH,
     regardless of confidence.

This module never decides PASS/FAIL/REVIEW - it only returns
text + confidence + status/flags. That decision lives in Laravel.
"""

import base64
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TUBE_DETECTOR_PATH = Path(__file__).parent / "models" / "tube_detector_v1" / "best.pt"
EMBOSS_FORMAT_PATTERNS_PATH = Path(__file__).parent / "emboss_format_patterns.json"
FALLBACK_LOG_PATH = Path(__file__).parent / "logs" / "fallback_cases.jsonl"

DEFAULT_FIELD_TYPE = "tube_emboss_default"

# Minimum YOLO detection confidence to trust the tube bbox at all. Below
# this, we don't even attempt OCR - route straight to REVIEW/RESCAN.
YOLO_MIN_CONFIDENCE = 0.25

# Crimp seal area = top fraction of the tube crop (emboss code lives on the
# tube shoulder/crimp, near the top of the cropped tube).
CRIMP_CROP_TOP_FRACTION = 0.15

# Separate from the document-OCR pipeline's 0.80 threshold (main.py) - this
# sub-pipeline's spec calls for 0.85.
GEMINI_FALLBACK_THRESHOLD = 0.85


def load_yolo_model(model_path: Path = TUBE_DETECTOR_PATH) -> YOLO:
    """Load once at startup and reuse across requests, same pattern as the
    PaddleOCR engine in main.py."""
    return YOLO(str(model_path))


def load_emboss_format_patterns() -> dict:
    # Re-read on every call (cheap, tiny file) so config edits take effect
    # without restarting uvicorn - matches load_field_patterns() in main.py.
    with open(EMBOSS_FORMAT_PATTERNS_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return {k: v for k, v in config.items() if not k.startswith("_")}


def _encode_debug_image(image_bgr) -> str:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def localize_tube(image_bgr, yolo_model, min_confidence: float = YOLO_MIN_CONFIDENCE) -> dict:
    """Stage 1: detect the tube, crop to its bbox, then crop to the top
    CRIMP_CROP_TOP_FRACTION of that crop (heuristic crimp-seal localization).
    """
    results = yolo_model.predict(source=image_bgr, verbose=False)
    boxes = results[0].boxes if results else None

    if boxes is None or len(boxes) == 0:
        return {"detected": False, "reason": "NO_TUBE_DETECTED", "confidence": None}

    confidences = boxes.conf.cpu().numpy()
    best_idx = int(confidences.argmax())
    best_confidence = float(confidences[best_idx])

    if best_confidence < min_confidence:
        return {
            "detected": False,
            "reason": "LOW_DETECTION_CONFIDENCE",
            "confidence": round(best_confidence, 4),
        }

    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().tolist()
    img_h, img_w = image_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))

    tube_crop = image_bgr[y1:y2, x1:x2]
    crimp_height = max(1, int(tube_crop.shape[0] * CRIMP_CROP_TOP_FRACTION))
    crimp_crop = tube_crop[0:crimp_height, :]

    return {
        "detected": True,
        "confidence": round(best_confidence, 4),
        "bbox": [x1, y1, x2, y2],
        "tube_crop": tube_crop,
        "crimp_crop": crimp_crop,
    }


def preprocess_crimp_for_ocr(crimp_crop_bgr):
    """Contrast-enhance + sharpen + upscale the crimp crop before OCR.
    Emboss text has low native contrast (raised metal, not printed ink) and
    the crimp crop is small (~200-280px tall) - this compensates for both.
    Pixel-only transform, runs before OCR sees the image - does not touch
    recognized text, so the never-strip/never-correct-text rule is untouched."""
    gray = cv2.cvtColor(crimp_crop_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    upscaled = cv2.resize(
        contrast_enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )

    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def recognize_crimp_text(crimp_crop, ocr_engine) -> dict:
    """Stage 2: PaddleOCR on the crimp crop. If PaddleOCR splits the code
    into multiple text detections, join them in reading order (top-to-bottom,
    left-to-right) - this only reorders/concatenates separate detections, it
    never edits characters within a detection."""
    results = ocr_engine.predict(input=crimp_crop)

    texts, scores, polys = [], [], []
    for res in results:
        texts.extend(res.get("rec_texts", []))
        scores.extend(res.get("rec_scores", []))
        polys.extend(res.get("rec_polys", []))

    if not texts:
        return {"detected": False}

    def centroid(poly):
        arr = np.array(poly)
        return (float(arr[:, 1].mean()), float(arr[:, 0].mean()))

    order = sorted(range(len(texts)), key=lambda i: centroid(polys[i]))
    combined_text = "".join(texts[i] for i in order)
    combined_confidence = min(float(scores[i]) for i in order)

    return {
        "detected": True,
        "text": combined_text,
        "confidence": round(combined_confidence, 4),
        "piece_count": len(texts),
    }


def recognize_crimp_text_dual(crimp_crop, ocr_engine) -> dict:
    """Runs recognize_crimp_text() twice - once on the raw crimp crop, once
    on preprocess_crimp_for_ocr()'s sharpened version - and cross-validates:

    - Both sides agree on the exact text -> trust it, confidence = the
      higher of the two (independent agreement is itself evidence).
    - Disagreement (including only one side detecting anything) -> never
      silently pick a side. Take whichever text has the higher confidence,
      but cap the returned confidence just under GEMINI_FALLBACK_THRESHOLD
      so analyze_tube_emboss() always routes it to LOW_CONFIDENCE (and the
      Gemini fallback, once implemented) instead of accepting a guess as OK.
    - Neither side detects anything -> not detected.

    This never edits characters within either OCR pass - it only chooses
    between two complete, untouched OCR outputs.

    Skips the sharpened pass entirely when the raw pass already scores below
    GEMINI_FALLBACK_THRESHOLD: at that confidence the request routes to
    LOW_CONFIDENCE regardless of what the sharpened pass says, so the second
    OCR call (the dominant cost of this function - see diagnose_sharpening_ab.py)
    would only add latency, never change the outcome. Measured on 18 real
    samples: this covers ~1/3 of requests with zero loss of the disagreement
    check's value, since the cases it actually catches (a confident misread
    that would otherwise slip through as OK) only occur when raw confidence
    is already >= threshold.
    """
    raw_result = recognize_crimp_text(crimp_crop, ocr_engine)
    raw_text = raw_result.get("text") if raw_result["detected"] else None
    raw_conf = raw_result.get("confidence") if raw_result["detected"] else None

    if raw_text is not None and raw_conf < GEMINI_FALLBACK_THRESHOLD:
        return {
            "detected": True,
            "text": raw_text,
            "confidence": raw_conf,
            "agreement": None,
            "raw_text": raw_text,
            "raw_confidence": raw_conf,
            "sharpened_text": None,
            "sharpened_confidence": None,
        }

    sharpened_crop = preprocess_crimp_for_ocr(crimp_crop)
    sharp_result = recognize_crimp_text(sharpened_crop, ocr_engine)
    sharp_text = sharp_result.get("text") if sharp_result["detected"] else None
    sharp_conf = sharp_result.get("confidence") if sharp_result["detected"] else None

    if raw_text is None and sharp_text is None:
        return {"detected": False}

    agreement = raw_text is not None and raw_text == sharp_text

    if agreement:
        return {
            "detected": True,
            "text": raw_text,
            "confidence": round(max(raw_conf, sharp_conf), 4),
            "agreement": True,
            "raw_text": raw_text,
            "raw_confidence": raw_conf,
            "sharpened_text": sharp_text,
            "sharpened_confidence": sharp_conf,
        }

    candidates = [(t, c) for t, c in ((raw_text, raw_conf), (sharp_text, sharp_conf)) if t is not None]
    best_text, best_conf = max(candidates, key=lambda tc: tc[1])
    capped_confidence = min(best_conf, GEMINI_FALLBACK_THRESHOLD - 0.01)

    return {
        "detected": True,
        "text": best_text,
        "confidence": round(capped_confidence, 4),
        "agreement": False,
        "raw_text": raw_text,
        "raw_confidence": raw_conf,
        "sharpened_text": sharp_text,
        "sharpened_confidence": sharp_conf,
    }


def gemini_vision_fallback_stub(image_bgr, field_type: str):
    """
    PLACEHOLDER - Gemini Vision API fallback is NOT wired up yet.

    TODO: implement the real call here (API key, request/response handling,
    error handling). Until then this always returns None, meaning "fallback
    unavailable" - the caller keeps the PaddleOCR result and engine_used
    stays "paddleocr".

    When implemented, must return the same shape as
    recognize_crimp_text()'s success case:
        {"detected": True, "text": str, "confidence": float}
    so it's a drop-in replacement at the call site in analyze_tube_emboss().
    """
    return None


def validate_emboss_format(text: str, format_cfg: dict) -> dict:
    """Stage 3: fixed-width block validation (see emboss_format_patterns.json
    for the 'blocks' schema). Never strips/trims/corrects text - a length
    mismatch or any block failing its check always flags FORMAT_MISMATCH,
    independent of OCR confidence.

    'int_range' blocks (day/month/year) check the chunk is numeric AND
    within [min, max] - e.g. day must be 1-31, not just "starts with 0-3",
    which would wrongly accept "39" or "00" as a valid-looking day.
    """
    blocks = format_cfg.get("blocks")
    if not blocks:
        return {
            "format_valid": None,
            "validation_status": "NOT_APPLICABLE",
            "expected_length": None,
            "actual_length": len(text),
            "block_mismatches": [],
        }

    expected_length = sum(b["width"] for b in blocks)
    if len(text) != expected_length:
        return {
            "format_valid": False,
            "validation_status": "FORMAT_MISMATCH",
            "expected_length": expected_length,
            "actual_length": len(text),
            "block_mismatches": [],
            "reason": f"expected {expected_length} chars, got {len(text)}",
        }

    mismatches = []
    pos = 0
    for block in blocks:
        width = block["width"]
        chunk = text[pos:pos + width]

        if block["type"] == "int_range":
            if not chunk.isdigit():
                mismatches.append({
                    "block": block["name"],
                    "value": chunk,
                    "detail": f"'{chunk}' is not numeric",
                })
            else:
                value = int(chunk)
                if not (block["min"] <= value <= block["max"]):
                    mismatches.append({
                        "block": block["name"],
                        "value": chunk,
                        "detail": f"'{chunk}' out of range {block['min']:02d}-{block['max']:02d}",
                    })
        elif block["type"] == "charset":
            pattern = block["pattern"]
            for offset, ch in enumerate(chunk):
                if re.fullmatch(pattern, ch) is None:
                    mismatches.append({
                        "block": block["name"],
                        "value": ch,
                        "detail": f"char '{ch}' at position {pos + offset} doesn't match {pattern}",
                    })

        pos += width

    return {
        "format_valid": len(mismatches) == 0,
        "validation_status": "FORMAT_OK" if not mismatches else "FORMAT_MISMATCH",
        "expected_length": expected_length,
        "actual_length": len(text),
        "block_mismatches": mismatches,
    }


def extract_date_blocks(text: str, format_cfg: dict) -> dict:
    """Pulls the day/month/year substrings out of OCR text using the
    field_type's own block widths - works for any tube_emboss field_type
    that defines day/month/year blocks (tube_emboss_default's MFD code and
    tube_exp_date's EXP-only code both do), ignoring any other blocks
    (batch/mfg) that may follow. Returns {} if the block config doesn't
    define all three - the caller decides what that means."""
    blocks = format_cfg.get("blocks") or []
    pos = 0
    found = {}
    for block in blocks:
        width = block["width"]
        if block["name"] in ("day", "month", "year"):
            found[block["name"]] = text[pos:pos + width]
        pos += width
    if not all(k in found for k in ("day", "month", "year")):
        return {}
    return found


def check_reference_date(raw_text, format_cfg: dict, format_valid, reference_date) -> dict:
    """Cross-checks the OCR'd day/month/year against a caller-supplied
    ground-truth date (e.g. QC-entered mixing date) - for testing whether
    OCR read the emboss correctly, NOT a business rule that EXP must equal
    that date. Exact string match only, same never-guess/never-fuzzy
    principle as validate_emboss_format(). Only runs when the format already
    passed block validation - a FORMAT_MISMATCH means the day/month/year
    slice positions aren't trustworthy to begin with, so comparing them
    would just be noise. Returns a data flag only; PASS/FAIL still never
    decided here."""
    if not reference_date:
        return {"checked": False, "reason": "NO_REFERENCE_DATE", "date_match": None, "reference_date": None, "ocr_date": None}

    if not (len(reference_date) == 6 and reference_date.isdigit()):
        return {"checked": False, "reason": "REFERENCE_DATE_NOT_DDMMYY", "date_match": None, "reference_date": reference_date, "ocr_date": None}

    if format_valid is not True:
        return {"checked": False, "reason": "FORMAT_MISMATCH", "date_match": None, "reference_date": reference_date, "ocr_date": None}

    date_blocks = extract_date_blocks(raw_text, format_cfg)
    if not date_blocks:
        return {"checked": False, "reason": "NO_DATE_BLOCKS_IN_FIELD_TYPE", "date_match": None, "reference_date": reference_date, "ocr_date": None}

    ocr_date = date_blocks["day"] + date_blocks["month"] + date_blocks["year"]
    return {
        "checked": True,
        "reason": None,
        "date_match": ocr_date == reference_date,
        "reference_date": reference_date,
        "ocr_date": ocr_date,
    }


def log_fallback_case(**fields) -> None:
    """Append a JSONL record for every LOW_CONFIDENCE or FORMAT_MISMATCH
    case - intentional future training data, not just debug noise."""
    FALLBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    with open(FALLBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def analyze_tube_emboss(
    image_bgr,
    *,
    field_type: str,
    request_id: str,
    ocr_engine,
    yolo_model,
    debug: bool = False,
    image_ref: str = "unknown",
    reference_date: str = None,
) -> dict:
    """Runs all 3 stages and returns the result object. Never raises for
    expected failure modes (no tube found, no text found) - those come back
    as status=ERROR with error_reason so the caller can route to
    REVIEW/RESCAN."""
    format_config = load_emboss_format_patterns()
    resolved_field_type = field_type if field_type in format_config else DEFAULT_FIELD_TYPE
    format_cfg = format_config.get(resolved_field_type, {})

    localization = localize_tube(image_bgr, yolo_model)

    if not localization["detected"]:
        result = {
            "request_id": request_id,
            "field_type": resolved_field_type,
            "raw_ocr_text": None,
            "confidence": None,
            "format_valid": None,
            "status": "ERROR",
            "error_reason": localization["reason"],
            "engine_used": None,
            "date_check": check_reference_date(None, format_cfg, None, reference_date),
        }
        if debug:
            result["debug"] = {
                "tube_detection_confidence": localization.get("confidence"),
                "note": "tube not localized - no crop available",
            }
        return result

    tube_crop = localization["tube_crop"]
    crimp_crop = localization["crimp_crop"]

    ocr_result = recognize_crimp_text_dual(crimp_crop, ocr_engine)

    if not ocr_result["detected"]:
        result = {
            "request_id": request_id,
            "field_type": resolved_field_type,
            "raw_ocr_text": None,
            "confidence": None,
            "format_valid": None,
            "status": "ERROR",
            "error_reason": "NO_TEXT_DETECTED",
            "engine_used": None,
            "tube_detection_confidence": localization["confidence"],
            "date_check": check_reference_date(None, format_cfg, None, reference_date),
        }
        if debug:
            result["debug"] = {
                "tube_bbox": localization["bbox"],
                "tube_crop_base64": _encode_debug_image(tube_crop),
                "crimp_crop_base64": _encode_debug_image(crimp_crop),
                "sharpened_crimp_crop_base64": _encode_debug_image(preprocess_crimp_for_ocr(crimp_crop)),
            }
        return result

    raw_text = ocr_result["text"]
    confidence = ocr_result["confidence"]
    ocr_agreement = ocr_result["agreement"]
    engine_used = "paddleocr"

    if confidence < GEMINI_FALLBACK_THRESHOLD:
        gemini_result = gemini_vision_fallback_stub(crimp_crop, resolved_field_type)
        if gemini_result is not None:
            raw_text = gemini_result["text"]
            confidence = gemini_result["confidence"]
            engine_used = "gemini_fallback"

    validation = validate_emboss_format(raw_text, format_cfg)
    format_valid = validation["format_valid"]
    status = "LOW_CONFIDENCE" if confidence < GEMINI_FALLBACK_THRESHOLD else "OK"

    if status == "LOW_CONFIDENCE" or format_valid is False:
        reasons = []
        if status == "LOW_CONFIDENCE":
            reasons.append("LOW_CONFIDENCE")
        if format_valid is False:
            reasons.append("FORMAT_MISMATCH")
        if ocr_agreement is False:
            reasons.append("OCR_DISAGREEMENT")
        log_fallback_case(
            request_id=request_id,
            field_type=resolved_field_type,
            image_ref=image_ref,
            raw_ocr_text=raw_text,
            confidence=confidence,
            engine_used=engine_used,
            status=status,
            format_valid=format_valid,
            reasons=reasons,
            validation=validation,
            ocr_agreement=ocr_agreement,
            ocr_raw_text=ocr_result.get("raw_text"),
            ocr_raw_confidence=ocr_result.get("raw_confidence"),
            ocr_sharpened_text=ocr_result.get("sharpened_text"),
            ocr_sharpened_confidence=ocr_result.get("sharpened_confidence"),
        )

    result = {
        "request_id": request_id,
        "field_type": resolved_field_type,
        "raw_ocr_text": raw_text,
        "confidence": confidence,
        "format_valid": format_valid,
        "status": status,
        "engine_used": engine_used,
        "ocr_agreement": ocr_agreement,
        "validation": validation,
        "tube_detection_confidence": localization["confidence"],
        "date_check": check_reference_date(raw_text, format_cfg, format_valid, reference_date),
    }

    if debug:
        result["debug"] = {
            "tube_bbox": localization["bbox"],
            "tube_crop_base64": _encode_debug_image(tube_crop),
            "crimp_crop_base64": _encode_debug_image(crimp_crop),
            "sharpened_crimp_crop_base64": _encode_debug_image(preprocess_crimp_for_ocr(crimp_crop)),
            "ocr_raw_text": ocr_result.get("raw_text"),
            "ocr_raw_confidence": ocr_result.get("raw_confidence"),
            "ocr_sharpened_text": ocr_result.get("sharpened_text"),
            "ocr_sharpened_confidence": ocr_result.get("sharpened_confidence"),
        }

    return result
