"""
Tube emboss code sub-pipeline (COSMAX IPC Vision Service).

Three stages, in order:
  1. YOLO11n localization (best.pt) -> crop to tube bbox -> crop again to the
     top ~15% of that crop (crimp seal area, where the date/batch code is
     embossed). If 0 tubes detected or confidence too low, stop before OCR.
  2. PaddleOCR recognition on the crimp crop (reuses the PaddleOCR instance
     already loaded in main.py - do not construct a second one, model load
     is slow and doubles memory).
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
        }
        if debug:
            result["debug"] = {
                "tube_detection_confidence": localization.get("confidence"),
                "note": "tube not localized - no crop available",
            }
        return result

    tube_crop = localization["tube_crop"]
    crimp_crop = localization["crimp_crop"]

    ocr_result = recognize_crimp_text(crimp_crop, ocr_engine)

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
        }
        if debug:
            result["debug"] = {
                "tube_bbox": localization["bbox"],
                "tube_crop_base64": _encode_debug_image(tube_crop),
                "crimp_crop_base64": _encode_debug_image(crimp_crop),
            }
        return result

    raw_text = ocr_result["text"]
    confidence = ocr_result["confidence"]
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
        )

    result = {
        "request_id": request_id,
        "field_type": resolved_field_type,
        "raw_ocr_text": raw_text,
        "confidence": confidence,
        "format_valid": format_valid,
        "status": status,
        "engine_used": engine_used,
        "validation": validation,
        "tube_detection_confidence": localization["confidence"],
    }

    if debug:
        result["debug"] = {
            "tube_bbox": localization["bbox"],
            "tube_crop_base64": _encode_debug_image(tube_crop),
            "crimp_crop_base64": _encode_debug_image(crimp_crop),
        }

    return result
