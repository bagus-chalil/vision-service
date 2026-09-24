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

field_types that set "label_anchor" in emboss_format_patterns.json (currently
tube_exp_date -> "EXP") skip steps 1-2 above and use a different, more
robust path instead (recognize_label_anchor_dual() / find_label_anchor_match()):
OCR the whole tube crop (or the whole frame if no tube was detected at all -
real EXP codes also show up on bottle caps, not just tube crimps) and search
each individually-detected text region for the label followed by 6 digits,
rather than cropping to a fixed pixel percentage first and joining everything
inside it into one blob. Added 2026-09-24 after real samples showed the old
top-15%-of-tube-bbox crop was framing-dependent - it pulled in the wrong
emboss line (MFD instead of EXP) or trailing printed body text depending on
how close the tube was held to the camera, which the label anchor sidesteps
by finding the code from its own printed label instead of its pixel position.

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

# Anchor-based extraction for field_types with "label_anchor" set in
# emboss_format_patterns.json (currently tube_exp_date -> "EXP"). Matches the
# label optionally followed by "." or ":" and/or whitespace, then exactly 6
# digits, e.g. "EXP 210428", "EXP.150728", "EXP:210428". Built from real
# sample photos (2026-09-24): EXP appears on tube crimps AND on bottle caps
# and printed labels, sometimes preceded by an unrelated batch/lot code
# ("FJZ EXP.130927") and/or followed by a short suffix code ("EXP.150728 TU")
# - re.search (not fullmatch) anchored on the literal label is what lets this
# survive that surrounding noise without ever touching the 6 matched digits.
LABEL_ANCHOR_PATTERN_TEMPLATE = r"{label}\.?\s*:?\s*(\d{{6}})"


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


# Only upscale inputs shorter than this (px) - see preprocess_crimp_for_ocr().
UPSCALE_MAX_INPUT_HEIGHT = 400


def preprocess_crimp_for_ocr(crimp_crop_bgr):
    """Contrast-enhance + sharpen (+ upscale, for small crops only) before
    OCR. Emboss text has low native contrast (raised metal, not printed ink)
    and the crimp crop is small (~200-280px tall) - this compensates for
    both. Pixel-only transform, runs before OCR sees the image - does not
    touch recognized text, so the never-strip/never-correct-text rule is
    untouched.

    The 2x upscale is skipped above UPSCALE_MAX_INPUT_HEIGHT: it's only
    needed to give a tiny crimp crop enough pixels for OCR to work with.
    Measured 2026-09-24 on the label_anchor path's whole-tube/whole-frame
    fallback (see analyze_label_anchor_field()) - a 1280x720 input upscaled
    to 2560x1440 took ~28s to OCR vs ~8s unscaled, a cost this preprocessing
    was never designed to pay and printed ink text (unlike embossed metal)
    doesn't need anyway."""
    gray = cv2.cvtColor(crimp_crop_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    if contrast_enhanced.shape[0] <= UPSCALE_MAX_INPUT_HEIGHT:
        upscaled = cv2.resize(
            contrast_enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
        )
    else:
        upscaled = contrast_enhanced

    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def ocr_text_pieces(image_bgr, ocr_engine) -> list:
    """Runs PaddleOCR once and returns each detected text region as its own
    piece {"text", "confidence"}, in reading order (top-to-bottom,
    left-to-right by box centroid) - no joining/concatenation. Shared by
    recognize_crimp_text() (which joins pieces back into one code) and the
    label-anchor search (which must evaluate each detected region on its own,
    see find_label_anchor_match()) so a piece of unrelated printed text next
    to the code can never get glued onto it."""
    results = ocr_engine.predict(input=image_bgr)

    texts, scores, polys = [], [], []
    for res in results:
        texts.extend(res.get("rec_texts", []))
        scores.extend(res.get("rec_scores", []))
        polys.extend(res.get("rec_polys", []))

    if not texts:
        return []

    def centroid(poly):
        arr = np.array(poly)
        return (float(arr[:, 1].mean()), float(arr[:, 0].mean()))

    order = sorted(range(len(texts)), key=lambda i: centroid(polys[i]))
    return [{"text": texts[i], "confidence": float(scores[i])} for i in order]


def recognize_crimp_text(crimp_crop, ocr_engine) -> dict:
    """Stage 2: PaddleOCR on the crimp crop. If PaddleOCR splits the code
    into multiple text detections, join them in reading order (top-to-bottom,
    left-to-right) - this only reorders/concatenates separate detections, it
    never edits characters within a detection."""
    pieces = ocr_text_pieces(crimp_crop, ocr_engine)
    if not pieces:
        return {"detected": False}

    combined_text = "".join(p["text"] for p in pieces)
    combined_confidence = min(p["confidence"] for p in pieces)

    return {
        "detected": True,
        "text": combined_text,
        "confidence": round(combined_confidence, 4),
        "piece_count": len(pieces),
    }


def find_label_anchor_match(pieces: list, label: str) -> dict:
    """Searches each OCR piece independently (never a joined blob - see
    ocr_text_pieces()) for `label` followed by exactly 6 digits, e.g. "EXP" ->
    matches "EXP 210428" within a piece that also contains other text before
    or after. re.search only locates where the 6-digit code starts within a
    piece that already exists as one atomic OCR detection - it never edits,
    strips, or guesses at the digits themselves, so it doesn't violate the
    never-auto-correct-OCR-text rule.

    Returns {"matched": False} if no piece contains the label, or
    {"matched": False, "ambiguous": True, "candidates": [...]} if multiple
    pieces match with DIFFERENT 6-digit codes - never silently pick one.
    Pieces matching with the same code (e.g. duplicate detections) resolve to
    the single highest-confidence match.
    """
    pattern = re.compile(LABEL_ANCHOR_PATTERN_TEMPLATE.format(label=re.escape(label)), re.IGNORECASE)
    hits = []
    for piece in pieces:
        m = pattern.search(piece["text"])
        if m:
            hits.append({
                "matched_text": piece["text"],
                "extracted_digits": m.group(1),
                "confidence": piece["confidence"],
            })

    if not hits:
        return {"matched": False}

    distinct_codes = {h["extracted_digits"] for h in hits}
    if len(distinct_codes) > 1:
        return {"matched": False, "ambiguous": True, "candidates": hits}

    best = max(hits, key=lambda h: h["confidence"])
    return {
        "matched": True,
        "matched_text": best["matched_text"],
        "extracted_digits": best["extracted_digits"],
        "confidence": round(best["confidence"], 4),
    }


def recognize_label_anchor_dual(image_bgr, ocr_engine, label: str) -> dict:
    """Label-anchor equivalent of recognize_crimp_text_dual(): runs the
    anchor search on the raw image, and - unless that raw pass already
    resolved with a confident match - also on preprocess_crimp_for_ocr()'s
    sharpened version, then cross-validates the two extracted 6-digit codes.
    Agreement -> trust it (confidence = higher of the two). Disagreement
    (including only one side matching at all) -> never silently pick a side;
    take the higher-confidence match but cap the returned confidence just
    under GEMINI_FALLBACK_THRESHOLD so the caller always routes it to
    LOW_CONFIDENCE. Neither side matches -> not detected (caller reports
    FORMAT_MISMATCH / label not found)."""
    raw_pieces = ocr_text_pieces(image_bgr, ocr_engine)
    raw_match = find_label_anchor_match(raw_pieces, label)

    if raw_match.get("matched") and raw_match["confidence"] >= GEMINI_FALLBACK_THRESHOLD:
        return {
            "detected": True,
            "text": raw_match["matched_text"],
            "extracted_digits": raw_match["extracted_digits"],
            "confidence": raw_match["confidence"],
            "agreement": None,
            "all_detected_text": "".join(p["text"] for p in raw_pieces),
        }

    sharpened = preprocess_crimp_for_ocr(image_bgr)
    sharp_pieces = ocr_text_pieces(sharpened, ocr_engine)
    sharp_match = find_label_anchor_match(sharp_pieces, label)

    all_detected_text = "".join(p["text"] for p in raw_pieces) or "".join(p["text"] for p in sharp_pieces)

    if not raw_match.get("matched") and not sharp_match.get("matched"):
        return {
            "detected": False,
            "all_detected_text": all_detected_text,
            "ambiguous": raw_match.get("ambiguous", False) or sharp_match.get("ambiguous", False),
            "candidates": raw_match.get("candidates") or sharp_match.get("candidates") or [],
        }

    agreement = (
        raw_match.get("matched") and sharp_match.get("matched")
        and raw_match["extracted_digits"] == sharp_match["extracted_digits"]
    )

    if agreement:
        return {
            "detected": True,
            "text": raw_match["matched_text"],
            "extracted_digits": raw_match["extracted_digits"],
            "confidence": round(max(raw_match["confidence"], sharp_match["confidence"]), 4),
            "agreement": True,
            "all_detected_text": all_detected_text,
        }

    candidates = [m for m in (raw_match, sharp_match) if m.get("matched")]
    best = max(candidates, key=lambda m: m["confidence"])
    capped_confidence = min(best["confidence"], GEMINI_FALLBACK_THRESHOLD - 0.01)

    return {
        "detected": True,
        "text": best["matched_text"],
        "extracted_digits": best["extracted_digits"],
        "confidence": round(capped_confidence, 4),
        "agreement": False,
        "all_detected_text": all_detected_text,
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


def analyze_label_anchor_field(
    image_bgr,
    *,
    label_anchor: str,
    format_cfg: dict,
    resolved_field_type: str,
    request_id: str,
    ocr_engine,
    yolo_model,
    debug: bool = False,
    image_ref: str = "unknown",
    reference_date: str = None,
) -> dict:
    """Anchor-based path for field_types with "label_anchor" set (see module
    docstring). Tries YOLO tube localization first to cut out background
    clutter, but - unlike analyze_tube_emboss()'s default path - doesn't hard
    fail when no tube is detected: real EXP samples also come from bottle
    caps and printed labels that the tube detector never will (and was never
    trained to) recognize, so this falls back to searching the whole frame."""
    localization = localize_tube(image_bgr, yolo_model)
    search_image = localization["tube_crop"] if localization["detected"] else image_bgr
    tube_detection_confidence = localization.get("confidence")

    ocr_result = recognize_label_anchor_dual(search_image, ocr_engine, label_anchor)

    if not ocr_result["detected"]:
        reason = "LABEL_AMBIGUOUS" if ocr_result.get("ambiguous") else "LABEL_NOT_FOUND"
        raw_text = ocr_result.get("all_detected_text") or None
        validation = {
            "format_valid": False,
            "validation_status": "FORMAT_MISMATCH",
            "expected_length": None,
            "actual_length": len(raw_text) if raw_text else 0,
            "block_mismatches": [],
            "reason": f"label '{label_anchor}' not found in OCR text" if reason == "LABEL_NOT_FOUND"
                      else f"multiple '{label_anchor}' matches with different dates - see candidates",
        }
        log_fallback_case(
            request_id=request_id,
            field_type=resolved_field_type,
            image_ref=image_ref,
            raw_ocr_text=raw_text,
            confidence=None,
            engine_used="paddleocr",
            status="LOW_CONFIDENCE",
            format_valid=False,
            reasons=[reason],
            validation=validation,
            ocr_agreement=None,
            candidates=ocr_result.get("candidates", []),
        )
        result = {
            "request_id": request_id,
            "field_type": resolved_field_type,
            "raw_ocr_text": raw_text,
            "confidence": None,
            "format_valid": False,
            "status": "LOW_CONFIDENCE",
            "error_reason": reason,
            "engine_used": "paddleocr",
            "validation": validation,
            "tube_detection_confidence": tube_detection_confidence,
            "date_check": check_reference_date(None, format_cfg, None, reference_date),
        }
        if debug:
            result["debug"] = {
                "tube_bbox": localization.get("bbox"),
                "search_image_base64": _encode_debug_image(search_image),
                "candidates": ocr_result.get("candidates", []),
            }
        return result

    raw_text = ocr_result["text"]
    extracted_digits = ocr_result["extracted_digits"]
    confidence = ocr_result["confidence"]
    ocr_agreement = ocr_result["agreement"]
    engine_used = "paddleocr"

    if confidence < GEMINI_FALLBACK_THRESHOLD:
        gemini_result = gemini_vision_fallback_stub(search_image, resolved_field_type)
        if gemini_result is not None:
            raw_text = gemini_result["text"]
            extracted_digits = gemini_result["text"]
            confidence = gemini_result["confidence"]
            engine_used = "gemini_fallback"

    validation = validate_emboss_format(extracted_digits, format_cfg)
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
            extracted_digits=extracted_digits,
            confidence=confidence,
            engine_used=engine_used,
            status=status,
            format_valid=format_valid,
            reasons=reasons,
            validation=validation,
            ocr_agreement=ocr_agreement,
        )

    result = {
        "request_id": request_id,
        "field_type": resolved_field_type,
        "raw_ocr_text": raw_text,
        "extracted_date_code": extracted_digits,
        "confidence": confidence,
        "format_valid": format_valid,
        "status": status,
        "engine_used": engine_used,
        "ocr_agreement": ocr_agreement,
        "validation": validation,
        "tube_detection_confidence": tube_detection_confidence,
        "date_check": check_reference_date(extracted_digits, format_cfg, format_valid, reference_date),
    }

    if debug:
        result["debug"] = {
            "tube_bbox": localization.get("bbox"),
            "search_image_base64": _encode_debug_image(search_image),
            "all_detected_text": ocr_result.get("all_detected_text"),
        }

    return result


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

    label_anchor = format_cfg.get("label_anchor")
    if label_anchor:
        return analyze_label_anchor_field(
            image_bgr,
            label_anchor=label_anchor,
            format_cfg=format_cfg,
            resolved_field_type=resolved_field_type,
            request_id=request_id,
            ocr_engine=ocr_engine,
            yolo_model=yolo_model,
            debug=debug,
            image_ref=image_ref,
            reference_date=reference_date,
        )

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
