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

# Minimum confidence to trust the tube bbox specifically on the label_anchor
# path (analyze_label_anchor_field) - deliberately higher than
# YOLO_MIN_CONFIDENCE. Added 2026-09-24 after a real sample (yellow bottle
# cap, "EXP.150728 TU") got a weak 41.6% "tube" detection that actually
# boxed an unrelated blurry background object, not the cap - the crop fed to
# OCR excluded the real text entirely (raw_ocr_text came back empty) even
# though the label_anchor path's whole-frame fallback (see module docstring)
# would have found it fine. On this path a false-accept is strictly worse
# than a false-reject: rejecting just means searching the whole frame
# instead (still robust, since anchor search doesn't depend on a precise
# crop), while accepting a wrong low-confidence bbox throws away the region
# that actually has the text. The tube detector was never trained on caps,
# so a weak score there is noise, not signal - unlike the default
# tube_emboss_default path where a wrong bbox is fatal either way, so its
# lower bar (just "is this worth attempting OCR at all") still makes sense.
LABEL_ANCHOR_MIN_TUBE_CONFIDENCE = 0.5

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

# Dotted format (added 2026-09-24, printed body-label sample e.g.
# "EXP: 27.01.2029"): day.month.year with a 4-digit year instead of the
# compact 6-digit run above - printed labels sometimes spell the year in
# full rather than the tube-emboss convention of 2 digits. The year is
# restricted to 20xx: if the OCR'd year doesn't start with "20" this pattern
# simply doesn't match at all (never forced/guessed) rather than assuming a
# century. Separator between day/month/year may be ".", "-", "/", or ":" -
# ":" is in there because a zoomed re-OCR pass (find_zoom_retry_match) of a
# real sample misread one of the two dots as a colon ("EXP: 27.01:2029") -
# recognized character, not guessed, so accepting it as a separator doesn't
# touch the never-strip/never-correct-digits rule.
LABEL_ANCHOR_DOTTED_PATTERN_TEMPLATE = r"{label}\.?\s*:?\s*(\d{{2}})[.\-/:](\d{{2}})[.\-/:](20\d{{2}})"


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


def ocr_text_pieces(image_bgr, ocr_engine, **predict_kwargs) -> list:
    """Runs PaddleOCR once and returns each detected text region as its own
    piece {"text", "confidence", "poly"}, in reading order (top-to-bottom,
    left-to-right by box centroid) - no joining/concatenation. Shared by
    recognize_crimp_text() (which joins pieces back into one code) and the
    label-anchor search (which must evaluate each detected region on its own,
    see find_label_anchor_match()) so a piece of unrelated printed text next
    to the code can never get glued onto it. "poly" (the region's own 4-point
    quad in the image passed in) is carried along so find_zoom_retry_match()
    can crop back to a specific piece's location without re-detecting it.

    **predict_kwargs are forwarded straight to PaddleOCR.predict() - e.g.
    find_zoom_retry_match() passes a lower text_det_box_thresh there to keep
    faint/dense text boxes the default threshold would otherwise discard.
    These are real per-call overrides in this installed paddleocr version
    (verified 2026-09-24 against PaddleOCR.predict()'s signature) - no need
    for a second PaddleOCR instance to use different detection parameters."""
    results = ocr_engine.predict(input=image_bgr, **predict_kwargs)

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
    return [{"text": texts[i], "confidence": float(scores[i]), "poly": polys[i]} for i in order]


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

    Tries the compact 6-digit pattern first, then - only if that doesn't
    match within a given piece - the dotted DD.MM.YYYY pattern (see
    LABEL_ANCHOR_DOTTED_PATTERN_TEMPLATE), normalizing the match to the same
    6-char day+month+year(last 2 digits) shape validate_emboss_format()
    expects. That truncation only ever drops the literal "20" century digits
    that pattern itself requires to match in the first place - it never
    edits or guesses at the day/month/year digits that carry the actual
    reported date, and matched_text still carries the full untouched OCR
    text (e.g. "EXP: 27.01.2029") for a human reviewer to see.

    Returns {"matched": False} if no piece contains the label, or
    {"matched": False, "ambiguous": True, "candidates": [...]} if multiple
    pieces match with DIFFERENT 6-digit codes - never silently pick one.
    Pieces matching with the same code (e.g. duplicate detections) resolve to
    the single highest-confidence match.
    """
    compact_pattern = re.compile(LABEL_ANCHOR_PATTERN_TEMPLATE.format(label=re.escape(label)), re.IGNORECASE)
    dotted_pattern = re.compile(LABEL_ANCHOR_DOTTED_PATTERN_TEMPLATE.format(label=re.escape(label)), re.IGNORECASE)
    hits = []
    for piece in pieces:
        m = compact_pattern.search(piece["text"])
        if m:
            hits.append({
                "matched_text": piece["text"],
                "extracted_digits": m.group(1),
                "confidence": piece["confidence"],
                "date_format": "compact_6digit",
            })
            continue
        m = dotted_pattern.search(piece["text"])
        if m:
            day, month, year4 = m.group(1), m.group(2), m.group(3)
            hits.append({
                "matched_text": piece["text"],
                "extracted_digits": day + month + year4[-2:],
                "confidence": piece["confidence"],
                "date_format": "dotted_4digit_year",
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
        "date_format": best["date_format"],
    }


# Labels of OTHER, distinctly different emboss/print date fields that
# digit_fallback must never mistake for the field it's actually searching
# for. Added 2026-09-24 after a real Pond's tube sample: "MFD 140624 8 QFZ"
# (manufacture date, printed on the tube body) was the only isolated
# 6-digit run digit_fallback could find at all - the real EXP code
# ("EXP 140627", embossed on the crimp) had its "140" fused/dropped by OCR
# in every pass tried (raw, sharpened, AND the zoomed re-OCR retry - see
# find_zoom_retry_match), leaving only a stray "627" (3 digits, too short to
# match). Since digit_fallback has no literal-label anchor of its own
# (that's the whole point of it - see the docstring below), nothing else
# stopped it from confidently returning MFD's date as if it were EXP's.
# "MFD" is a standard packaging abbreviation (manufacture date), not
# SKU-specific data, and the same tier of literal signal find_label_anchor_
# match() already keys on for "EXP" - so rejecting a piece that spells out
# a DIFFERENT date field's own label isn't a guess, it's reading text that
# says outright "this isn't the date you're looking for."
DIGIT_FALLBACK_EXCLUDED_LABELS = ("MFD",)


def find_digit_fallback_match(pieces: list, format_cfg: dict, other_format_configs: list = None) -> dict:
    """Fallback for label_anchor fields when no literal label is found at all
    in either OCR pass (see recognize_label_anchor_dual). Added 2026-09-24
    after a real sample tube cap ("AJA120927 PU") turned out to have no "EXP"
    text anywhere near its date - a different physical emboss layout than the
    three samples that motivated the label_anchor approach in the first
    place (see emboss_format_patterns.json's tube_exp_date entry).

    Searches each OCR piece for an isolated 6-digit run - (?<!\\d) / (?!\\d)
    guards mean a run that's actually part of a longer ALL-DIGIT sequence
    never gets truncated into a false 6-digit match - and keeps only runs
    that pass this field_type's own day/month/year int_range validation
    (rejects clearly-wrong candidates like month=88). Pieces containing a
    DIGIT_FALLBACK_EXCLUDED_LABELS entry (see its own comment) are skipped
    entirely before any of that - a clean, correctly-formatted date sitting
    right next to the literal word "MFD" is still the wrong field, not a
    valid unlabeled candidate. Never invents or edits digits, exactly like
    find_label_anchor_match() - it only decides which already-detected run
    to trust. Multiple distinct valid candidates -> ambiguous, same
    never-guess rule as the label-anchor path.

    `other_format_configs` (2026-09-25) closes a gap the isolated-digit-run
    guard above does NOT cover: tube_emboss_default's own code is DDMMYY
    immediately followed by 3 alnum + 1 digit with no separator (e.g.
    "310826QHB8") - since a LETTER, not a digit, follows the date, the
    (?!\\d) lookahead is satisfied trivially and the leading "310826" still
    matches as an "isolated" 6-digit run. A real sample (Pond's UV Miracle,
    tube with ONLY that combined code, no separate EXP line at all)
    confirmed this: digit_fallback confidently returned "310826" as an EXP
    candidate purely because that tube's manufacture date happens to also be
    a syntactically valid DDMMYY. Fix: before accepting a candidate, check
    whether the WHOLE piece's text (not just the extracted digits) already
    validates cleanly as a DIFFERENT known tube-emboss field_type's exact
    format (caller passes every OTHER field_type's config from
    emboss_format_patterns.json). A piece that's a clean whole match for
    e.g. tube_emboss_default's 10-char shape is structurally that code, not
    an unlabeled EXP date that happens to start with 6 valid digits - skip
    it entirely, don't just downrank it. This is data-driven (reads
    whatever other field_types exist in the config file) rather than a
    hardcoded label string like DIGIT_FALLBACK_EXCLUDED_LABELS, so a future
    new field_type entry is covered automatically without code changes.

    This is inherently less certain than a literal label match (no "EXP" to
    anchor on), so the caller always caps the returned confidence below
    GEMINI_FALLBACK_THRESHOLD - a digit-fallback match can never come back as
    a confident OK, only LOW_CONFIDENCE for human review."""
    digit_pattern = re.compile(r"(?<!\d)(\d{6})(?!\d)")
    other_format_configs = other_format_configs or []
    hits = []
    for piece in pieces:
        text_lower = piece["text"].lower()
        if any(label.lower() in text_lower for label in DIGIT_FALLBACK_EXCLUDED_LABELS):
            continue
        if any(validate_emboss_format(piece["text"], other_cfg)["format_valid"] for other_cfg in other_format_configs):
            continue
        for m in digit_pattern.finditer(piece["text"]):
            digits = m.group(1)
            if validate_emboss_format(digits, format_cfg)["format_valid"]:
                hits.append({
                    "matched_text": piece["text"],
                    "extracted_digits": digits,
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


# How far above/below a label-containing piece's own detected box to crop
# before zooming in (find_zoom_retry_match) - as a multiple of that piece's
# own height. Generous on purpose: the whole reason the label's own box has
# no digits attached is that the detector already mis-boxed this dense text,
# so a tight crop right at the piece's edges could just as easily clip the
# date again. A wider band re-exposes the full physical text line to a fresh
# detection pass instead of trusting the original (already wrong) box shape.
ZOOM_RETRY_VERTICAL_PADDING_FACTOR = 1.5

# Upscale the cropped band until it's roughly this many px tall. Sample
# measurement 2026-09-24: a real label's MFG/EXP/LOT block was ~120px tall in
# an 845px-tall tube crop - too small for the detector to keep the date
# digits as a real motivating case (see gotcha #11 in CLAUDE.md); upscaling
# just that band clean enough to read.
ZOOM_RETRY_UPSCALE_TARGET_HEIGHT = 200

# Lower than the text detection model's own default (~0.45, see its
# inference.yml) - keeps faint/dense text boxes the default threshold drops
# entirely. Confirmed empirically 2026-09-24: this only mattered once
# combined with the crop+upscale above - at full-image resolution alone it
# did nothing (the digits were already gone, not just below-threshold).
ZOOM_RETRY_BOX_THRESH = 0.3


def _piece_y_range(poly):
    arr = np.array(poly)
    return float(arr[:, 1].min()), float(arr[:, 1].max())


def _piece_y_center(poly):
    return float(np.array(poly)[:, 1].mean())


def _piece_x_center(poly):
    return float(np.array(poly)[:, 0].mean())


def group_zoomed_pieces_into_lines(pieces: list) -> list:
    """Clusters OCR pieces that sit on the same visual line into one
    synthesized piece each, text concatenated in left-to-right (x-position)
    order within the line. Used ONLY by find_zoom_retry_match() on its own
    tiny re-OCR'd band - safe there specifically because that crop is
    already scoped to ~1-3 physical lines' worth of image (see
    ZOOM_RETRY_VERTICAL_PADDING_FACTOR), unlike the whole-frame/tube-crop
    pieces find_label_anchor_match() and find_digit_fallback_match() work on
    directly, which deliberately never concatenate across detections (see
    ocr_text_pieces() docstring) to avoid gluing unrelated printed text
    together.

    Added 2026-09-24 after the zoomed re-OCR of a real dense-label sample
    correctly read the EXP line's digits ('27.01.2029', 99.5% confidence)
    but as a SEPARATE detection from its own 'EXP:' label piece -
    find_label_anchor_match() only ever looks inside a single piece's text,
    so without grouping these two together they'd never be recognized as
    belonging to the same line.

    Groups by Y-CENTER proximity between each piece and the one immediately
    before it in y-sorted order (sequential/single-linkage), never against a
    cluster's own fixed seed piece and never by expanding the cluster's own
    y-range as more pieces join it. Went through two earlier designs that
    each failed on a real sample: (1) range-overlap chain-merged adjacent-
    but-DIFFERENT physical lines into one giant group, because the label's
    lines are packed close enough that consecutive lines' bounding boxes
    already overlap (that tight spacing is the whole reason this zoom path
    exists - see gotcha #11/#12 in CLAUDE.md); (2) comparing every candidate
    against a fixed seed (the first piece popped off the list, i.e. topmost)
    fixed that chain-merge but introduced a new bug on a real Pond's/
    Elsheskin sample (2026-09-24) where the digits belonged to the EXP line
    but sat almost equidistant between the MFD line's seed and EXP's own
    piece - "27.01.2029" (y-center 98) was only 3px from "EXP:" (101) but
    still landed just inside the MFD seed's threshold (gap 29.5 vs a 30px
    limit) because the MFD piece got popped as seed first and claimed it
    before EXP ever got a turn to be compared against. Sequential adjacent-
    gap grouping sidesteps both failure modes: a piece is only ever compared
    to its immediate neighbor in sorted order, so it can never be glued to a
    distant seed just because of pop order, and a run can only grow one
    small gap at a time rather than via an ever-expanding shared range."""
    ordered = sorted(
        (p for p in pieces if p.get("poly") is not None),
        key=lambda p: _piece_y_center(p["poly"]),
    )

    def _finalize(group):
        group.sort(key=lambda p: _piece_x_center(p["poly"]))
        return {
            "text": "".join(p["text"] for p in group),
            "confidence": min(p["confidence"] for p in group),
            "poly": None,
        }

    lines = []
    current_group = []
    prev_center = None
    prev_height = None
    for p in ordered:
        y1, y2 = _piece_y_range(p["poly"])
        height = max(1.0, y2 - y1)
        center = _piece_y_center(p["poly"])
        if current_group and abs(center - prev_center) > 0.5 * min(prev_height, height):
            lines.append(_finalize(current_group))
            current_group = []
        current_group.append(p)
        prev_center, prev_height = center, height
    if current_group:
        lines.append(_finalize(current_group))
    return lines


def find_zoom_retry_match(image_bgr, ocr_engine, pieces: list, label: str) -> dict:
    """Second attempt for label_anchor fields, tried right after
    find_label_anchor_match() fails on the full search image and BEFORE
    find_digit_fallback_match() (see recognize_label_anchor_dual - reordered
    2026-09-24, see that function's docstring for why). Real sample
    (2026-09-24): a printed label with MFG/EXP/LOT lines packed close
    together lost its EXP line's digits entirely at whole-image OCR
    resolution - the label text itself ("EXP:") was still detected as its
    own piece, just with no digits attached to it (dropped or fused into the
    MFG line above), so neither the label-anchor nor the digit-fallback path
    had anything usable to find.

    Finds any already-detected `pieces` entry that contains the literal
    label (this only works when the label survived at whole-image
    resolution, even without its digits - if "EXP" isn't in any piece at
    all, this returns unmatched immediately, same as the other paths), crops
    a generously padded horizontal band around that piece's own location
    from the ORIGINAL image (see ZOOM_RETRY_VERTICAL_PADDING_FACTOR),
    upscales it, and re-runs OCR on just that band with a lower
    text_det_box_thresh - then retries the ordinary label search on that
    fresh, higher-resolution read. This never re-crops based on a fixed
    pixel position (the crop is always anchored to wherever THIS image's
    OCR actually found the label) and never edits/guesses at digits - it
    only gives the detector a second, better-resolved look at the same
    physical text line. Multiple label-containing pieces yielding different
    codes -> ambiguous, same never-guess rule as every other path here."""
    label_pattern = re.compile(re.escape(label), re.IGNORECASE)
    label_pieces = [p for p in pieces if p.get("poly") is not None and label_pattern.search(p["text"])]
    if not label_pieces:
        return {"matched": False}

    img_h, img_w = image_bgr.shape[:2]
    hits = []
    for piece in label_pieces:
        poly = np.array(piece["poly"])
        y1, y2 = float(poly[:, 1].min()), float(poly[:, 1].max())
        box_height = max(1.0, y2 - y1)
        pad = box_height * ZOOM_RETRY_VERTICAL_PADDING_FACTOR
        crop_y1 = max(0, int(y1 - pad))
        crop_y2 = min(img_h, int(y2 + pad))
        band = image_bgr[crop_y1:crop_y2, 0:img_w]
        if band.shape[0] < 2 or band.shape[1] < 2:
            continue

        scale = max(1.0, ZOOM_RETRY_UPSCALE_TARGET_HEIGHT / band.shape[0])
        zoomed = cv2.resize(band, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        zoom_pieces = ocr_text_pieces(zoomed, ocr_engine, text_det_box_thresh=ZOOM_RETRY_BOX_THRESH)
        zoom_lines = group_zoomed_pieces_into_lines(zoom_pieces)
        zoom_match = find_label_anchor_match(zoom_lines, label)
        if zoom_match.get("matched"):
            hits.append(zoom_match)

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
        "date_format": best["date_format"],
    }


def recognize_label_anchor_dual(image_bgr, ocr_engine, label: str, format_cfg: dict, other_format_configs: list = None) -> dict:
    """Label-anchor equivalent of recognize_crimp_text_dual(): runs the
    anchor search on the raw image, and - unless that raw pass already
    resolved with a confident match - also on preprocess_crimp_for_ocr()'s
    sharpened version, then cross-validates the two extracted 6-digit codes.
    Agreement -> trust it (confidence = higher of the two). Disagreement
    (including only one side matching at all) -> never silently pick a side;
    take the higher-confidence match but cap the returned confidence just
    under GEMINI_FALLBACK_THRESHOLD so the caller always routes it to
    LOW_CONFIDENCE. Neither side matches the label at all -> try
    find_zoom_retry_match() first, THEN find_digit_fallback_match() across
    both passes' pieces before giving up; still not detected -> caller
    reports FORMAT_MISMATCH / label not found.

    zoom_retry is tried before digit_fallback (reordered 2026-09-24, real
    sample: a Pond's tube where OCR split "EXP" and its digits "140627" into
    two separate pieces on a physical line right next to an unrelated MFD
    date, "MFD 140624 8 QFZ", detected as its own clean single piece).
    digit_fallback has no idea a literal "EXP" exists anywhere - it just
    grabs whatever isolated 6-digit run passes day/month/year validation -
    so if it runs first and MFD's date happens to be the only (or the
    highest-confidence) candidate it can find, it confidently returns the
    WRONG date without ever attempting the anchor-aware zoom retry, which
    exists precisely to recover the true EXP digits in this situation. Since
    zoom_retry only fires at all when some piece already contains the
    literal label (see its own docstring), running it first costs nothing
    when the label is genuinely absent (it just returns unmatched
    immediately) but gives the real anchor priority whenever the label IS
    present, per COSMAX's explicit ask after seeing this exact MFD/EXP
    mixup. If zoom_retry also can't resolve it (e.g. the zoomed re-OCR
    itself misreads a digit), digit_fallback still runs as the last resort -
    unchanged from before, just demoted one step."""
    raw_pieces = ocr_text_pieces(image_bgr, ocr_engine)
    raw_match = find_label_anchor_match(raw_pieces, label)

    if raw_match.get("matched") and raw_match["confidence"] >= GEMINI_FALLBACK_THRESHOLD:
        return {
            "detected": True,
            "text": raw_match["matched_text"],
            "extracted_digits": raw_match["extracted_digits"],
            "confidence": raw_match["confidence"],
            "agreement": None,
            "match_method": "label_anchor",
            "date_format": raw_match["date_format"],
            "all_detected_text": "".join(p["text"] for p in raw_pieces),
        }

    sharpened = preprocess_crimp_for_ocr(image_bgr)
    sharp_pieces = ocr_text_pieces(sharpened, ocr_engine)
    sharp_match = find_label_anchor_match(sharp_pieces, label)

    all_detected_text = "".join(p["text"] for p in raw_pieces) or "".join(p["text"] for p in sharp_pieces)

    if not raw_match.get("matched") and not sharp_match.get("matched"):
        zoom_match = find_zoom_retry_match(image_bgr, ocr_engine, raw_pieces, label)
        if zoom_match.get("matched"):
            capped_confidence = min(zoom_match["confidence"], GEMINI_FALLBACK_THRESHOLD - 0.01)
            return {
                "detected": True,
                "text": zoom_match["matched_text"],
                "extracted_digits": zoom_match["extracted_digits"],
                "confidence": round(capped_confidence, 4),
                "agreement": None,
                "match_method": "zoom_retry",
                "date_format": zoom_match["date_format"],
                "all_detected_text": all_detected_text,
            }

        fallback_match = find_digit_fallback_match(raw_pieces + sharp_pieces, format_cfg, other_format_configs)
        if fallback_match.get("matched"):
            capped_confidence = min(fallback_match["confidence"], GEMINI_FALLBACK_THRESHOLD - 0.01)
            return {
                "detected": True,
                "text": fallback_match["matched_text"],
                "extracted_digits": fallback_match["extracted_digits"],
                "confidence": round(capped_confidence, 4),
                "agreement": None,
                "match_method": "digit_fallback",
                "date_format": None,
                "all_detected_text": all_detected_text,
            }

        return {
            "detected": False,
            "all_detected_text": all_detected_text,
            "ambiguous": raw_match.get("ambiguous", False) or sharp_match.get("ambiguous", False),
            "candidates": raw_match.get("candidates") or sharp_match.get("candidates") or [],
            "digit_fallback_ambiguous": fallback_match.get("ambiguous", False),
            "digit_fallback_candidates": fallback_match.get("candidates", []),
            "zoom_retry_ambiguous": zoom_match.get("ambiguous", False),
            "zoom_retry_candidates": zoom_match.get("candidates", []),
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
            "match_method": "label_anchor",
            "date_format": raw_match["date_format"],
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
        "match_method": "label_anchor",
        "date_format": best["date_format"],
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
    other_format_configs: list = None,
) -> dict:
    """Anchor-based path for field_types with "label_anchor" set (see module
    docstring). Tries YOLO tube localization first to cut out background
    clutter, but - unlike analyze_tube_emboss()'s default path - doesn't hard
    fail when no tube is detected: real EXP samples also come from bottle
    caps and printed labels that the tube detector never will (and was never
    trained to) recognize, so this falls back to searching the whole frame.

    Uses LABEL_ANCHOR_MIN_TUBE_CONFIDENCE (higher than the default path's
    YOLO_MIN_CONFIDENCE) to decide whether to trust the bbox at all - see
    that constant's comment for why a false-accept here is worse than a
    false-reject, unlike the default path.

    `other_format_configs` (every OTHER field_type's config from
    emboss_format_patterns.json, i.e. not this field's own format_cfg) is
    forwarded to find_digit_fallback_match() via recognize_label_anchor_dual
    - see find_digit_fallback_match()'s docstring for why (a piece that's a
    clean whole match for a different field_type's exact format, e.g.
    tube_emboss_default's 10-char code, should never be cannibalized for a
    digit_fallback candidate here just because its leading 6 digits happen
    to look like a valid date)."""
    localization = localize_tube(image_bgr, yolo_model, min_confidence=LABEL_ANCHOR_MIN_TUBE_CONFIDENCE)
    search_image = localization["tube_crop"] if localization["detected"] else image_bgr
    tube_detection_confidence = localization.get("confidence")

    ocr_result = recognize_label_anchor_dual(search_image, ocr_engine, label_anchor, format_cfg, other_format_configs)

    if not ocr_result["detected"]:
        if ocr_result.get("zoom_retry_ambiguous"):
            reason = "ZOOM_RETRY_AMBIGUOUS"
            reason_detail = "no literal label match at full resolution, and a zoomed re-read of the label's own location found multiple different dates - see candidates"
        elif ocr_result.get("digit_fallback_ambiguous"):
            reason = "DIGIT_FALLBACK_AMBIGUOUS"
            reason_detail = "no literal label match, and multiple isolated 6-digit candidates pass format validation - see candidates"
        elif ocr_result.get("ambiguous"):
            reason = "LABEL_AMBIGUOUS"
            reason_detail = f"multiple '{label_anchor}' matches with different dates - see candidates"
        else:
            reason = "LABEL_NOT_FOUND"
            reason_detail = f"label '{label_anchor}' not found in OCR text (including a zoomed re-read of any partial label match), and no valid 6-digit fallback candidate either"
        raw_text = ocr_result.get("all_detected_text") or None
        validation = {
            "format_valid": False,
            "validation_status": "FORMAT_MISMATCH",
            "expected_length": None,
            "actual_length": len(raw_text) if raw_text else 0,
            "block_mismatches": [],
            "reason": reason_detail,
        }
        all_candidates = (
            ocr_result.get("candidates", [])
            + ocr_result.get("digit_fallback_candidates", [])
            + ocr_result.get("zoom_retry_candidates", [])
        )
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
            candidates=all_candidates,
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
                "candidates": all_candidates,
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
            match_method=ocr_result.get("match_method"),
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
        "match_method": ocr_result.get("match_method"),
        "date_format": ocr_result.get("date_format"),
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
            other_format_configs=[
                cfg for key, cfg in format_config.items()
                if key != resolved_field_type and isinstance(cfg, dict) and cfg.get("blocks")
            ],
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
