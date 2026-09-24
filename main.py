"""
IPC OCR test tool - Vision Service prototype (COSMAX)

Standalone local testing tool to eyeball PaddleOCR accuracy on Label Bulk/FG
and WI Filling/Packing documents, before defining Fixed ROI per field.

Adds a format-validation layer on top of OCR confidence: per field_type
expected patterns live in field_patterns.json (edit that file, not this one).
A text that fails its pattern is flagged FORMAT_MISMATCH regardless of
confidence - this tool never guesses/trims text, it only flags for review.
This service still never decides PASS/FAIL; that stays in Laravel.

NOT production code: no auth, no HTTPS, no request_id/API contract yet.
Run with:  uvicorn main:app --reload --port 8000
"""

import json
import re
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR

import tube_emboss_pipeline

FIELD_PATTERNS_PATH = Path(__file__).parent / "field_patterns.json"

# Same threshold used for the Gemini Vision fallback in the real architecture.
CONFIDENCE_THRESHOLD = 0.80

DECODE_ERROR = {"error": "Could not decode image. Unsupported or corrupt file."}

app = FastAPI(title="Vision Service - OCR Test Tool")

# Wide open CORS: this is a local testing tool, not production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup and reused across requests (model load is slow).
# enable_mkldnn=False works around a oneDNN crash
# (NotImplementedError: ConvertPirAttribute2RuntimeAttribute ...) seen on this
# machine with paddlepaddle 3.3.1 + paddleocr 3.7.0 on CPU inference.
print("Loading PaddleOCR model, please wait...")
ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    enable_mkldnn=False,
)
print("PaddleOCR ready.")

# Tube emboss sub-pipeline: YOLO11n tube detector, loaded once and reused
# (same reasoning as ocr_engine above - reused across requests, not
# reloaded per-request, and NOT a second PaddleOCR instance).
print("Loading tube detector (YOLO11n), please wait...")
tube_yolo_model = tube_emboss_pipeline.load_yolo_model()
print("Tube detector ready.")


def load_field_patterns():
    # Re-read on every call (cheap, tiny file) so edits to field_patterns.json
    # take effect immediately without restarting uvicorn.
    with open(FIELD_PATTERNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_against_pattern(text, field_cfg):
    """Format validation, independent of OCR confidence. Never modifies text."""
    pattern = field_cfg.get("pattern")
    if not pattern:
        return {
            "validation_status": "NOT_APPLICABLE",
            "expected_pattern": None,
            "expected_length": field_cfg.get("expected_length"),
            "actual_length": len(text),
            "roi_hint": None,
        }

    matched = re.fullmatch(pattern, text) is not None
    result = {
        "validation_status": "FORMAT_OK" if matched else "FORMAT_MISMATCH",
        "expected_pattern": pattern,
        "expected_length": field_cfg.get("expected_length"),
        "actual_length": len(text),
        "roi_hint": None,
    }
    # Surface the ROI note only when it's actually relevant (a mismatch happened).
    if not matched and field_cfg.get("roi_hint"):
        result["roi_hint"] = field_cfg["roi_hint"]
    return result


def combine_status(confidence, validation_status):
    """
    Format mismatch always wins over confidence - a confident misread (e.g.
    92% on a knurl ridge misread as an extra character) must still route to
    review, not PASS. This function only proposes a status; PASS/FAIL is
    decided downstream in Laravel, never here.
    """
    if validation_status == "FORMAT_MISMATCH":
        return "FORMAT_MISMATCH"
    if confidence < CONFIDENCE_THRESHOLD:
        return "LOW_CONFIDENCE"
    return "OK"


@app.get("/api/health")
def health():
    return {"status": "ok"}


def run_generic_pipeline(image, field_type):
    """Whole-image OCR + per-detection regex validation, no localization/ROI
    step - every text region PaddleOCR finds gets checked against
    field_type's pattern. Fine for field types that don't need isolating
    from surrounding text/noise; NOT what you want for something like the
    tube cap emboss (see run_tube_emboss_pipeline)."""
    field_config = load_field_patterns()
    field_cfg = field_config.get(field_type, {}) if field_type else {}

    start = time.time()
    results = ocr_engine.predict(input=image)
    elapsed_ms = round((time.time() - start) * 1000, 1)

    detections = []
    for res in results:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        polys = res.get("rec_polys", [])

        for text, score, poly in zip(texts, scores, polys):
            confidence = round(float(score), 4)
            validation = validate_against_pattern(text, field_cfg)
            status = combine_status(confidence, validation["validation_status"])

            detections.append(
                {
                    "text": text,
                    "confidence": confidence,
                    # quad polygon [[x,y], [x,y], [x,y], [x,y]] in image pixel coords
                    "box": poly.tolist() if hasattr(poly, "tolist") else poly,
                    "validation_status": validation["validation_status"],
                    "expected_pattern": validation["expected_pattern"],
                    "expected_length": validation["expected_length"],
                    "actual_length": validation["actual_length"],
                    "roi_hint": validation["roi_hint"],
                    "status": status,
                }
            )

    return {
        "pipeline": "generic",
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "processing_time_ms": elapsed_ms,
        "field_type": field_type,
        "detection_count": len(detections),
        "detections": detections,
    }


def run_tube_emboss_pipeline(image, field_type, request_id, debug, image_ref, reference_date=None):
    """Tube emboss code sub-pipeline: YOLO11n localization -> crimp-area
    crop -> PaddleOCR -> format validation. See tube_emboss_pipeline.py.
    Still never decides PASS/FAIL/REVIEW - that stays in Laravel.

    reference_date (optional, DDMMYY string e.g. a QC-entered mixing date)
    is only used for the date_check cross-check in analyze_tube_emboss() -
    comparing it against the OCR'd day/month/year is for testing OCR read
    accuracy, not a business rule that the emboss must equal that date."""
    start = time.time()
    result = tube_emboss_pipeline.analyze_tube_emboss(
        image,
        field_type=field_type or tube_emboss_pipeline.DEFAULT_FIELD_TYPE,
        request_id=request_id,
        ocr_engine=ocr_engine,
        yolo_model=tube_yolo_model,
        debug=debug,
        image_ref=image_ref,
        reference_date=reference_date,
    )
    result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    result["pipeline"] = "tube_emboss"
    return result


@app.get("/api/field-types")
def field_types():
    """Unified field_type catalog for the frontend dropdown (index.html) -
    merges field_patterns.json (generic, whole-image OCR, no localization)
    and emboss_format_patterns.json (tube_emboss sub-pipeline: YOLO localize
    + crop before OCR), each tagged with its 'pipeline' so /api/analyze
    knows which one to dispatch to without field keys hardcoded in JS."""
    merged = []
    for key, cfg in load_field_patterns().items():
        merged.append(
            {
                "key": key,
                "label": cfg.get("label", key),
                "description": cfg.get("description"),
                "expected_length": cfg.get("expected_length"),
                "pipeline": "generic",
            }
        )
    for key, cfg in tube_emboss_pipeline.load_emboss_format_patterns().items():
        merged.append(
            {
                "key": key,
                "label": cfg.get("label", key),
                "description": cfg.get("description"),
                "expected_length": sum(b["width"] for b in cfg.get("blocks", [])) or None,
                "pipeline": "tube_emboss",
            }
        )
    return merged


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    field_type: str = Form(None),
    request_id: str = Form(None),
    debug: bool = Form(False),
    reference_date: str = Form(None),
):
    """Single entrypoint for index.html: looks up field_type's pipeline in
    emboss_format_patterns.json vs field_patterns.json and dispatches to the
    matching pipeline above. This is the seam where future field types
    (WI, exp date, etc.) plug in their own localization pipeline the same
    way tube_emboss did, without the frontend needing to know or hardcode
    which pipeline each field_type uses.

    reference_date (optional, DDMMYY string) is a QC-entered ground-truth
    date - e.g. mixing date - used only by the tube_emboss pipeline's
    date_check cross-check (see tube_emboss_pipeline.check_reference_date);
    the generic pipeline ignores it."""
    raw_bytes = await file.read()
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        return DECODE_ERROR

    is_tube_emboss = bool(field_type) and field_type in tube_emboss_pipeline.load_emboss_format_patterns()
    if is_tube_emboss:
        return run_tube_emboss_pipeline(
            image,
            field_type=field_type,
            request_id=request_id or str(uuid.uuid4()),
            debug=debug,
            image_ref=file.filename or "unknown",
            reference_date=reference_date,
        )
    return run_generic_pipeline(image, field_type)


@app.get("/api/tube-emboss/field-types")
def tube_emboss_field_types():
    """Lets the standalone tube emboss test page (tube_emboss.html) build
    its field_type dropdown from emboss_format_patterns.json directly."""
    config = tube_emboss_pipeline.load_emboss_format_patterns()
    return [
        {
            "key": key,
            "label": cfg.get("label", key),
            "description": cfg.get("description"),
            "expected_length": sum(b["width"] for b in cfg.get("blocks", [])) or None,
        }
        for key, cfg in config.items()
    ]


@app.post("/api/ocr")
async def run_ocr(file: UploadFile = File(...), field_type: str = Form(None)):
    raw_bytes = await file.read()
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        return DECODE_ERROR

    return run_generic_pipeline(image, field_type)


@app.post("/api/tube-emboss/analyze")
async def tube_emboss_analyze(
    file: UploadFile = File(...),
    request_id: str = Form(...),
    field_type: str = Form(None),
    debug: bool = Form(False),
    reference_date: str = Form(None),
):
    """Tube emboss code sub-pipeline, kept as its own endpoint for the
    standalone tube_emboss.html page and tests/test_tube_emboss.py.

    reference_date: see analyze()'s docstring above."""
    raw_bytes = await file.read()
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        return DECODE_ERROR

    return run_tube_emboss_pipeline(
        image,
        field_type=field_type or tube_emboss_pipeline.DEFAULT_FIELD_TYPE,
        request_id=request_id,
        debug=debug,
        image_ref=file.filename or "unknown",
        reference_date=reference_date,
    )
