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
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR

FIELD_PATTERNS_PATH = Path(__file__).parent / "field_patterns.json"

# Same threshold used for the Gemini Vision fallback in the real architecture.
CONFIDENCE_THRESHOLD = 0.80

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


@app.get("/api/field-types")
def field_types():
    """Lets the frontend build its field_type dropdown from field_patterns.json
    instead of hardcoding field names in JS."""
    config = load_field_patterns()
    return [
        {
            "key": key,
            "label": cfg.get("label", key),
            "description": cfg.get("description"),
            "expected_length": cfg.get("expected_length"),
        }
        for key, cfg in config.items()
    ]


@app.post("/api/ocr")
async def run_ocr(file: UploadFile = File(...), field_type: str = Form(None)):
    raw_bytes = await file.read()
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not decode image. Unsupported or corrupt file."}

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
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "processing_time_ms": elapsed_ms,
        "field_type": field_type,
        "detection_count": len(detections),
        "detections": detections,
    }
