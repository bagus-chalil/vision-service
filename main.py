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
import os
import re
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import coding_formula
import insights
import llm_extract
import tube_emboss_pipeline

FIELD_PATTERNS_PATH = Path(__file__).parent / "field_patterns.json"

# Same threshold used for the Gemini Vision fallback in the real architecture.
CONFIDENCE_THRESHOLD = 0.80

DECODE_ERROR = {"error": "Could not decode image. Unsupported or corrupt file."}

app = FastAPI(title="Vision Service - OCR Test Tool")

# CORS origins: wide open ("*") by default, which is fine for local testing
# (localhost frontend hitting localhost backend) but should be locked down
# for pilot deployment on a shared server. Set the env var below to a
# comma-separated allowlist before starting uvicorn, e.g.:
#   $env:VISION_SERVICE_ALLOWED_ORIGINS = "http://100.100.160.23,http://127.0.0.1:5500"
# This is a stopgap for the pilot, not the final auth story - the real
# architecture still has no login/API key/HTTPS here (see CLAUDE.md); network
# access should also be restricted at the firewall (see ops/configure_firewall.ps1).
_allowed_origins_env = os.environ.get("VISION_SERVICE_ALLOWED_ORIGINS", "").strip()
if _allowed_origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = ["*"]
    print(
        "WARNING: VISION_SERVICE_ALLOWED_ORIGINS not set - CORS is wide open (*). "
        "Fine for local testing; for pilot deployment, set that env var to the "
        "specific frontend origin(s) and restrict port access at the firewall."
    )

# API key opsional (stopgap pilot, BUKAN auth final): bila VISION_API_KEY diset, semua
# /api/* kecuali /api/health wajib header X-API-Key yang sama. Kosong = tanpa auth
# (perilaku lama). Halaman uji (/, /startup-check) tidak ikut dijaga; batasi lewat firewall.
VISION_API_KEY = os.environ.get("VISION_API_KEY", "").strip()


@app.middleware("http")
async def _api_key_guard(request, call_next):
    path = request.url.path
    if (VISION_API_KEY and path.startswith("/api/") and path != "/api/health"
            and request.method != "OPTIONS"
            and request.headers.get("x-api-key") != VISION_API_KEY):
        return JSONResponse(status_code=401, content={"status": "ERROR", "error": "X-API-Key tidak valid."})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD_PADDLEOCR (default aktif): "0"/"false" = jangan muat PaddleOCR (hemat ~1-1,5 GB
# RAM). Jalur LLM (label_bulk, im_document, coding_print, tube_exp_llm) tidak butuh
# PaddleOCR; field_type jalur OCR (generic + tube_emboss) dijawab 503 bila dimatikan.
LOAD_PADDLEOCR = os.environ.get("LOAD_PADDLEOCR", "1").strip().lower() not in ("0", "false", "no", "off")

# Loaded once at startup and reused across requests (model load is slow).
# enable_mkldnn=False works around a oneDNN crash
# (NotImplementedError: ConvertPirAttribute2RuntimeAttribute ...) seen on this
# machine with paddlepaddle 3.3.1 + paddleocr 3.7.0 on CPU inference.
ocr_engine = None
if LOAD_PADDLEOCR:
    from paddleocr import PaddleOCR

    print("Loading PaddleOCR model, please wait...")
    ocr_engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
    )
    print("PaddleOCR ready.")
else:
    print("LOAD_PADDLEOCR=0 - PaddleOCR TIDAK dimuat; hanya jalur LLM yang tersedia.")

# PaddleOCR (dan YOLO di jalur tube_emboss) tidak dijamin aman dipanggil paralel dari
# threadpool -> satu kunci untuk seluruh jalur OCR. Perilaku antrean sama seperti
# sebelumnya (satu request OCR sekali jalan), tetapi event loop tidak lagi tertahan
# (health check dan request jalur LLM tetap jalan selama OCR berlangsung).
_OCR_LOCK = threading.Lock()


def _locked(fn, *args, **kwargs):
    with _OCR_LOCK:
        return fn(*args, **kwargs)


async def run_ocr_path(fn, *args, **kwargs):
    """Jalankan pipeline OCR di threadpool (satu per satu). 503 bila PaddleOCR dimatikan."""
    if ocr_engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "ERROR", "error": "PaddleOCR dimatikan di server ini (LOAD_PADDLEOCR=0); field_type ini butuh PaddleOCR."},
        )
    return await run_in_threadpool(_locked, fn, *args, **kwargs)


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


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def serve_index():
    """Sajikan index.html dari server yang sama (testing lokal): Backend URL
    otomatis = origin halaman, tanpa soal CORS/file://."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/startup-check", include_in_schema=False)
def serve_startup_check():
    """Halaman uji gabungan Startup Check (alat testing; hitung EXP/batch
    diharapkan di browser, keputusan resmi tetap di ipc_app)."""
    return FileResponse(Path(__file__).parent / "startup_check.html")


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


def analyze_tube_exp_llm(image):
    """tube_exp_llm: Python (YOLO) memotong crop crimp, Ollama hanya mentranskripsi tiap
    crop, aturan sepakat 3-dari-4 ada di llm_extract.analyze_tube_exp. Tidak ada PaddleOCR.
    Tidak ada tube terdeteksi -> LOW_CONFIDENCE tanpa memanggil LLM."""
    crops = tube_emboss_pipeline.tube_exp_crops(image, tube_yolo_model)
    if not crops["detected"]:
        result = llm_extract.analyze_tube_exp(image, [])
        result.update(tube_detected=False, yolo_reason=crops.get("reason"))
        return result
    result = llm_extract.analyze_tube_exp(image, crops["crops"])
    result.update(tube_detected=True, tube_bbox=crops["bbox"], yolo_confidence=crops["confidence"])
    return result


LLM_FIELD_TYPES = {
    "tube_exp_llm": {
        "label": "Tube EXP (LLM, crop YOLO)",
        "description": "EXP emboss crimp tube: YOLO crop (Python) + transkripsi LLM (Ollama GPU), OK hanya bila >=3 dari 4 sumber sepakat. Tanpa PaddleOCR.",
        "fn": analyze_tube_exp_llm,
    },
    "label_bulk": {
        "label": "Label Bulk",
        "description": "No Batch + Mixing Date + Kode Bulk dibaca LLM (Ollama GPU), konsensus multi-ukuran",
        "fn": llm_extract.analyze_label_bulk,
    },
    "im_document": {
        "label": "IM - Umur Simpan",
        "description": "Nomor IM + umur simpan per kemasan (teks asli, tahun/bulan apa adanya) dibaca LLM (Ollama GPU)",
        "fn": llm_extract.analyze_im_document,
    },
    "coding_print": {
        "label": "Coding non-tube (batch + EXP)",
        "description": "Batch + EXP tercetak di kotak/jar/tutup/botol dibaca LLM (Ollama GPU); EXP DDMMYY atau bulan-tahun apa adanya, packaging_type hanya flag",
        "fn": llm_extract.analyze_coding_print,
    },
}


def run_llm_pipeline(image, field_type, request_id):
    """Document field_types read by the Ollama vision LLM (no PaddleOCR).
    Returns raw values + legibility + status only - never PASS/FAIL."""
    start = time.time()
    result = LLM_FIELD_TYPES[field_type]["fn"](image)
    result["field_type"] = field_type
    result["request_id"] = request_id
    result["pipeline"] = "llm"
    result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
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
                "pilot_ready": cfg.get("pilot_ready", True),
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
                "pilot_ready": cfg.get("pilot_ready", True),
                "exp_from_shelf_life": cfg.get("exp_from_shelf_life", False),
            }
        )
    for key, cfg in LLM_FIELD_TYPES.items():
        merged.append(
            {
                "key": key,
                "label": cfg["label"],
                "description": cfg["description"],
                "expected_length": None,
                "pipeline": "llm",
                "pilot_ready": True,
            }
        )
    merged.append(
        {
            "key": CODING_FORMULA_FIELD,
            "label": "Coding menurut formula (batch + EXP)",
            "description": "Batch/EXP/MFG dicari menurut formula SKU di mana pun letaknya (emboss/inkjet/label); "
                           "tube boleh 2 foto. Butuh form field coding_formula. LLM Ollama hanya menyalin.",
            "expected_length": None,
            "pipeline": "llm",
            "pilot_ready": True,
            "needs_formula": True,
        }
    )
    return merged


CODING_FORMULA_FIELD = "coding_formula"
CODING_FORMULAS_PATH = Path(__file__).parent / "coding_formulas.json"


def load_coding_formula_catalog():
    """Contoh formula (dibaca ulang tiap request, edit tanpa restart)."""
    data = json.loads(CODING_FORMULAS_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def resolve_coding_formula(raw):
    """Form field coding_formula -> (list segmen, sumber). Terima: kosong / "auto" (gabungan semua
    formula katalog, sumber "auto"), kunci katalog (sumber "catalog"), JSON array string atau satu
    template string (sumber "sku"). Selalu divalidasi (FormulaError bila salah)."""
    raw = (raw or "").strip()
    catalog = load_coding_formula_catalog()
    if not raw or raw.lower() == coding_formula.AUTO:
        segments = coding_formula.auto_segments(catalog)
        coding_formula.compile_formula(segments)
        return segments, "auto"
    if raw in catalog:
        coding_formula.compile_formula(catalog[raw]["segments"])
        return catalog[raw]["segments"], "catalog"
    if raw.startswith("["):
        try:
            segments = json.loads(raw)
        except ValueError as exc:
            raise coding_formula.FormulaError(f"coding_formula bukan JSON array yang valid: {exc}") from exc
    else:
        segments = [raw]
    coding_formula.compile_formula(segments)
    return segments, "sku"


def run_coding_formula_pipeline(images, segments, request_id, formula_source="sku"):
    start = time.time()
    required = ["batch", "exp"] if formula_source == "auto" else None
    result = llm_extract.analyze_coding_formula(images, segments, required=required)
    result.update(field_type=CODING_FORMULA_FIELD, request_id=request_id, pipeline="llm",
                  formula_source=formula_source,
                  processing_time_ms=round((time.time() - start) * 1000, 1))
    return result


# Menu halaman uji (index.html) = input yang dipakai integrasi ipc_app (Startup Check):
# Label Bulk, IM, coding non-tube (1 foto), tube (2 foto, batch + EXP). Field type lama tetap
# bisa dipilih lewat centang "dev" di halaman. Ini hanya alat uji: di production field_type +
# formula dikirim ipc_app per langkah alur kerja / per SKU, Vision Service tidak menebak.
PRODUCTION_MENU = [
    {"id": "label_bulk", "label": "Label Bulk - No Batch + Mixing Date", "field_type": "label_bulk", "photos": 1},
    {"id": "im", "label": "IM - umur simpan", "field_type": "im_document", "photos": 1},
    {"id": "coding", "label": "Coding non-tube - batch + EXP (1 foto)", "field_type": CODING_FORMULA_FIELD,
     "photos": 1, "default_formula": coding_formula.AUTO},
    {"id": "tube", "label": "Tube - batch + EXP (foto 2 sisi)", "field_type": CODING_FORMULA_FIELD,
     "photos": 2, "default_formula": coding_formula.AUTO},
]


@app.get("/api/test-menu")
def test_menu():
    return PRODUCTION_MENU


@app.get("/api/coding-formulas")
def coding_formulas():
    """Katalog token + contoh formula, untuk ipc_app/QC menyusun formula per SKU."""
    return {
        "tokens": {k: {"field": v["field"], "description": v["desc"]} for k, v in coding_formula.TOKENS.items()},
        "generic_tokens": "L<n> huruf besar, D<n> angka, X<n> huruf besar/angka (field lewat @, mis. {X7@batch})",
        "syntax": "{TOKEN} / {TOKEN@field}; teks = label literal; [a|b] opsional; (a|b) alternatif wajib; "
                  "pemisah (spasi / . : - _ baris baru) selalu opsional",
        "catalog": load_coding_formula_catalog(),
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    field_type: str = Form(None),
    request_id: str = Form(None),
    debug: bool = Form(False),
    reference_date: str = Form(None),
    coding_formula_spec: str = Form(None, alias="coding_formula"),
    file2: UploadFile = File(None),
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

    if field_type == CODING_FORMULA_FIELD:
        try:
            segments, formula_source = resolve_coding_formula(coding_formula_spec)
        except coding_formula.FormulaError as exc:
            return JSONResponse(status_code=400, content={"status": "ERROR", "error": f"FORMULA_INVALID: {exc}"})
        images = [image]
        if file2 is not None and file2.filename:
            image2 = cv2.imdecode(np.frombuffer(await file2.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image2 is None:
                return DECODE_ERROR
            images.append(image2)
        return await run_in_threadpool(
            run_coding_formula_pipeline, images, segments, request_id or str(uuid.uuid4()), formula_source
        )

    if field_type in LLM_FIELD_TYPES:
        # LLM jalur (Ollama GPU). Panggilan HTTP-nya blocking dan lama (beberapa
        # detik), jadi dilempar ke threadpool supaya event loop tidak tertahan
        # (gotcha #16; jalur OCR kini juga di threadpool lewat run_ocr_path).
        return await run_in_threadpool(run_llm_pipeline, image, field_type, request_id or str(uuid.uuid4()))

    is_tube_emboss = bool(field_type) and field_type in tube_emboss_pipeline.load_emboss_format_patterns()
    if is_tube_emboss:
        return await run_ocr_path(
            run_tube_emboss_pipeline,
            image,
            field_type=field_type,
            request_id=request_id or str(uuid.uuid4()),
            debug=debug,
            image_ref=file.filename or "unknown",
            reference_date=reference_date,
        )
    return await run_ocr_path(run_generic_pipeline, image, field_type)


class SummarizeTextRequest(BaseModel):
    system: str = ""
    prompt: str


@app.post("/api/summarize-text")
async def summarize_text(body: SummarizeTextRequest):
    """Text-only LLM call (same Ollama server as the OCR readers). Used by ipc_app's "Analisis AI"
    over its field-issue log. The caller must send already-anonymised text; nothing is stored here."""
    if not body.prompt.strip() or len(body.prompt) > 12000 or len(body.system) > 4000:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "prompt kosong atau terlalu panjang"})
    try:
        text = await run_in_threadpool(llm_extract.chat_text, body.system, body.prompt)
    except llm_extract.LlmUnavailable as exc:
        return JSONResponse(status_code=503, content={"status": "ERROR", "error": f"Ollama tidak tersedia: {exc}"})
    return {"status": "OK", "text": text}


class InsightRequest(BaseModel):
    kind: str
    facts: list[str]


@app.get("/api/insights/kinds")
def insight_kinds():
    """Which insight kinds this build can write (a consumer can check before showing its button)."""
    return {"kinds": insights.kinds()}


@app.post("/api/insights/summarize")
async def insight_summarize(body: InsightRequest):
    """Typed text insight for consumer apps (ipc_app, new_trial_validation_app). The consumer sends
    its own already-anonymised FACTS and a `kind`; the prompt, writing rules and output check live in
    insights.py, so no consumer ships a prompt of its own. Nothing is stored here. An answer that
    recommends approve/reject/hold is refused (422) so the consumer falls back to its rule-based text."""
    try:
        text = await run_in_threadpool(insights.summarize, body.kind, body.facts)
    except insights.InsightError as exc:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": str(exc)})
    except insights.AdvisoryViolation:
        return JSONResponse(
            status_code=422,
            content={"status": "ERROR", "error": "jawaban AI melanggar aturan advisory (menyarankan keputusan), ditolak"},
        )
    except llm_extract.LlmUnavailable as exc:
        return JSONResponse(status_code=503, content={"status": "ERROR", "error": f"Ollama tidak tersedia: {exc}"})
    return {"status": "OK", "kind": body.kind, "text": text}


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

    return await run_ocr_path(run_generic_pipeline, image, field_type)


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

    return await run_ocr_path(
        run_tube_emboss_pipeline,
        image,
        field_type=field_type or tube_emboss_pipeline.DEFAULT_FIELD_TYPE,
        request_id=request_id,
        debug=debug,
        image_ref=file.filename or "unknown",
        reference_date=reference_date,
    )
