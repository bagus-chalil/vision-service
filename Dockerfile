# Vision Service (OCR + LLM-extract) - image produksi untuk server .151 (pola VendorHub).
# CPU only: torch dari index CPU (baris pertama requirements.txt). Python 3.11 = versi
# venv yang sudah divalidasi (CLAUDE.md, Infra #2).
FROM python:3.11-slim

# libgl1 + libglib2.0-0: dibutuhkan opencv-python (non-headless) saat import (Infra #10).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app
ENV HOME=/home/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/home/app/.config/Ultralytics
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chown -R app:app /app /home/app
USER app

# Unduh model PaddleOCR SAAT BUILD (parameter sama dengan main.py) supaya start container
# tidak butuh internet dan tidak mengunduh ulang. Build butuh internet sekali.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True, enable_mkldnn=False)"

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=10 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# 1 proses (model dimuat sekali; lihat CLAUDE.md gotcha #16 soal concurrency).
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
