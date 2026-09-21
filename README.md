# Vision Service — OCR Test Tool (Prototype)

Local testing tool untuk uji akurasi PaddleOCR pada dokumen Label Bulk/FG dan WI
Filling/Packing, sebelum kita define Fixed ROI per field. **Ini bukan production
code** — tidak ada auth, HTTPS, atau request_id/API contract final.

## Struktur project

```
main.py                       FastAPI backend (PaddleOCR + validasi format)
tube_emboss_pipeline.py       Sub-pipeline: YOLO tube localize + crimp OCR + format validasi
field_patterns.json           Config regex per field_type, dipakai /api/ocr (edit manual, no restart)
emboss_format_patterns.json   Config block-based (day/month/year/batch/mfg_code), dipakai tube emboss pipeline
index.html                    Frontend testing generic (upload/camera + hasil OCR)
tube_emboss.html              Frontend testing khusus tube emboss sub-pipeline
tests/                        Script verifikasi manual (test_ocr.py, test_tube_emboss.py)
tools/                        Script diagnostic/batch dev-only, bukan bagian dari service
images/                       Sample foto testing
models/                       Model weights (YOLO tube detector)
logs/                         Log runtime (gitignored, regenerated)
debug_output/                 Crop debug dari tools/ & tests/test_tube_emboss.py (gitignored, regenerated)
venv/                         Python 3.11 virtualenv
```

## Setup (first time / clone baru)

```powershell
cd C:\laragon\www\vision-service
python -m venv venv        # skip kalau venv sudah ada
.\venv\Scripts\Activate.ps1
pip install paddleocr paddlepaddle opencv-python fastapi uvicorn python-multipart
```

Versi yang sudah tested jalan di venv ini: `paddleocr==3.7.0`,
`paddlepaddle==3.3.1`, `opencv-python==5.0.0.93`, `fastapi==0.141.1`,
`uvicorn==0.53.0`, `python-multipart==0.0.32`.

## Menjalankan backend

```powershell
cd C:\laragon\www\vision-service
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

Tunggu sampai muncul log `PaddleOCR ready.` (load model pertama kali agak
lambat, beberapa detik).

## Menjalankan frontend

Buka `index.html` langsung di browser (double-click, atau lewat ekstensi Live
Server di VS Code). Pastikan field **Backend URL** di panel kiri sudah
`http://127.0.0.1:8000`.

## Cara pakai

1. Pilih **Field type** di dropdown (opsional) — menentukan pattern validasi
   apa yang dipakai (lihat `field_patterns.json`).
2. Upload foto dari file, atau klik **Use camera** untuk ambil foto langsung
   (simulasi kamera HP).
3. Klik **Run OCR**.
4. Lihat hasil: teks terbaca, confidence, kolom **Validation**
   (`FORMAT_OK` / `FORMAT_MISMATCH` / `NOT_APPLICABLE`), kolom **Status**
   gabungan (`OK` / `LOW_CONFIDENCE` / `FORMAT_MISMATCH`), dan bounding box
   digambar di atas gambar (warna ikut status, bukan confidence mentah).
5. Kalau ada catatan ROI (misal field emboss tutup tube), muncul di panel
   kuning **ROI review note(s)** di bawah tabel hasil.

## Menambah / edit field type

Edit `field_patterns.json` langsung — file ini dibaca ulang setiap request,
**tidak perlu restart backend**. Struktur satu entry:

```json
"date_code": {
  "label": "Date/Batch Code (tube cap emboss)",
  "description": "DDMMYY + 4-char alnum batch suffix, e.g. 3108260HB8",
  "pattern": "^\\d{6}[A-Za-z0-9]{4}$",
  "expected_length": 10,
  "roi_hint": "Catatan opsional, muncul di UI saat FORMAT_MISMATCH terjadi pada field ini."
}
```

- `pattern` — regex Python (`re.fullmatch`, harus match keseluruhan teks).
  Set `null` kalau field ini tidak perlu validasi format (selalu
  `NOT_APPLICABLE`, hanya pakai confidence seperti biasa).
- `roi_hint` — catatan yang tampil **hanya saat mismatch terjadi**, biasanya
  dipakai untuk field yang sudah diketahui rawan salah baca karena elemen
  fisik di sekitar teks (contoh: garis knurl/ridge di emboss tutup tube).
  Set `null` kalau tidak ada catatan.
- Teks hasil OCR **tidak pernah** di-strip/trim otomatis oleh sistem —
  dicocokkan apa adanya ke pattern. Kalau mismatch, itu tetap harus direview
  manusia, bukan ditebak/dipotong otomatis.

## Known issues / workarounds

- **oneDNN crash saat inference di CPU**
  (`NotImplementedError: ConvertPirAttribute2RuntimeAttribute...`) — sudah
  di-fix dengan `enable_mkldnn=False` saat konstruksi `PaddleOCR(...)` di
  `main.py`. Jangan dihapus kalau tidak mau error ini balik lagi.
- **Port 8000 sudah terpakai** — kadang proses uvicorn sebelumnya nyangkut
  (terutama kalau server dimatikan paksa/force-kill). Cek dengan:
  ```powershell
  netstat -ano | findstr :8000
  taskkill /F /PID <pid>
  ```
  Kalau `taskkill /IM uvicorn.exe` tidak mempan, cari proses `python.exe`
  dengan memory usage besar (model OCR yang ter-load ada di situ) lewat
  `tasklist /FI "IMAGENAME eq python.exe"` dan kill PID-nya langsung. Atau
  paling gampang, jalankan di port lain: `uvicorn main:app --port 8001` lalu
  update field Backend URL di `index.html`.

## Prinsip desain (jangan diubah tanpa diskusi)

- Vision Service = OCR + color extraction + validasi format saja. **Tidak
  pernah** memutuskan PASS/FAIL — itu keputusan Laravel di `100.100.160.23`
  (tabel audit `ipc_logs`).
- Confidence rendah **ATAU** format tidak sesuai pattern → harus di-route ke
  REVIEW/RESCAN, bukan dipaksa PASS/FAIL.
- Tidak pernah strip/trim/menebak karakter hasil OCR secara otomatis — flag
  ke manusia untuk keputusan akhir.

Lihat `CLAUDE.md` untuk konteks arsitektur lengkap dan progress log.
