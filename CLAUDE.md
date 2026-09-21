# CLAUDE.md — Vision Service (IPC OCR Prototype)

## Project context

Bagian dari sistem IPC (In-Process Control) OCR COSMAX untuk validasi dokumen
produksi di stasiun Filling & Packing. Sistem membaca & cross-validate field
dari dokumen (Label Bulk, Label FG, WI Filling, WI Packing), lalu menentukan
status PASS/FAIL per batch.

**KPI utama:** minimalkan false PASS mendekati nol. Confidence rendah ATAU
format tidak sesuai = harus di-route ke REVIEW/RESCAN, jangan dipaksa
PASS/FAIL.

### Final architecture (sudah settled — jangan diubah tanpa user minta eksplisit)

- **Frontend (production):** React, capture kamera — nanti hosting di
  `100.100.160.23`.
- **Backend (production):** Laravel + MySQL @ `100.100.160.23` — **satu-satunya**
  pembuat keputusan PASS/FAIL, ada tabel audit `ipc_logs`.
- **Vision Service (repo ini):** server terpisah @ `100.100.160.24`, Python,
  **HANYA** OCR + color extraction — **TIDAK PERNAH** membuat keputusan
  PASS/FAIL, cuma return value + confidence + flag validasi.
- **Pendekatan OCR:** Fixed template layout → OpenCV homography alignment →
  Fixed ROI per field → PaddleOCR (pretrained, inference-only, tanpa training
  custom).
- **Fallback:** Gemini Vision API kalau confidence PaddleOCR < 80%.
- **Field warna:** matching swatch pakai Lab color space delta-E, BUKAN OCR
  teks.
- **API contract final (belum diimplementasi):** `POST /api/vision/analyze`
  dengan `request_id` (UUID v4), `field_type`, `image_base64` di request;
  `confidence`, `engine_used`, `status` (OK/LOW_CONFIDENCE/ERROR) di response.

## What exists in this directory right now

Ini **local prototype/testing tool**, bukan Vision Service production. Belum
ada `request_id`, belum ada API contract final, belum ada auth/HTTPS.

| File/Folder | Isi |
|---|---|
| `main.py` | FastAPI backend, PaddleOCR di-load sekali saat startup. Import `tube_emboss_pipeline` langsung (plain import, bukan package) — file ini harus tetap sejajar (sibling) dengan `tube_emboss_pipeline.py` |
| `tube_emboss_pipeline.py` | Sub-pipeline khusus emboss tutup tube: YOLO tube localize → crimp crop → OCR dual-pass (raw + sharpened) → `validate_emboss_format()` block-based (day/month/year/batch/mfg_code) |
| `field_patterns.json` | Config regex per `field_type` (`pattern`, `expected_length`, `description`, `roi_hint`), dipakai `/api/ocr` — dibaca ulang tiap request, edit langsung tanpa restart |
| `emboss_format_patterns.json` | Config block-based (`day`/`month`/`year` = `int_range`, `batch_1..3`/`mfg_code` = `charset`) khusus tube emboss, dipakai `/api/tube-emboss/*` — juga dibaca ulang tiap request |
| `index.html` | Frontend single-file vanilla JS generic: upload file / camera capture (getUserMedia) / Run OCR / tabel hasil (Validation + Status columns) / bounding box overlay / panel ROI hint |
| `tube_emboss.html` | Frontend testing khusus tube emboss sub-pipeline (field_type dari `emboss_format_patterns.json`) |
| `tests/` | Script verifikasi manual: `test_ocr.py` (PaddleOCR + cv2 bisa load), `test_tube_emboss.py` (hit `/api/tube-emboss/analyze`, simpan debug crop ke `debug_output/`) |
| `tools/` | Script diagnostic/batch dev-only (bukan bagian dari service, tidak dipanggil `main.py`): `diagnose_images_folder.py`, `diagnose_sharpening_ab.py` — keduanya `sys.path.insert` ke root supaya bisa `import tube_emboss_pipeline`, output ke `debug_output/` |
| `images/` | Sample foto testing (di-commit, jadi fixture tetap) |
| `models/` | Model weights (YOLO tube detector `tube_detector_v1/best.pt`) |
| `logs/` | Log runtime (`fallback_cases.jsonl`) — gitignored, regenerated |
| `debug_output/` | Crop debug dari `tools/*.py` dan `tests/test_tube_emboss.py` — gitignored, regenerated, jangan commit isinya |
| `README.md` | Instruksi setup & pemakaian |
| `.gitignore` | Exclude `venv/`, `__pycache__/`, cache PaddleX, `logs/`, `debug_output/`, dll |
| `venv/` | Python 3.11 venv. Terinstall: `paddleocr==3.7.0`, `paddlepaddle==3.3.1`, `opencv-python==5.0.0.93`, `fastapi==0.141.1`, `uvicorn==0.53.0`, `python-multipart==0.0.32` |

Endpoint `main.py`:
- `POST /api/ocr` — multipart file upload + optional `field_type` form field →
  jalankan OCR + validasi format, return per deteksi: `text`, `confidence`,
  `box` (quad polygon), `validation_status`, `expected_pattern`,
  `expected_length`, `actual_length`, `roi_hint`, `status`.
- `GET /api/field-types` — list field type dari `field_patterns.json` (frontend
  build dropdown dari sini, tidak hardcode di JS).
- `GET /api/health`.

## Key technical gotchas (jangan re-discover ini lagi)

1. **`paddleocr` 3.7.0 API BEDA TOTAL dari versi 2.x lama.** Pakai
   `ocr.predict(input=...)` (terima path string ATAU numpy array/BGR image
   langsung — tidak perlu tulis file temp). JANGAN pakai `.ocr()` lama. Hasil
   berupa list of dict-like object dengan key: `rec_texts` (list[str]),
   `rec_scores` (list[float] 0–1), `rec_polys` (list of 4-point [x,y] quad,
   numpy int16), `rec_boxes` (axis-aligned [x1,y1,x2,y2]).
2. **oneDNN crash saat inference CPU di mesin ini**: `PaddleOCR(...)` HARUS
   dikonstruksi dengan `enable_mkldnn=False`, kalau tidak `ocr.predict()`
   throw `NotImplementedError: (Unimplemented)
   ConvertPirAttribute2RuntimeAttribute...`. Sudah di-root-cause dan di-fix di
   `main.py` — jangan dihapus.
3. **Format validation adalah layer TERPISAH dari confidence.** Ditambahkan
   karena sample nyata ("13108260HB8", emboss tutup tube, confidence
   92–99%) salah baca — ada karakter tambahan dari garis knurl/ridge di
   sebelah teks, dan confidence tinggi TIDAK menangkap error ini.
   `validate_against_pattern()` di `main.py` **tidak pernah** men-trim/strip
   teks OCR — mismatch pattern SELALU routing ke `FORMAT_MISMATCH` terlepas
   dari confidence, titik. Ini prinsip desain eksplisit dari user (lihat
   bagian bawah) — jangan "membantu" dengan auto-correct atau strip karakter
   di sini.
4. **Windows path gotcha** (kalau testing lewat Bash tool / Git Bash): path
   POSIX-style seperti `/c/Users/.../file.png` yang di-pass ke cv2/Python
   native file API di venv Windows-native GAGAL diam-diam (`cv2.imwrite`
   return `False` tanpa exception, bukan `FileNotFoundError` yang jelas).
   Selalu pakai path Windows proper (`C:\Users\...` atau `C:/Users/...` dengan
   drive letter) untuk file yang akan dibaca/ditulis PaddleOCR/cv2.
5. **Proses "hantu" di port 8000**: pernah ada proses uvicorn/python yang
   selamat dari `taskkill /IM uvicorn.exe` (image-name match tidak selalu
   kena), dan `netstat`/`Get-NetTCPConnection` bisa nunjuk PID yang
   `Get-Process`/`tasklist` sendiri tidak temukan. Kalau port 8000 kelihatan
   "dipakai" tapi prosesnya tidak ketemu, cari kandidat `python.exe` dengan
   memory usage besar (`tasklist /FI "IMAGENAME eq python.exe"` — model OCR
   yang ke-load ada di situ) dan kill PID-nya langsung, atau paling gampang
   jalankan di port lain.
6. **`venv/` sempat ke-stage ~10000 file** sebelum `.gitignore` dibuat (belum
   sempat commit). `.gitignore` sudah menambahkan `venv/`, `__pycache__/`,
   `.paddlex/`, dll — pastikan `git status` bersih dari isi venv sebelum
   commit pertama.
7. **`main.py` dan `tube_emboss_pipeline.py` pakai plain `import`, bukan
   package-relative** (`import tube_emboss_pipeline`, tanpa `from . import`).
   Ini artinya keduanya HARUS tetap sejajar (sibling) di root — jangan
   pindahkan salah satunya ke subfolder tanpa refactor jadi proper package.
   Script di `tools/` (`diagnose_images_folder.py`,
   `diagnose_sharpening_ab.py`) yang butuh `import tube_emboss_pipeline`
   walau lokasinya sudah di subfolder, pakai `sys.path.insert(0,
   str(Path(__file__).parent.parent))` sebelum import — kalau bikin script
   diagnostic baru di `tools/`, ikuti pola yang sama.

## Status / progress log

- [x] venv + paddleocr/opencv/paddlepaddle terinstall & terverifikasi load
- [x] FastAPI + PaddleOCR test backend (`main.py`) dengan endpoint OCR
      (text/confidence/bbox)
- [x] Frontend single-file HTML/JS: upload + camera capture + bbox overlay
- [x] Layer validasi format (`field_patterns.json` +
      `validation_status`/`status`/`roi_hint`) — ditambahkan setelah temuan
      real-sample: misread confidence tinggi pada date code emboss tutup tube
      akibat garis knurl
- [x] `README.md` + `.gitignore` (exclude venv, biar tidak ke-commit)
- [ ] Dokumen template real untuk define Fixed ROI per field
- [ ] OpenCV homography alignment step
- [ ] Lab color space delta-E swatch matching untuk field warna
- [ ] Gemini Vision API fallback untuk confidence < 80%
- [ ] Kontrak `/api/vision/analyze` final (`request_id`, `field_type`,
      `image_base64` in/out) — `/api/ocr` saat ini cuma shortcut testing,
      bukan kontrak final
- [ ] Backend Laravel + tabel audit `ipc_logs` + logic keputusan PASS/FAIL
- [ ] Frontend React untuk capture production
- [ ] Push ke GitHub (repo lokal sudah di-init, belum ada commit — cek
      `.gitignore` sudah benar sebelum commit pertama)

## Prinsip desain (jangan diubah tanpa diskusi eksplisit dengan user)

- Vision Service stateless, **tidak pernah** memutuskan PASS/FAIL — hanya
  return data + flag. Logic keputusan ada di Laravel.
- Minimalkan false PASS di atas segalanya: confidence rendah ATAU format
  mismatch → route ke REVIEW/RESCAN, jangan pernah dipaksa tebak PASS/FAIL.
- Tidak pernah auto-strip/trim/menebak-koreksi teks OCR. Flag ke manusia,
  bukan sistem yang menebak.
- Arsitektur (Laravel @ .23, Vision Service @ .24, PaddleOCR-only/no
  training, Fixed ROI + homography, Gemini fallback <80%, Lab delta-E untuk
  warna) sudah settled — jangan usulkan perubahan tanpa diminta eksplisit.
