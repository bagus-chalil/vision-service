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
| `main.py` | FastAPI backend, PaddleOCR di-load sekali saat startup. Import `tube_emboss_pipeline` langsung (plain import, bukan package) — file ini harus tetap sejajar (sibling) dengan `tube_emboss_pipeline.py`. `/api/analyze` adalah entrypoint unified — dispatch ke pipeline generic atau tube_emboss berdasarkan field_type, lihat bagian Endpoint di bawah |
| `tube_emboss_pipeline.py` | Sub-pipeline khusus emboss tutup tube: YOLO tube localize → crimp crop → OCR dual-pass (raw + sharpened) → `validate_emboss_format()` block-based (day/month/year/batch/mfg_code) |
| `field_patterns.json` | Config regex per `field_type` (`pattern`, `expected_length`, `description`, `roi_hint`) untuk pipeline GENERIC (whole-image OCR, no localization) — dibaca ulang tiap request, edit langsung tanpa restart. `date_code` sudah DIHAPUS dari sini (Sep 2026) karena digantikan `tube_emboss_default` di `emboss_format_patterns.json` yang localization-nya benar — jangan tambahkan lagi field tube-emboss-related di file ini |
| `emboss_format_patterns.json` | Config block-based (`day`/`month`/`year` = `int_range`, `batch_1..3`/`mfg_code` = `charset`) untuk pipeline TUBE_EMBOSS (YOLO localize + crop sebelum OCR) — juga dibaca ulang tiap request. Field_type dengan key di file ini otomatis di-route `/api/analyze` ke pipeline ini, TIDAK lewat field_patterns.json. Ada 2 entry: `tube_emboss_default` (MFD+batch+mfg, 10 char) dan `tube_exp_date` (EXP-only, 6 char DDMMYY, baris emboss terpisah dari MFD — lihat flow reference_date di bawah) |
| `index.html` | Frontend UNIFIED single-file vanilla JS: 1 dropdown field_type gabungan dari kedua config di atas, upload file/camera capture, tombol Run → `POST /api/analyze` → render otomatis sesuai `data.pipeline` (tabel per-detection untuk generic, atau kv single-result + tube/crimp crop preview untuk tube_emboss) |
| `tube_emboss.html` | Frontend standalone khusus tube emboss sub-pipeline (dev/debug only, hit `/api/tube-emboss/analyze` langsung) — dipertahankan terpisah dari `index.html`, tidak wajib dipakai user akhir |
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
- `POST /api/analyze` — **entrypoint utama, dipakai `index.html`.** Multipart
  file + optional `field_type`/`request_id`/`debug`/`reference_date` form
  field. Cek apakah `field_type` ada di `emboss_format_patterns.json` → kalau
  ya, jalankan `run_tube_emboss_pipeline()` (hasil: `{pipeline: "tube_emboss",
  ...}`, bentuk sama seperti `/api/tube-emboss/analyze`); kalau tidak,
  jalankan `run_generic_pipeline()` (hasil: `{pipeline: "generic", ...}`,
  bentuk sama seperti `/api/ocr`). Field baru (WI/dll) yang butuh
  localization sendiri didaftarkan dengan pola yang sama: tambah config file +
  pipeline function-nya, lalu cek membership di config itu di `analyze()`.
  `reference_date` (DDMMYY, opsional) cuma dipakai pipeline tube_emboss —
  lihat item date_check di bawah.
- `GET /api/field-types` — **dipakai `index.html`.** Merge
  `field_patterns.json` (tag `pipeline: "generic"`) +
  `emboss_format_patterns.json` (tag `pipeline: "tube_emboss"`) jadi satu
  list, supaya frontend bisa build 1 dropdown tanpa hardcode field key di JS.
- `POST /api/ocr` — endpoint generic-only, dipertahankan untuk backward
  compat (dipanggil `run_generic_pipeline()` yang sama dengan `/api/analyze`).
- `POST /api/tube-emboss/analyze`, `GET /api/tube-emboss/field-types` —
  endpoint khusus tube emboss, dipakai `tube_emboss.html` (standalone) dan
  `tests/test_tube_emboss.py`. `request_id` wajib di endpoint ini (beda dari
  `/api/analyze` yang auto-generate kalau kosong).
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
8. **`tube_emboss_default` di `emboss_format_patterns.json` BUKAN format
   universal untuk semua SKU** — dikonfirmasi Ganjar Ahfadan (PPIC, chat
   Teams 2026-09-21): (a) 1-digit manufacture code tidak selalu `8`, ada
   kemungkinan SKU lain dengan manufacture non-Cosmax yang formatnya beda
   total (bukan cuma beda digit terakhir) — kalau ketemu, daftarkan sebagai
   entry baru di config ini, jangan modifikasi `tube_emboss_default`; (b)
   3-char batch block bisa gabung huruf+angka (misal `AL1`, `AL2`), sudah
   di-fix dengan ganti charset `batch_1..3` dari `[A-Za-z]` jadi
   `[A-Za-z0-9]`. Karena Vision Service tidak pernah mutuskan PASS/FAIL,
   over-strict validation di sini cuma nambah noise ke REVIEW queue (arah
   aman), bukan bikin false PASS — jadi kalau nemu variasi format lain,
   default-nya perluas config, jangan asumsikan satu schema ini benar buat
   semua tube.
9. **`tube_exp_date` (EXP-only emboss) beda baris fisik dari
   `tube_emboss_default` (MFD+batch+mfg)**, walau sama-sama di crimp/shoulder
   tube — dari sample foto real (2026-09-24), keduanya bisa muncul di foto
   yang sama tapi baris terpisah. `date_check` (reference_date cross-check di
   `tube_emboss_pipeline.check_reference_date()`) TIDAK menghitung
   EXP = tanggal_mixing + shelf-life — cuma exact string match DDMMYY vs
   DDMMYY, tujuannya ngetes akurasi baca OCR pakai ground-truth yang sudah
   diketahui (tanggal mixing diinput manual saat testing), BUKAN aturan
   bisnis nyata bahwa EXP harus sama dengan tanggal mixing. Cross-check ini
   cuma jalan kalau `format_valid` True dulu (FORMAT_MISMATCH selalu skip
   date_check, gapeduli reference_date-nya ada) — tetap flag data doang,
   Vision Service tetap tidak pernah mutuskan match/mismatch jadi PASS/FAIL.
10. **`tube_exp_date` pindah dari fixed-crop-position ke `label_anchor`
    (2026-09-24)** — crimp-crop top-15%-of-tube-bbox (`CRIMP_CROP_TOP_FRACTION`)
    framing-dependent: dari sample foto asli user, crop itu ikut menelan baris
    body-text kemasan ("EXP 051228" jadi kebaca gabung sama
    "Elsheskin Barrier+pH Balance..." → FORMAT_MISMATCH walau confidence
    tinggi). Fix: field_type dengan `"label_anchor": "EXP"` di
    `emboss_format_patterns.json` sekarang lewat jalur baru
    (`analyze_label_anchor_field()` / `find_label_anchor_match()` di
    `tube_emboss_pipeline.py`) — OCR seluruh area (tube crop kalau YOLO
    detect tube, fallback ke full frame kalau tidak — EXP juga muncul di tutup
    botol bulat yang YOLO tube-detector tidak akan pernah kenali), lalu cari
    per-region (bukan gabungan/concat semua region) teks yang match
    `EXP\.?\s*:?\s*(\d{6})` - ini yang bikin batch/lot code sebelum "EXP"
    (`"FJZ EXP.130927"`) atau suffix code sesudah 6 digit (`"EXP.150728 TU"`)
    gak ikut ngerusak match, tanpa pernah strip/edit 6 digit yang match.
    Confirmed fix di sample Elsheskin tube asli (raw text jadi bersih
    `"EXP 210428"`, confidence 99.8%). **Known gap**: kalau baris EXP dan MFD
    di-emboss BERDEKATAN banget (sample Pond's UV Protect, 2026-09-24),
    PaddleOCR text-DETECTION-nya sendiri (bukan recognition) kadang
    menggabungkan 2 baris itu jadi SATU region sebelum OCR sempat
    baca — `label_anchor` search gak bisa misahin lagi kalau sudah kegabung di
    level ini. Perbaikan butuh instance PaddleOCR kedua dengan
    `text_det_unclip_ratio` lebih kecil (constructor-only param, tidak bisa
    di-override per-call di `predict()`), yang bentrok sama constraint
    "jangan construct PaddleOCR dua kali" di `main.py` — belum digarap,
    next step kalau kasus ini kejadian di produksi.

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
- [x] Tube emboss sub-pipeline (`tube_emboss_pipeline.py`): YOLO11n tube
      localize → crimp crop → OCR dual-pass (raw + sharpened) → format
      validation block-based, ditambah `tube_emboss.html` standalone test
      page + Gemini fallback stub (belum API asli, lihat item Gemini di
      bawah)
- [x] Unified `/api/analyze` + `/api/field-types` di `main.py` — 1
      dropdown/1 endpoint di `index.html` yang otomatis dispatch ke pipeline
      generic atau tube_emboss berdasarkan field_type, supaya UI tidak perlu
      tahu field mana butuh localization dan mana tidak. `date_code` di
      `field_patterns.json` sudah dihapus (digantikan `tube_emboss_default`)
- [x] Field `tube_exp_date` (EXP-only emboss, DDMMYY, reuse pipeline
      tube_emboss) + `reference_date` cross-check (`date_check` di response) —
      alur: QC input tanggal mixing (ground-truth, dikonversi ke DDMMYY di
      frontend dari `<input type=date>`) → OCR baca EXP → kalau format_valid,
      dibandingkan exact-match ke reference_date. Ditambahkan di
      `index.html` + `tube_emboss.html`.
- [x] `tube_exp_date` di-uji ke foto real "gambar lengkap" (2026-09-24) —
      crimp-crop fixed-percentage ternyata framing-dependent (lihat item 9 &
      10 di Key technical gotchas), diganti pendekatan `label_anchor`
      (cari literal "EXP" + 6 digit per-region OCR result, bukan posisi crop
      tetap). Confirmed fix di sample tube asli user. Known gap: baris
      EXP+MFD yang di-emboss berdekatan (sample Pond's) masih bisa
      ketelen jadi satu region oleh PaddleOCR text detector sebelum
      `label_anchor` sempat misahin - butuh instance PaddleOCR kedua dengan
      `text_det_unclip_ratio` lebih kecil, belum digarap (lihat item 10)
- [ ] Pipeline localization khusus untuk `wo_number` dan `batch_no` — saat
      ini masih numpang di pipeline generic (whole-image OCR + regex, tanpa
      ROI/localization apa pun), jadi rawan false FORMAT_MISMATCH kalau ada
      teks lain di foto yang kena regex-nya juga. Sama seperti tube emboss:
      butuh strategi localization sendiri (kemungkinan Fixed ROI + homography
      dulu kalau dokumennya flat/template tetap), baru didaftarkan sebagai
      pipeline baru menyusul pola `tube_emboss_default`
- [ ] Dokumen template real untuk define Fixed ROI per field (WI/Label
      Bulk/FG/exp date, dll — field selain tube emboss)
- [ ] OpenCV homography alignment step
- [ ] Lab color space delta-E swatch matching untuk field warna
- [ ] Gemini Vision API fallback untuk confidence < 80% — stub sudah ada di
      `tube_emboss_pipeline.py` (`gemini_vision_fallback_stub`), belum
      terhubung ke API Gemini asli
- [ ] Kontrak `/api/vision/analyze` final (`request_id`, `field_type`,
      `image_base64` in/out) — `/api/analyze` saat ini cuma shortcut testing
      lokal (multipart file, bukan base64; belum ada auth/HTTPS), bukan
      kontrak final
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
- `field_type` **selalu dikirim sebagai parameter dari luar** (aplikasi
  pemanggil/workflow context), Vision Service **tidak pernah** menebak
  field_type dari isi foto (no image classification/auto-detect). Di
  production nanti, React frontend yang tahu field_type dari step alur kerja
  saat itu (user tinggal foto, gak perlu pilih apa-apa manual) — dropdown
  field type di `index.html`/`tube_emboss.html` cuma alat testing prototype,
  bukan bagian dari desain production. Alasan: auto-classify dari foto
  berisiko salah pilih format validasi secara diam-diam (lebih parah dari
  sekadar length-mismatch yang kelihatan jelas sebagai FORMAT_MISMATCH).
