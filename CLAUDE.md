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
| `field_patterns.json` | Config regex per `field_type` (`pattern`, `expected_length`, `description`, `roi_hint`) untuk pipeline GENERIC (whole-image OCR, no localization) — dibaca ulang tiap request, edit langsung tanpa restart. `date_code` sudah DIHAPUS dari sini (Sep 2026) karena digantikan `tube_emboss_default` di `emboss_format_patterns.json` yang localization-nya benar — jangan tambahkan lagi field tube-emboss-related di file ini. `wo_number`/`batch_no` sekarang punya `"pilot_ready": false` (2026-09-25, lihat status log) — field ini masih numpang generic pipeline tanpa localization, jadi disembunyikan dari dropdown pilot secara default |
| `emboss_format_patterns.json` | Config block-based (`day`/`month`/`year` = `int_range`, `batch_1..3`/`mfg_code` = `charset`) untuk pipeline TUBE_EMBOSS (YOLO localize + crop sebelum OCR) — juga dibaca ulang tiap request. Field_type dengan key di file ini otomatis di-route `/api/analyze` ke pipeline ini, TIDAK lewat field_patterns.json. Ada 3 entry: `tube_emboss_default` (MFD+batch+mfg, 10 char), `tube_exp_date` (EXP-only, 6 char DDMMYY, baris emboss terpisah dari MFD — lihat flow reference_date di bawah), dan `tube_mfd_date` (MFD print di body, `label_anchor: "MFD"`, `digit_fallback: false`, `exp_from_shelf_life: true` — lihat gotcha #18) |
| `index.html` | Frontend UNIFIED single-file vanilla JS: 1 dropdown field_type gabungan dari kedua config di atas, upload file/camera capture, tombol Run → `POST /api/analyze` → render otomatis sesuai `data.pipeline` (tabel per-detection untuk generic, atau kv single-result + tube/crimp crop preview untuk tube_emboss) |
| `tube_emboss.html` | Frontend standalone khusus tube emboss sub-pipeline (dev/debug only, hit `/api/tube-emboss/analyze` langsung) — dipertahankan terpisah dari `index.html`, tidak wajib dipakai user akhir |
| `tests/` | Script verifikasi manual: `test_ocr.py` (PaddleOCR + cv2 bisa load), `test_tube_emboss.py` (hit `/api/tube-emboss/analyze`, simpan debug crop ke `debug_output/`) |
| `tools/` | Script diagnostic/batch dev-only (bukan bagian dari service, tidak dipanggil `main.py`): `diagnose_images_folder.py` (tube_emboss_default), `diagnose_exp_date_folder.py` (tube_exp_date, ditambahkan 2026-09-25), `diagnose_sharpening_ab.py` — semua `sys.path.insert` ke root supaya bisa `import tube_emboss_pipeline`, output ke `debug_output/` (kecuali `diagnose_exp_date_folder.py` yang cuma print summary table, gak nyimpen crop) |
| `integrations/laravel-ipc/` | Bundle kode SISI LARAVEL (bukan Python) untuk `ipc_app` (`C:\laragon\www\new_trial_validation_app\ipc_app`, Laravel 12 + Inertia React): `files/` = mirror root `ipc_app` (semua file baru, tinggal copy), `README.md` = langkah integrasi + 3 edit manual. Di sinilah keputusan PASS/FAIL/REVIEW dibuat (`VisionEvaluator`), EXP = MFD + `master_products.shelf_life_months`, dan audit `ipc_logs`. Ditambahkan 2026-09-28, lihat status log |
| `llm_extract.py` | Ekstraksi via LLM (Ollama, vision): `analyze_label_bulk()`, `analyze_im_document()`, `analyze_coding_print()` (coding NON-TUBE, gotcha #20), `analyze_tube_exp()` (EXP crimp tube, cascade, gotcha #22). Plain import sibling `main.py`, didaftarkan lewat `LLM_FIELD_TYPES` di `main.py`. Config via env (`OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_THINK`, `LABEL_VIEW_SIDES`/`IM_VIEW_SIDES`/`CODING_VIEW_SIDES`, dst) — lihat gotcha #19 |
| `tools/evaluate_label_bulk.py` | Evaluasi akurasi per field + kecepatan `label_bulk` vs kunci jawaban (`tools/eval_data/label_bulk_truth.json`, `--truth` untuk file lain). Membedakan WRONG_CONFIDENT (lolos gerbang tapi salah = berbahaya) dari salah-tapi-ter-flag |
| `tools/evaluate_coding_print.py` | Evaluasi `coding_print`. `--truth` menerima kunci bacaan foto (`batch`/`exp`) ATAU kunci data DB (`db_batch`/`db_exp_*`; OK yang beda DB = `OK_BEDA_DB_CEK_VISUAL`, wajib dicek mata, karena coding bisa memang beda dari DB) |
| `testdata_private/` | Foto produksi + kunci jawaban privat untuk eval. **GITIGNORED, jangan pernah di-commit** (repo punya remote GitHub). Isi: `date_bulk/` (Label Bulk), `im_number/` (IM), `label_bulk_truth.json`, `coding_print/` (foto coding, nama = `ipc_attachments.id`; `_set2_db.json` = kunci DB set 2), `coding_print_truth.json` (kunci bacaan foto set 1). `tube/` (2026-11-12, `tools/extract_tube_dataset.py`): `photos/` = 405 foto `image_tube` (filling) dari DB + 36 fixture `images/`, `manifest.json` = kunci DB per foto. CATATAN: `image_tube` DB mayoritas foto produk utuh/jauh (banyak BUKAN tube, tanpa emboss terbaca) - belum dipilah. `tube/photos_candidates/` (235 foto, `tools/classify_tube_candidates.py` -> `tube/classify.json`) = foto coding packing/startup yang Qwen (1 view, penyaring awal, belum dicek penuh) nilai TUBE; 175 dengan kode emboss di crimp, 150 di antaranya ada garis hitam menimpa kode (sensor/eye mark, posisi sama di semua tube - lihat gotcha #14). Baseline 2026-11-12 (`tools/baseline_tube_exp.py`, jalur Ollama `coding_print`, 175 foto tube berkode crimp, kunci DB): 5 OK (semua cocok DB), 170 LOW, 0 false OK; garis hitam 1/128 OK vs tanpa garis 4/47 OK. `tube_exp_date` (PaddleOCR) BELUM diukur di set ini (CPU, bukan di laptop). Prototype mode topeng (`tools/proto_masked_exp.py`, model baca EXP 6 char dgn '?' tanpa diberi nilai DB, 3 view per-posisi, cocok ke DB dgn wildcard): FULL_MATCH 23, PARTIAL_CONSISTENT 42, CONFLICT 26, NO_SIGNAL 84 (dari 175). TEMUAN: model HALUSINASI di bawah garis hitam walau 3 view konsisten (mis. 7723 baca '9' di posisi tertutup) -> CONFLICT bisa konflik palsu, jangan dipakai utk FAIL; perlu deteksi garis hitam via OpenCV utk memaksa posisi di bawahnya '?' (belum dikerjakan). Detektor garis hitam OpenCV DICOBA & DIBUANG (2026-11-12): hanya menemukan 54 dari ±128 foto bergaris (garis cuma ±3 px di foto jauh, butuh lokalisasi crimp dulu) + false positive (stiker kuning, huruf logo) - flag Qwen `black_mark_over_code` justru lebih akurat di cek visual. Garis ini tercetak di crimp SEMUA tube yang dilihat, jadi kebijakan yang diusulkan: ANGGAP selalu ada, 1 posisi selalu tak terverifikasi (toleransi 1 beda -> REVIEW, bukan konflik; FULL_MATCH bukan bukti kuat). Dari 26 CONFLICT: 11 beda 1 posisi, 15 beda >=2. Prototype BELUM masuk main.py/llm_extract.py |
| `images/` | Sample foto testing (di-commit, jadi fixture tetap) |
| `models/` | Model weights (YOLO tube detector `tube_detector_v1/best.pt`) |
| `logs/` | Log runtime (`fallback_cases.jsonl`) — gitignored, regenerated |
| `debug_output/` | Crop debug dari `tools/*.py` dan `tests/test_tube_emboss.py` — gitignored, regenerated, jangan commit isinya |
| `README.md` | Instruksi setup & pemakaian |
| `.gitignore` | Exclude `venv/`, `__pycache__/`, cache PaddleX, `logs/`, `debug_output/`, dll |
| `venv/` | Python 3.11 venv. Isi persis sama seperti `requirements.txt` (lihat itemnya sendiri) |
| `requirements.txt` | Pinned `pip freeze` dari venv yang sudah tested (termasuk `ultralytics`/`torch`/`torchvision` yang sebelumnya cuma kesebut di README, bukan file requirements beneran). Baris pertama `--extra-index-url https://download.pytorch.org/whl/cpu` **WAJIB ada** — lihat gotcha Infra #1 |
| `.gitlab-ci.yml` | Pipeline CI/CD (stage `validate` + `deploy`), di-gate `CI_COMMIT_BRANCH == "production"`, jalan di GitLab Runner yang tag-nya `vps-aiocr` (satu-satunya runner di project ini, terdaftar langsung di VPS target) |
| `.github/workflows/mirror-to-gitlab.yml` | Push-mirror branch `production` doang dari GitHub ke project GitLab (`cosmaxidn/vision-service`) tiap ada push — lihat section Infra di bawah kenapa polanya begini |
| `ops/provision_vps.sh` | Bootstrap sekali-jalan (idempotent) buat VPS pilot Linux baru — install Python 3.11, swap, user `visionsvc`, GitLab Runner, systemd unit, sudoers sempit |
| `ops/deploy.sh` | Script privileged TUNGGAL yang boleh dijalankan GitLab Runner via sudo — sync kode, install deps, smoke test (`tests/test_ocr.py`), baru restart service. Di-install ke `/usr/local/bin/vision-service-deploy.sh` oleh `provision_vps.sh` |
| `ops/vision-service.service` | systemd unit Linux, setara `ops/run_service.ps1` versi Windows |
| `ops/configure_firewall.sh` | ufw allowlist per-IP untuk VPS Linux, setara `ops/configure_firewall.ps1` versi Windows |
| `ops/run_service.ps1`, `ops/health_check.ps1`, `ops/register_tasks.ps1`, `ops/configure_firewall.ps1` | Versi Windows (dipakai di server on-prem `.24`, BUKAN untuk VPS pilot Linux — lihat item Deploy di README) |

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
    `"EXP 210428"`, confidence 99.8%). Known gap kalau baris EXP dan MFD
    di-emboss BERDEKATAN banget (sample Pond's UV Protect, 2026-09-24) sudah
    RESOLVED via `find_zoom_retry_match()` — lihat item 12.
11. **`label_anchor` (item 10) diperluas 3x lagi (2026-09-24) setelah tes ke
    sample foto real** — semua di `tube_emboss_pipeline.py`:
    - **Digit fallback tanpa label "EXP" sama sekali** (`find_digit_fallback_match()`):
      sample tutup tube nyata ("AJA120927 PU") ternyata gak ada kata "EXP"-nya
      di embossnya sama sekali, cuma digit polos — beda layout fisik dari 3
      sample yang jadi basis `label_anchor` di awal. Kalau literal-label
      search gagal total di kedua pass (raw + sharpened), sekarang dicoba
      cari run 6-digit yang TERISOLASI (`(?<!\d)(\d{6})(?!\d)` — gak bakal
      kepotong dari tengah kode 10-digit `tube_emboss_default`) dan yang lolos
      validasi day/month/year field_type itu sendiri. Kalau ketemu, confidence
      SELALU di-cap di bawah `GEMINI_FALLBACK_THRESHOLD` (gak pernah jadi OK
      diam-diam, karena gak ada anchor tekstual sama sekali — cuma
      LOW_CONFIDENCE dengan `match_method: "digit_fallback"` di response buat
      transparansi ke consumer). Kalau ada >1 kandidat digit valid yang beda
      → tetap `DIGIT_FALLBACK_AMBIGUOUS`, gak pernah nebak salah satu.
    - **Format tanggal titik + tahun 4-digit** (`LABEL_ANCHOR_DOTTED_PATTERN_TEMPLATE`):
      sample label cetak (botol "if you inbalance") pakai `"EXP: 27.01.2029"`
      (DD.MM.YYYY, bukan DDMMYY 6-digit nempel). `find_label_anchor_match()`
      sekarang coba pattern compact dulu, baru dotted kalau gak match — tahun
      dibatasi harus mulai `20` (gak nebak abad kalau OCR baca beda), lalu
      cuma buang prefix "20" itu buat masuk ke block schema 2-digit-year yang
      sudah ada (day+month+2-digit-year tetap dari digit yang sama persis
      yang diketik OCR, gak pernah diedit/ditebak). Ini pattern yang akhirnya
      match sample Pond's setelah fix di item 12 (`date_format:
      "dotted_4digit_year"`, lewat `find_zoom_retry_match()` bukan langsung
      dari raw/sharpened pass — lihat item 12 untuk kenapa raw/sharpened saja
      gak cukup).
    - **`LABEL_ANCHOR_MIN_TUBE_CONFIDENCE` (0.5, lebih tinggi dari
      `YOLO_MIN_CONFIDENCE` 0.25 punya jalur default)**: sample tutup botol
      kuning ("EXP.150728 TU") dapat deteksi "tube" YOLO 41.6% confidence
      yang salah total — nge-crop ke objek blur di background, bukan ke tutup
      botolnya, jadi OCR gak baca apa-apa (`raw_ocr_text: null`). Di jalur
      `label_anchor` khusus, false-accept (pakai bbox lemah yang salah) jauh
      lebih mahal dari false-reject (fallback ke whole-frame, yang memang
      sudah didesain robust — lihat item 10), makanya ambang kepercayaan
      buat TRUST crop-nya dinaikkan ke 0.5 khusus di jalur ini
      (`analyze_label_anchor_field()` manggil `localize_tube(..., min_confidence=LABEL_ANCHOR_MIN_TUBE_CONFIDENCE)`),
      jalur default `tube_emboss_default` tetap pakai `YOLO_MIN_CONFIDENCE`
      lama karena di situ bbox yang salah fatal juga baik diterima maupun
      ditolak. Confirmed fix di sample yang sama (sekarang ketemu
      `"EXP.150728 TU"` via whole-frame fallback). Regression-tested ke 32
      foto sample di `images/` — nggak ada sample lain yang jadi rusak gara2
      threshold ini naik.
12. **`find_zoom_retry_match()` (2026-09-24) — last-resort fallback buat
    `label_anchor` fields, dicoba setelah `find_label_anchor_match()` DAN
    `find_digit_fallback_match()` sama-sama gagal di raw+sharpened pass.**
    Motivasi: sample Pond's/Elsheskin (item 10/11's "Known gap") di
    whole-tube-crop resolution ternyata gak benar-benar "kegabung jadi satu
    region" seperti dugaan awal — piece "EXP:" tetap kedetect sendiri, cuma
    TANPA digit nempel (digitnya kepotong/fused ke piece tanggal MFD di
    sekitarnya, karena baris MFD/EXP/LOT di-emboss rapat banget). Fix-nya:
    begitu ketemu piece yang mengandung label literal (misal "EXP:") tapi
    tanpa 6-digit match, crop ulang band vertikal generous di sekitar posisi
    piece itu dari gambar ASLI (`ZOOM_RETRY_VERTICAL_PADDING_FACTOR` = 1.5x
    tinggi piece-nya), upscale ke `ZOOM_RETRY_UPSCALE_TARGET_HEIGHT` (200px),
    lalu re-OCR band itu sendirian dengan `text_det_box_thresh` lebih rendah
    (`ZOOM_RETRY_BOX_THRESH` = 0.3, vs default ~0.45) supaya box teks
    padat/kecil gak didrop duluan oleh detector. Re-OCR band kecil ini
    (bukan whole-image) yang bikin digit dan labelnya kebaca sebagai
    detection terpisah dengan resolusi cukup — tanpa PaddleOCR instance
    kedua dan tanpa pernah edit/tebak digit yang terbaca, cuma kasih
    detector "kesempatan kedua" di resolusi lebih tinggi.

    Karena band hasil re-OCR ini masih berisi >1 baris fisik (MFD/EXP/LOT),
    hasil `ocr_text_pieces()`-nya perlu di-cluster per baris dulu
    (`group_zoomed_pieces_into_lines()`) sebelum dilempar ke
    `find_label_anchor_match()` yang biasa — beda dari cara kerja normal
    `find_label_anchor_match()`/`find_digit_fallback_match()` yang sengaja
    TIDAK pernah menggabung antar-detection (lihat `ocr_text_pieces()`
    docstring), karena di jalur zoom_retry band-nya sudah pasti cuma
    1-3 baris jadi risiko nggabung teks yang gak berhubungan jauh lebih
    kecil. Clustering ini sendiri melewati 2 desain gagal sebelum settled:
    (1) range-overlap (gabungin cluster kalau bounding-box-nya overlap) —
    chain-merge SEMUA baris jadi satu group raksasa, karena baris-baris yang
    rapat itu bounding box-nya emang udah overlap satu sama lain; (2)
    fixed-seed (bandingkan tiap piece ke piece PERTAMA yang jadi representasi
    cluster) — betulin chain-merge, tapi bikin bug baru persis di sample
    Pond's: piece digit "27.01.2029" (y-center 98) cuma 3px dari piece
    "EXP:" (101), TAPI piece MFD ("20.01.2026", y-center 68.5) kepop duluan
    jadi seed karena posisinya paling atas, dan jarak "27.01.2029" ke seed
    MFD itu (29.5px) kebetulan masih di bawah threshold (30px) — jadi
    ke-klaim MFD duluan sebelum "EXP:" sempat jadi seed sendiri. Fix final
    (`group_zoomed_pieces_into_lines()` sekarang): urutkan semua piece by
    y-center, lalu grouping SEQUENTIAL — tiap piece cuma dibandingkan ke
    piece SEBELUMNYA di urutan itu (bukan ke seed tetap, bukan ke range
    cluster yang membesar), mulai group baru begitu gap-nya lewat threshold
    (`0.5 * min(tinggi kedua piece)`). Ini imun ke kedua bug sebelumnya:
    gak ada range yang membesar (jadi gak chain-merge), dan gak ada
    urutan-pop yang nentuin siapa "menang" klaim suatu piece (jadi
    "27.01.2029" kebanding ke tetangga langsungnya di urutan, bukan ke seed
    yang kebetulan lebih dulu diproses).

    **Confirmed fix** di sample asli (`WhatsApp Image 2026-09-24 at
    11.31.20.jpeg`, Pond's/Elsheskin): sebelumnya `LABEL_NOT_FOUND` total,
    sekarang `EXP:27.01.2029` → extracted `270129`, `match_method:
    "zoom_retry"`, `date_format: "dotted_4digit_year"`, `FORMAT_OK`,
    confidence di-cap 0.84 (`LOW_CONFIDENCE`, sesuai desain — zoom_retry
    match TIDAK PERNAH jadi confident `OK` diam-diam, sama seperti
    `digit_fallback`, karena ini re-baca band yang tadinya gagal, bukan
    baca langsung yang bersih). Regression-tested ke semua 32 foto di
    `images/` (field_type `tube_exp_date`): 31 foto lainnya hasilnya
    IDENTIK sebelum/sesudah fix ini (status/extracted/confidence/
    match_method sama persis) — cuma sample Pond's yang berubah, dari
    `LABEL_NOT_FOUND` jadi resolved. Known gap item 10/11 soal sample
    Pond's ini sekarang RESOLVED.
13. **`find_digit_fallback_match()` dipanggil SEBELUM `find_zoom_retry_match()`
    di `recognize_label_anchor_dual()` sampai 2026-09-24 — urutannya DIBALIK**
    (sekarang zoom_retry duluan, baru digit_fallback) setelah user melaporkan
    hasil salah di sample tube Pond's UV Protect ("EXP 140627" di
    crimp/shoulder + "MFD 140624 8 QFZ" tercetak di body, satu foto): OCR
    pecah "EXP" dan "140627" jadi 2 piece terpisah (fisik saling menempel,
    tapi PaddleOCR tetap deteksi sebagai 2 box beda), sementara
    "MFD 140624 8 QFZ" kebaca bersih sebagai 1 piece. `find_digit_fallback_match()`
    SAMA SEKALI tidak tahu ada literal "EXP" di foto — dia cuma ambil isolated
    6-digit run mana pun yang lolos validasi day/month/year — jadi begitu dia
    jalan duluan dan (di capture tertentu, misal setelah kompresi kamera)
    cuma nemu SATU kandidat valid ("140624" dari MFD, karena piece "140627"
    kebetulan tidak lolos/tidak terdeteksi di capture itu), dia langsung
    confidently return itu sebagai tanggal EXP — padahal itu tanggal MFD,
    salah field sama sekali — dan `find_zoom_retry_match()` (yang justru
    didesain khusus buat kasus "EXP" ketemu tapi digitnya kepisah) tidak
    pernah kebagian giliran coba dulu karena digit_fallback sudah "sukses"
    (non-ambiguous) lebih dulu. Fix: `find_zoom_retry_match()` sekarang
    dicoba LEBIH DULU begitu `find_label_anchor_match()` gagal di raw+sharp
    pass — karena zoom_retry hanya jalan kalau ada piece yang literally
    mengandung label ("EXP"), dia costless kalau labelnya memang tidak ada
    sama sekali (langsung `{"matched": False}`), tapi kasih kesempatan
    pertama ke anchor yang benar setiap kali labelnya ADA, sebelum
    digit_fallback yang buta-konteks sempat menebak dari digit manapun.
    `find_digit_fallback_match()` tetap jalan sebagai last-resort kalau
    zoom_retry juga gagal (tidak dihapus, cuma didemosi 1 langkah).

    **Regression-tested** ke sample Pond's yang sama persis (file PNG asli,
    tanpa kompresi kamera): urutan lama maupun baru sama-sama menghasilkan
    `DIGIT_FALLBACK_AMBIGUOUS` (karena di file asli ini, `find_zoom_retry_match()`
    JUGA gagal — re-OCR band yang di-zoom salah baca karakter emboss "4"
    jadi "A" akibat noise garis knurl di sekitarnya, `"EXP1A0"` + `"627"`,
    sesuai gotcha #3 — jadi tetap tidak match pattern EXP+6digit persis).
    Tapi ini justru hasil yang BENAR secara desain: sebelumnya (sesuai
    laporan user, kemungkinan dari capture kamera browser yang lebih
    terkompresi, bukan file PNG asli) sistem confidently return `"140624"`
    (tanggal MFD, SALAH) sebagai jawaban `FORMAT_OK`/`LOW_CONFIDENCE` — versi
    baru berubah jadi `DIGIT_FALLBACK_AMBIGUOUS` (kedua kandidat "140627" DAN
    "140624" ketemu, tidak bisa dibedakan aman, jadi di-flag ambiguous) —
    dari "salah tapi kelihatan valid" jadi "tidak yakin, minta REVIEW", sesuai
    prinsip minimalkan false PASS.

    **UPDATE 2026-09-25 — opsi lanjutan di atas TERNYATA SUDAH diimplementasikan**
    di commit yang sama (`92ac7bd`), cuma catatan ini belum di-sync: lihat
    `DIGIT_FALLBACK_EXCLUDED_LABELS = ("MFD",)` di `find_digit_fallback_match()`
    — piece yang mengandung literal "MFD" sekarang di-skip total sebelum
    masuk pencarian isolated-6-digit, jadi kasus di atas seharusnya sudah
    resolve ke `"140627"` (bukan cuma ambiguous) SELAMA "140627" itu sendiri
    berhasil terbaca lengkap di salah satu pass. Confirmed re-test 2026-09-25
    ke foto asli (`3.jpg`, real sample, bukan PNG lama) via `/api/analyze`
    field_type `tube_exp_date`: exclusion-nya jalan (MFD's "140624" tidak
    pernah muncul jadi kandidat), TAPI hasilnya tetap `LABEL_NOT_FOUND` untuk
    sample INI karena masalah baru/berbeda — lihat gotcha #14: 3 digit
    pertama emboss EXP ("140") gak terbaca sama sekali (bukan grup-tapi-gagal,
    bukan ambiguous, literally invisible ke OCR) di raw, sharpened, MAUPUN
    zoom_retry, jadi tidak ada kandidat 6-digit lengkap sama sekali untuk
    di-exclude atau dipilih. Exclusion MFD ini tetap kerja seperti didesain,
    cuma gak cukup buat sample yang emang separah ini.

14. **Diagnosis real-sample 2026-09-25 (bukan bug, konfirmasi sistem bekerja
    sesuai desain)** — user lapor 2 hasil "trouble" dari `index.html`, di-trace
    langsung pakai foto asli (bukan screenshot) lewat `/api/analyze` yang
    sama persis dipakai frontend:
    - **Sharpened pass bisa menyisipkan digit palsu di AWAL kode** (sample
      Pond's UV Miracle tutup tube, field_type `tube_emboss_default`): raw
      pass baca `"310826QHB8"` (10 char, FORMAT_OK, confidence 98.5%),
      sharpened pass baca `"1310826QHB8"` (11 char, ada "1" ekstra nempel di
      depan — kemungkinan `preprocess_crimp_for_ocr()`'s sharpening
      menangkap noise/pantulan di tepi crop jadi karakter palsu). Karena
      `recognize_crimp_text_dual()` mendeteksi `raw_text != sharp_text`,
      hasilnya correctly di-cap `LOW_CONFIDENCE` (bukan `OK`) — never-silently-
      pick-a-side ini persis yang didesain di gotcha #3/prinsip
      never-auto-correct. Raw pass kemungkinan besar benar, tapi sistem
      sengaja gak percaya diri 100% dan minta konfirmasi manusia — ini
      contoh nyata cross-check-nya jalan seperti seharusnya, bukan bug.
    - **3 digit pertama emboss EXP bisa benar-benar tidak terbaca sama
      sekali di SEMUA pass** (sample Pond's UV Protect, "EXP 140627" di
      crimp + "MFD 140624 8 QFZ" di body, field_type `tube_exp_date`): trace
      piece-by-piece nunjukin "140" dari "140627" jadi teks acak beda-beda
      di tiap pass (`"MF00FZ"` di raw, `"TAI"` waktu zoom_retry) — cuma
      "627" yang konsisten kebaca bersih di semua percobaan (raw, sharpened,
      DAN zoom_retry yang khusus didesain buat kasus EXP+MFD berdempetan,
      lihat item 12). Kemungkinan penyebab: posisi 3 digit itu pas kena
      noise garis knurl/ridge tepi crimp yang berkilau (foil emas), sama
      seperti gotcha #3 tapi kali ini separah "digit-nya ilang", bukan cuma
      "ketambahan karakter". Karena gak ada satupun pass yang berhasil
      merakit 6 digit lengkap yang valid, `LABEL_NOT_FOUND`/`LOW_CONFIDENCE`
      adalah jawaban yang BENAR (bukan sistem gagal nyari, tapi digitnya
      literally gak ada yang bisa dibaca) — lihat update di gotcha #13 soal
      `DIGIT_FALLBACK_EXCLUDED_LABELS` yang sudah jalan tapi tetap gak cukup
      buat sample seburuk ini.

    **Actionable buat QC** (bukan sesuatu yang perlu di-patch di kode):
    foto ulang dengan sudut/pencahayaan yang mengurangi silau di foil emas
    dan posisi lebih dekat/stabil ke garis crimp, kalau field_type EXP/tube
    emboss hasilnya `LOW_CONFIDENCE` berulang di tube yang sama.
15. **`find_digit_fallback_match()` bisa salah anggap 6-digit PERTAMA dari
    kode `tube_emboss_default` (10 char) sebagai kandidat EXP (2026-09-25)** —
    guard isolated-digit-run (`(?<!\d)(\d{6})(?!\d)`) di gotcha #11 cuma
    melindungi dari digit yang nempel ke digit lain. Sample nyata (Pond's UV
    Miracle, tube yang CUMA punya kode gabungan `310826QHB8`, tidak ada baris
    EXP terpisah sama sekali) membuktikan guard itu tidak cukup: setelah 6
    digit tanggalnya langsung diikuti HURUF ("Q"), bukan digit, jadi lookahead
    `(?!\d)` lolos trivial dan `"310826"` ke-anggap kandidat EXP yang valid
    padahal itu tanggal manufaktur dari kode lain. Fix: `find_digit_fallback_match()`
    sekarang terima `other_format_configs` (semua field_type LAIN dari
    `emboss_format_patterns.json`, bukan cuma field_type yang lagi dicari) —
    sebelum piece dijadikan kandidat, cek dulu apakah SELURUH teks piece itu
    sudah valid utuh sebagai format field_type lain (misal 10-char
    `tube_emboss_default`); kalau ya, skip piece itu total. Ini data-driven
    (baca config file), bukan hardcode label string kayak
    `DIGIT_FALLBACK_EXCLUDED_LABELS` — field_type baru yang ditambah ke
    config otomatis ikut terlindungi tanpa ubah kode. Diteruskan lewat
    `analyze_tube_emboss()` → `analyze_label_anchor_field()` →
    `recognize_label_anchor_dual()` → `find_digit_fallback_match()`.

    **Regression-tested** ke 35 foto di `images/` (32 lama + 3 baru,
    `tools/diagnose_exp_date_folder.py`): diff SEBELUM/SESUDAH fix cuma di 4
    baris yang jadi target (semua `"310826"` false-positive berubah jadi
    `LABEL_NOT_FOUND` yang benar) — 31 foto lainnya identik persis (termasuk
    4 match `OK` via `label_anchor`, `zoom_retry` case Pond's/Elsheskin, dan
    digit_fallback yang legit seperti `051228`/`150728`/`200628`/`120927`).
    Tidak ada yang rusak.
16. **Endpoint `/api/analyze` (dan `/api/ocr`, `/api/tube-emboss/analyze`)
    `async def`, tapi manggil pipeline OCR/YOLO-nya (`run_generic_pipeline`,
    `run_tube_emboss_pipeline`) secara LANGSUNG dan synchronous — TIDAK ADA
    `run_in_threadpool`/`asyncio.to_thread`/`run_in_executor` di manapun di
    codebase (2026-09-28, dicek lewat grep).** `ops/vision-service.service`
    juga jalanin uvicorn tanpa `--workers` (1 proses). Konsekuensinya:
    kalau 2-3 request `/api/analyze` datang bersamaan, request kedua/ketiga
    BENAR-BENAR ANTRE — diproses satu-satu, bukan paralel, karena
    inference yang blocking itu nahan satu-satunya event loop thread yang
    ada. Dikonfirmasi empiris lewat `htop` di VPS: yang kelihatan kayak
    "banyak proses" itu sebenarnya 1 proses uvicorn (1 PID) + thread-thread
    OS internal BLAS/OpenMP milik PaddlePaddle/OpenCV yang mem-paralelkan
    komputasi 1 request tunggal ke semua vCPU, BUKAN beberapa request
    beda yang diproses bersamaan. Ini bukan bug — buat prototype/pilot
    testing gak masalah — tapi jadi bottleneck nyata kalau production
    nanti dihit banyak stasiun sekaligus. Belum diputuskan/dikerjakan
    solusinya (opsi: multi-worker uvicorn [mahal RAM, model ke-load
    berkali-kali], thread-pool offload [event loop bebas tapi tetap
    CPU-core-bound], atau queue terpisah) — murni observasi, nunggu
    keputusan eksplisit user kalau/kapan concurrency jadi kebutuhan nyata.
17. **2 foto real-sample "gagal" yang dilaporkan user (2026-09-28) dikonfirmasi
    BUKAN bug, dua penyebab fisik berbeda — field_type `tube_exp_date`:**
    - **Klip/penjepit fisik bisa menutupi digit emboss-nya sendiri** (tube
      "POND'S SKIN INSTITUTE"): dari foto, klip hitam yang dipakai buat
      menjepit crimp saat pemotretan menutupi tepat satu digit di tengah kode
      ("EXP 3⬛0229") — bukan soal OCR gagal baca, framenya sendiri memang
      gak punya info digit itu. `LABEL_NOT_FOUND` di sini benar; gak ada
      perbaikan kode yang bisa menolong kasus ini, cuma foto ulang dengan
      klip dipindah dari area emboss.
    - **Sample kedua adalah sample yang SAMA PERSIS dengan gotcha #14**
      (tube "POND'S UV PROTECT SUN SERUM", "EXP 140627" + "MFD 140624 8 QFZ"
      di foto yang sama) — re-test menghasilkan `LABEL_NOT_FOUND` yang
      identik, konsisten dengan root cause yang sudah didiagnosis (3 digit
      pertama EXP gak kebaca di raw/sharpened/zoom_retry akibat
      glare/knurl di foil emas). Bukan regresi baru.

    Actionable buat QC (sama seperti gotcha #14): pastikan klip/penjepit
    yang dipakai buat motret gak menutupi garis emboss-nya, dan untuk tube
    foil emas coba sudut/pencahayaan yang mengurangi silau di garis crimp.
18. **Field_type `tube_mfd_date` (2026-09-28) — baca MFD yang di-PRINT di body
    tube** (misal "MFD 140624 8 QFZ" di Pond's UV Protect), sebagai pendamping
    `tube_exp_date` kalau EXP di crimp gak kebaca (glare/klip, gotcha #14/#17).
    Cuma entry config (`label_anchor: "MFD"`), reuse jalur label_anchor +
    zoom_retry yang sudah ada. **Vision Service (API) TIDAK menghitung EXP =
    MFD + shelf-life** — shelf-life aturan bisnis per-SKU, aturan final-nya
    tetap buat Laravel. Sebagai alat testing sementara, `index.html` punya
    kalkulator client-side: flag config `exp_from_shelf_life: true`
    (diteruskan `/api/field-types`, jadi JS tetap gak hardcode field key)
    memunculkan input "Shelf-life (bulan)" yang diisi QC per SKU, lalu
    `computed_exp` = MFD hasil OCR + N bulan ditampilkan di tabel hasil
    (dilabeli "dihitung di browser - bukan hasil OCR"). Aturannya: cuma
    dihitung kalau `format_valid` true; kalau tanggal-nya gak ada di bulan
    tujuan (misal 31 + 1 bulan → Februari) di-flag "cek manual", GAK
    di-roll-over/clamp (cara SKU menangani itu aturan bisnis); kalau status
    MFD bukan `OK`, ada peringatan hasil hitungan ikut perlu review.
    Confirmed end-to-end 2026-09-28: sample
    `ponds_uvprotect_exp140627_mfd140624_closeup.jpg` → MFD `140624` + 36
    bulan = `140627`, sama dengan EXP asli di crimp (yang `tube_exp_date`
    gagal baca, gotcha #14). Dua jebakan yang
    ketemu dari regression run pertama ke 36 foto `images/` (7 foto
    mengembalikan tanggal EXP sebagai MFD):
    - `DIGIT_FALLBACK_EXCLUDED_LABELS` cuma hardcode `("MFD",)` → saat cari
      MFD, piece "EXP 210428" gak di-skip. Fix: `find_digit_fallback_match()`
      sekarang terima `label` dan skip list-nya = hardcoded + `label_anchor`
      milik semua field_type LAIN di config, dikurangi label yang sedang
      dicari (supaya field gak menolak labelnya sendiri).
    - Digit tanpa label sama sekali ("AJA120927 PU") secara fisik gak bisa
      dibedakan EXP atau MFD → flag config baru `"digit_fallback": false`
      (default `true`) mematikan digit_fallback per field_type;
      `tube_mfd_date` pakai `false`, jadi tanpa literal "MFD" hasilnya
      `LABEL_NOT_FOUND`, bukan nebak.

    **Regression setelah fix** (36 foto): `tube_exp_date` identik dengan run
    sebelum fix; `tube_mfd_date` cuma 2 match, dua-duanya benar (`140624`
    di `Media (5).jpg` dan `ponds_uvprotect_exp140627_mfd140624_closeup.jpg`),
    tapi dua-duanya `LOW_CONFIDENCE` (confidence di-cap 0.84 — raw vs
    sharpened pass gak sepakat), belum pernah `OK`. Label cetak "MFG" (bukan
    "MFD", misal sample Elsheskin 11.31.20) TIDAK ke-match — kalau butuh,
    tambah entry/label baru, jangan longgarkan regex-nya. `pilot_ready`
    di-set `true` (default) atas permintaan eksplisit user 2026-09-28,
    walau baru terbukti di 2 foto real — aman dari sisi false PASS karena
    tanpa digit_fallback field ini cuma bisa match dari literal "MFD", dan
    hasilnya sejauh ini selalu `LOW_CONFIDENCE` (masuk review manusia).

19. **Startup AI Cross-Check — tahap 1 `label_bulk` (2026-11-11, belum di-wire ke `/api/analyze`).**
    Rencana lengkap: `docs/startup-ai-crosscheck-plan.md` (vision-service
    mengerjakan tahap 0,1,2,6; ipc_app tahap 3,4,5). Keputusan & temuan:
    - **LLM = Ollama di server LLM kantor (`100.100.160.26`, GPU), model
      `qwen3.8:27b`** (bukan `cosmax-image-prompt`, bukan `gpt-oss:20b`).
      Ollama hanya listen `127.0.0.1:11434` di server itu → dari laptop lewat
      terowongan `ssh -N -o ServerAliveInterval=30 -L 11434:127.0.0.1:11434
      devopscosmax@100.100.160.26` (sempat putus sendiri: `Connection reset`;
      tidak auto-reconnect). **User tegas: semua inferensi berat HARUS di
      server Ollama (GPU); jangan jalankan OCR/PaddleOCR di laptop atau
      server aplikasi (tanpa GPU, 11-29 dtk/foto).** Tidak perlu training
      model baru. Per request wajib override default model: `temperature 0`,
      `seed 0`, `think:false`, `format` = JSON schema (sudah di kode).
    - **Prompt saja TIDAK cukup mencegah halusinasi.** Di foto produksi
      (480x640, glare/terpotong; mis. `date_bulk/1033.jpg`) model MENEBAK
      karakter yang tertutup dan tetap `legible=true`; memperkuat prompt
      ("tertutup = null") malah bikin tebakan lolos cek format. Temuan
      kunci: tebakan itu BERUBAH-ubah menurut ukuran gambar input, sedangkan
      bacaan benar stabil di semua ukuran (kecuali 480 yang terlalu kecil).
      Mitigasi yang dipakai: **konsensus multi-ukuran** (`LABEL_VIEW_SIDES`,
      default `1024,640,1600,2048`, up- dan down-scale) — nilai `OK` hanya
      kalau SEMUA view identik + format valid + legible. Respons memuat
      `view_agreement`, `views_read`. Confidence BUKAN probabilitas
      terkalibrasi (turunan legible+format+agreement).
    - **Kontrak** `label_bulk`: `no_batch_raw`, `mixing_date_raw`,
      `kode_bulk`, `legible{}`, `format_ok{}`, `view_agreement{}`,
      `confidence`, `status` (OK/LOW_CONFIDENCE/ERROR). Nilai persis seperti
      terbaca, tidak pernah strip/koreksi; format cek = flag saja. LLM mati =
      `ERROR`, bukan PASS. Satu retry bila JSON rusak. Label Bulk: abaikan
      "Exp. Date" tulisan tangan & stiker timestamp putih.
    - **Hasil ukur (14 foto, kecil!)**: 3 sampel `images/sources/` + 11
      produksi `date_bulk`. 8 foto `OK` semuanya terverifikasi benar manual;
      0 nilai salah lolos gerbang. Foto glare berat → `LOW_CONFIDENCE`; 4
      foto lain juga `LOW_CONFIDENCE` karena view tak sepakat (belum dicek
      apakah memang tak terbaca atau terlalu ketat). Kecepatan ~3,5 dtk
      (foto glare) – ~7,5 dtk (foto konsisten) per foto, jauh di bawah
      asumsi 25-30 dtk. Belum cukup data utk klaim "nol false PASS".
    - **Update 2026-11-11 (sesi 2)**: `label_bulk` + `im_document` SUDAH
      terdaftar di `main.py` (`LLM_FIELD_TYPES`, pipeline `"llm"`, dijalankan
      lewat `run_in_threadpool`; jalur OCR lama tetap sync, gotcha #16).
      `index.html` dilayani dari server (`GET /`) → buka `http://localhost:8000/`,
      jangan `file://`. **Ukur label_bulk, 35 foto produksi** (kunci:
      `testdata_private/label_bulk_truth_v2.json`, dibaca Claude, 2 item
      `_verify` perlu cek manusia): 17 OK / 18 LOW; **0 false OK** (11 OK baru
      + sebelumnya semuanya cocok foto); mixing_date 100% benar, no_batch 86%
      (5 salah semuanya ter-flag), kode_bulk 94%; ~7,5 dtk/foto. Penyebab LOW:
      umumnya HANYA view 640 px yang menyimpang (3 view lain sepakat) → what-if
      tanpa view 640: 25/35 OK, tetap 0 false OK (SIMULASI dari `views_read`,
      belum jadi default; n kecil). 4 foto LOW lama (2841/4257/4832/7302):
      bukan gerbang terlalu ketat — ada view yang benar-benar salah baca
      (mis. 16/10 dibaca 18/10); 7302 memang buram. **`im_document`** (10 foto
      IM): 3 OK (1810, 5081, 7265, semua cocok foto), 7 LOW wajar (bukan IM /
      judul saja / terpotong / buram). **Diperluas ke 29 foto IM** (19 acak
      baru): total 6 OK (+3409, 4351, 683, semua cocok foto), 23 LOW, 0 false
      OK; ~6-9 dtk/foto. Laju OK rendah (±21%) karena foto 480x640 membuat
      teks umur simpan sangat kecil/buram → hambatan utamanya KUALITAS
      FOTO (keputusan terbuka rencana #4: panduan zoom ke bagian umur
      simpan), bukan gerbangnya. Kunci jawaban + skrip: `testdata_private/im_truth.json`
      (8 foto, bias: 6 dipilih karena sudah OK) + `tools/evaluate_im_document.py`.
      **Keputusan user 2026-11-11:** konsensus umur simpan dinilai dari NILAI
      (tahun/bulan) saja, bukan label/kelengkapan Botol-Unit Box: dipercaya bila
      semua view memberi himpunan nilai identik dan TEPAT SATU nilai; respons
      menambah `shelf_life_value` ({years, months} apa adanya). Re-run 35 foto
      IM: 7 OK, 0 false OK (tidak mengubah laju OK; penghambat sebenarnya
      adalah 1 view yang mengembalikan list kosong). **DITERAPKAN (user setuju,
      2026-11-11):** view yang mengembalikan list kosong = abstain (bukan tidak
      sepakat); syarat >=3 view membaca, maks 1 abstain, semua yang membaca legible,
      gabungan nilai TEPAT SATU; view yang membaca nilai BERBEDA tetap membatalkan.
      Re-run 35 foto IM: 10 OK (+2109, 4360, 6569, semua cocok foto), 0 false OK.
      6554 tidak ikut naik (im_number antar-view tak sepakat). Skema memakai `years`/`months`
      nullable (bukan `value`+`unit`) karena ada "1 Tahun 6 bulan"; tidak ada
      konversi ke bulan di sini. Umur simpan TIDAK selalu ada di foto IM.
    - **Data**: `filling/date_bulk` di `ipc_attachments` (DB `ipc_system`,
      SELECT saja!) memang Label Bulk (449 foto); `startup/im_number` (553).
      Foto IM contoh (`im_number/2439.jpg`) hanya memperlihatkan judul IM
      tanpa pernyataan umur simpan → RISIKO tahap 2, cek lebih banyak sampel.
    - **Update 2026-11-11 (sesi 3) — halaman uji gabungan + tube batch:**
      `GET /startup-check` (`startup_check.html`): 4 slot foto (Label Bulk,
      IM, tube EXP, tube batch) + input manual EXP/batch untuk non-tube;
      EXP diharapkan = mixing date + umur simpan (hari sama + N bulan, hari
      yang tak ada → REVIEW, tanpa roll-over) dan batch diharapkan = tepat 3
      huruf setelah "/" pada No Batch, KEDUANYA dihitung di browser (alat
      testing, bukan keputusan; PASS/FAIL tetap di ipc_app). Dropdown
      `index.html` disederhanakan (label pendek, `generic` = "Umum").
      Tube bolak-balik: depan = kode 10-char (tanggal+batch+mfg), belakang =
      EXP; non-tube memakai foto coding (belum ada field_type).
      `tube_emboss_pipeline.py` menambah `batch_code` (potongan blok `batch_*`,
      hanya bila `format_valid`) dan `batch_code_letters_ok` (flag: tepat 3
      huruf; aturan user — blok validasi tetap alfanumerik, gotcha #8).
      Salah baca Q→0 / B→8 pada emboss tertangkap flag huruf ini, tidak
      pernah dikoreksi otomatis.
    - **Tahap 6 `tube_batch_code` DIKERJAKAN (2026-11-11):** entry baru di
      `emboss_format_patterns.json`, TANPA `label_anchor` (jalur default
      crimp-crop), 3 blok `batch_1..3` charset `[A-Za-z]` → untuk foto yang
      barisnya HANYA batch (mis. `images/Media 9.jpg` = "CDA", conf 0.98).
      `startup_check.html` slot batch mencoba `tube_emboss_default` dulu; bila
      `format_valid` bukan true, coba `tube_batch_code` dan pakai hanya bila
      valid. Regression 38 foto `images/`: hanya Media 9 lolos; foto kode
      penuh/EXP tetap FORMAT_MISMATCH (tak ada batch palsu). 5 foto WhatsApp
      → `ERROR` (YOLO tak deteksi tube), belum diselidiki.
    - **Keputusan terbuka:** buang view 640 px untuk label_bulk? (what-if 25/35
      OK, 0 false OK, belum diterapkan); user perlu cek 2 item `_verify`
      (3435, 5844) di `label_bulk_truth_v2.json`. Server harus di-restart
      setelah ubah kode Python (`llm_extract.py`/`main.py`); HTML/JSON config
      tidak perlu.
    - `tools/eval_data/` (belum di-gitignore) berisi kunci jawaban turunan foto
      produksi — periksa sebelum commit apa pun.
    - `models/vision-ai-ocr.ipynb` (notebook Kaggle latih YOLO tube) berisi
      API key Roboflow plaintext, belum pernah di-commit → jangan di-commit
      sebelum key dihapus/di-rotate.

20. **`coding_print` — coding NON-TUBE (batch + EXP tercetak di kotak/jar/tutup/
    botol/label), 2026-11-11.** LLM Ollama `qwen3.8:27b`, tanpa training.
    - **Sumber foto uji**: DB `ipc_system` (MySQL Laragon lokal, root tanpa
      password, SELECT saja) tabel `ipc_attachments`, `field_label` IN
      (`coding`, `coding_primary`, `coding_secondary`, `primary_coding_batch_exp`,
      `secondary_coding_batch_exp`); file di `C:\Users\cosmax\Downloads\ipc-attachments\{batch}\{stage}\`.
      Nama file di-hash, jenis foto HANYA dari DB. `startup_checks.coding_format_*`
      = format coding yang diketik operator (contoh "Batch / EXP DDMMYY XX",
      XX = inisial LL & IPC). Folder "coding" banyak berisi tube tampak depan,
      formulir tulisan tangan, frame hitam → wajar jadi LOW.
    - **Format yang ditemui**: `OKB EXP 231029 CW`; 2 baris (`AKB` / `EXP 191029 QW`);
      `BATCH : BKB` / `EXP DATE : 301029`; `BN :BKB / MFG :10/2026 / EXP :04/2029`
      (jebakan MFG); Lip Coat `AKB EXP 10 2029 GK` (bulan-tahun). Kontrak:
      `batch_raw`, `exp_raw` (apa adanya), `exp_suffix`, `exp_format`
      (`DDMMYY`/`MM_YYYY`/`UNKNOWN`), `exp_parts` (potongan string, TANPA
      konversi), `packaging_type` (flag saja, lihat bawah), `read_via`, `rescue`.
      `MM_YYYY` → ipc_app membandingkan bulan+tahun saja.
    - **Alur**: (1) konsensus 4 ukuran seluruh foto; (2) HANYA bila (1) tidak OK:
      Qwen menunjuk kotak coding → crop + baca ulang konsensus 4 ukuran; hasil
      crop dipakai hanya bila nilai yang sama persis sudah muncul utuh di salah
      satu view tahap (1). OK tahap (1) tidak pernah diubah.
    - **Kenapa konsensus SEMUA view, bukan mayoritas**: foto 3044 → 3 dari 4 view
      membaca 151029, yang benar 151028.
    - **`packaging_type` (tube/non_tube/unknown) hanya FLAG, tidak memilih
      jalur** (keputusan user 2026-11-11, prinsip field_type dari luar tetap).
      YOLO tube detector TIDAK bisa membedakan tube vs non-tube (cuma 1 kelas;
      jar putih dapat 0.93-0.95). Qwen: 56 benar, 10 unknown, 0 salah (76 foto).
      User sempat mengira Ollama "punya memori/belajar": TIDAK — tiap request
      dari nol; "mengajari" = ubah prompt lalu ukur ulang.
    - **Hasil ukur**: set 1 (76 foto, kunci bacaan foto) 44 OK semua benar,
      0 false OK, 18 MM_YYYY OK. Set 2 (200 foto baru, kunci DB) 100 OK: 90 cocok
      DB, 8 beda DB tapi DICEK MATA coding memang beda/DB tanpa huruf (coding ≠
      DB nyata: 6299/6300 291029 vs DB 291129, 418 300928 vs 300929, 3279, 5393,
      4850 CKB vs DB CJB, 3585 LKB vs AKB), 2 TIDAK BISA DIPASTIKAN: 2468
      (label thermal buram, baca 181029, DB 151029) dan 847 (label jelas
      081029, DB 061029, kemungkinan label memang salah cetak). ~7 dtk/foto,
      ~14 dtk bila rescue.
    - **Batas yang belum terpecahkan**: konsensus hanya menangkap tebakan yang
      BERUBAH antar ukuran; salah baca konsisten (5/6→8 di label buram) lolos.
      Dicoba & DIBATALKAN: (a) verifikasi "karakter mana yang meragukan" ke Qwen
      → dua kasus tetap lolos; (b) gerbang tinggi teks px → tidak memisahkan
      (758 lebih kecil tapi benar). Akar masalah: 139/144 foto 480x640 (ipc_app
      mengecilkan saat unggah) → rekomendasi: unggah resolusi lebih tinggi.
      Catatan KPI: salah baca jadi "beda dari diharapkan" = false FAIL/REVIEW;
      false PASS hanya bila salah baca kebetulan sama persis dengan nilai yang
      diharapkan.

21. **Slot "Primary Coding" tunggal di `/startup-check` (2026-11-12) — meniru
    form ipc startup yang cuma punya satu upload primary coding.** Slot 3/4/5
    lama (tube EXP / tube batch / non-tube) digabung jadi 1 slot (+ foto ke-2
    opsional khusus tube). Alur (`analyzePrimary()` di `startup_check.html`):
    (1) `coding_print` dulu -> `packaging_type`; (2) `tube` -> foto yang sama
    juga lewat `tube_exp_date` + `tube_emboss_default` (fallback
    `tube_batch_code` bila format kode penuh tak valid); `non_tube` -> pakai
    hasil `coding_print`; `unknown` -> tampil tapi REVIEW, tidak menebak.
    Foto ke-2 (`mergePrimary()`): hanya digabung bila KEDUA foto terbaca
    `tube`; per field dipilih hasil terbaik (OK+format valid > format valid >
    foto 1), nilai tidak pernah digabung per karakter; jenis beda -> `unknown`.
    - **Ini MENGUBAH keputusan lama gotcha #20** ("`packaging_type` hanya flag,
      tidak memilih jalur"): di halaman uji ini flag itu sekarang memilih jalur,
      atas permintaan eksplisit user. field_type API tetap dari luar; ini hanya
      alur halaman uji. Kalau diintegrasikan ke ipc_app, putuskan ulang.
    - **`packaging_type` diputuskan >= 3 dari 4 view sepakat** (`_majority_kind()`
      di `llm_extract.py`), bukan semua view. Bug nyata: `images/Media (2).jpg`
      (tube jelas) -> 3 view `tube` + 1 view (kecil) `non_tube` -> aturan lama
      "semua sama" memberi `unknown`. Nilai batch/EXP TETAP konsensus semua view.
      Hanya diukur di beberapa foto, belum di 76 foto set `coding_print`.
    - **Jebakan operasional:** setelah ubah `llm_extract.py` server WAJIB
      di-restart (instance lama diam-diam tetap jalan dan bikin `[Errno 10048]`
      saat start ulang); HTML di-cache browser -> hard refresh (Ctrl+Shift+R).
    - **Akurasi tube MASIH KURANG** (tes 2026-11-12, `Media 10.jpg` x2): EXP
      `LOW_CONFIDENCE`/`format_valid false` (tidak terbaca), kode depan
      `310826QHB8` `LOW_CONFIDENCE`, batch `QHB` vs diharapkan `BJB` ->
      "beda & sumber belum yakin -> REVIEW" (arah aman, bukan false PASS).
      Sesi baru direncanakan khusus membahas peningkatan akurasi tube.

22. **`tube_exp_llm` — EXP crimp tube via YOLO crop + Ollama, cascade 3-dari-4
    (2026-11-13).** Hasil sesi "akurasi tube". Field_type baru di `LLM_FIELD_TYPES`
    (`main.py`, jalur threadpool, tanpa PaddleOCR). **Pembagian tugas:** Python =
    YOLO lokalisasi + crop (`tube_emboss_pipeline.tube_exp_crops()`, dengan
    `_YOLO_LOCK` karena YOLO tidak aman dipanggil paralel dari threadpool), validasi
    DDMMYY, aturan sepakat; Ollama = HANYA transkripsi tiap crop (prompt `coding_print`
    yang sama). Fungsinya `llm_extract.analyze_tube_exp()`. Tidak ada tube terdeteksi ->
    `LOW_CONFIDENCE` tanpa memanggil LLM. Tidak pernah strip/koreksi nilai.
    - **Aturan:** 4 sumber = crop tube utuh, 50% atas, 30% atas, foto utuh. Satu
      sumber `trusted` bila semua view (4 ukuran) membaca nilai sama + legible +
      DDMMYY (early-exit: berhenti di view pertama yang tak sepakat, hasil identik).
      `OK` hanya bila >=3 sumber trusted bernilai SAMA dan tak ada sumber trusted lain
      bernilai beda. LOW berhenti setelah 2 sumber gagal.
    - **Kenapa cascade, bukan satu crop:** garis hitam (sensor mark) menimpa SATU digit
      crimp. Crop lebar membuat model MENEBAK digit di bawah garis secara konsisten di
      semua view, jadi konsensus view TIDAK menangkapnya. Dua false OK sejati ditemukan
      (dicek mata): 2267 (`1?1029`, model 101029, DB 161029; hanya crop 50%) dan 6993
      (`2?1031`, model 201031, DB 291031; hanya tube utuh). Keduanya muncul di SATU skala
      saja, sedangkan semua bacaan yang sama dari >=3 sumber cocok DB atau benar saat
      dicek mata. 2709 (`14042?`, model 140426, DB 140428) ambigu/kemungkinan salah,
      ditolak aturan ini (hanya 2 sumber).
    - **Hasil (fungsi final, `tools/evaluate_tube_exp_llm.py`):** 175 foto set lama:
      11 OK (8 cocok DB, 3 beda DB tapi cetakan memang beda: 4277 cetak 210929 vs DB
      211029; 6962/6986 cetak `301029` vs DB 301129), 0 false OK terlihat; baseline foto
      utuh 5 OK. **Hold-out 19 foto baru (1600x900, 18/19 bergaris): 0 OK**, 0 false OK
      -> aturan 3-dari-4 ditemukan SETELAH melihat set lama, BELUM tervalidasi (sampel
      kecil), jangan dianggap "optimal". Waktu: median 5,1 dtk, rata-rata 10,5 dtk
      (OK 4 sumber x 4 view bisa 25-80 dtk), kasus LOW median 5 dtk. Satu crop dengan
      early-exit 2,2 dtk/foto tapi tidak aman (lolos tebakan di bawah garis).
    - **Akar masalah = garis hitam, bukan resolusi:** foto baru 1600x900 (ipc_app naik
      resolusi) tetap 0 OK. Tuning software tak akan memperbaikinya; yang membantu foto
      ulang dengan sudut yang tak menimpa garis. OK dari jalur ini BUKAN bukti kuat
      (satu posisi di bawah garis bisa tak terverifikasi) - kebijakan toleransi/REVIEW
      tetap di ipc_app (VisionEvaluator). PaddleOCR (`tube_exp_date`) di set ini BELUM
      diukur (aturan user: tak jalan di laptop, harus di server).
    - **`num_ctx` DIHAPUS dari request Ollama (2026-11-13, permintaan atasan user).**
      Server .26: `OLLAMA_CONTEXT_LENGTH=32768`, `MAX_LOADED_MODELS=2`, `KEEP_ALIVE=10m`,
      flash attention + KV cache q8_0. Sebelumnya dipaksa 8192. Terukur: view terbesar
      (2048 px) cuma 3847 token; regresi 176 foto IDENTIK (13 OK sama, 0 bacaan berubah,
      waktu sama) -> bukan peningkatan akurasi/kecepatan, hanya menyelaraskan konteks
      dengan klien lain (hindari reload model). `OLLAMA_NUM_CTX` env masih bisa memaksa.
    - **Data/alat eksperimen (semua di `testdata_private/tube/`, gitignored):**
      `classify.json` (set lama, 175 foto tube berkode crimp), `classify_v2.json`
      (+64 foto coding baru setelah DB disinkron prod 2026-11-13 -> 19 tube berkode
      crimp = hold-out), `tune_*.json`, `final_*.json`. Skrip: `tools/tune_tube_exp.py`
      (sapuan crop/prompt/early-exit/paralel + waktu), `tools/experiment_tube_crop.py`
      (eksperimen awal), `tools/evaluate_tube_exp_llm.py` (fungsi final).
      `tools/classify_tube_candidates.py` kini menulis `classify_v2.json` (classify.json
      hanya dibaca) dan mengulang entri `error`. Path foto di classify.json menunjuk
      `Downloads\ipc-attachments` yang sempat hilang; salinan di `photos_candidates/`.
    - **Jebakan:** terowongan SSH ke Ollama putus sendiri di tengah run panjang (tak
      auto-reconnect) -> entri ERROR; skrip eksperimen kini tidak men-cache ERROR.
      Prompt khusus crimp (tanpa "legible" longgar) terlalu ketat: hanya 5 OK vs 13.
      Belum diuji: view paralel, kurangi view per sumber (mis. 2) untuk kecepatan.

23. **Kecepatan `tube_exp_llm` + ukur PaddleOCR `tube_exp_date` di set tube (2026-11-13).**
    Set lama = 176 foto tube berkode crimp (`classify.json`), hold-out = 19 foto (`classify_v2.json`
    minus set lama). Kunci = DB; OK beda DB dicek visual (crop diperbesar). Tidak ada default produksi
    yang berubah. Kode: env baru `TUBE_EXP_VIEW_SIDES` (fallback `CODING_VIEW_SIDES`) dan
    `TUBE_EXP_PARALLEL` (default 1) di `llm_extract.py`; `tools/evaluate_tube_exp_llm.py --tag`;
    `tools/evaluate_tube_exp_paddle.py` (baru, PaddleOCR di laptop atas izin khusus sesi ini).
    - **Tugas 1 (kecepatan), set lama, workers=1, server .26 sedang lengang (baseline dijalankan ulang
      supaya waktu sebanding; ukuran lama 10,5 dtk rata-rata dipengaruhi beban server):**

      | Varian | OK | cocok DB | false OK | median / mean / p90 dtk | waktu kasus OK |
      |---|---|---|---|---|---|
      | Baseline 4 view (1024,640,1600,2048) | 11 | 8 | 0 | 2,6 / 5,0 / 10,9 | ±26 |
      | 2 view (1024,1600) | 16 | 12 | **1** (3948: baca 130423, cetakan 130428, digit akhir mirip 8) | 2,6 / 4,3 / 10,8 | ±12 |
      | 4 sumber paralel (`TUBE_EXP_PARALLEL=4`) | 11 (identik) | 8 | 0 | 5,0 / 7,8 / 16,5 | ±25 |

      2 view DITOLAK: view ke-3/4 memang yang menangkap salah baca di 3948 (OK baru 4 benar + 1 salah);
      hemat waktu rata-rata cuma 14%. Paralel DITOLAK: waktu kasus OK tidak membaik (server Ollama
      tampak memproses serial) dan LOW malah 2x lebih lambat karena early-stop antar sumber hilang;
      status identik dengan sekuensial. Kasus LOW (94% foto) sudah cepat (median 2,6 dtk) - biaya ada di
      kasus OK (16 panggilan). Hold-out, konfigurasi default: 0 OK, 0 false OK, median 2,6 / mean 2,8 dtk.
    - **Varian 3 view (set lama, workers=1):** tanpa 640 (1024,1600,2048) = 14 OK, **false OK 3948 lagi**
      (view 640 px = penangkapnya, jangan dibuang); tanpa 1600 (1024,640,2048) = 12 OK (+6778 benar), 0
      false, waktu OK 19,6 dtk; tanpa 2048 (1024,640,1600) = 12 OK (+2308 benar), 0 false, waktu OK 17,3 dtk
      (baseline 24,3). Rata-rata keseluruhan tetap ±5 dtk (hanya ±6% foto kena kasus OK). `1024,640,1600`
      KANDIDAT (env `TUBE_EXP_VIEW_SIDES`), belum default: bukti keamanan hanya dari 1 foto penentu dan 1 OK
      baru, hold-out 0 OK tak informatif. Validasi di foto nyata ipc_app sebelum dijadikan default.
    - **Server Ollama (.26) = DOCKER** (`ollama/ollama:0.40.13`), bukan systemd. Env: KEEP_ALIVE=10m,
      FLASH_ATTENTION=1, KV_CACHE_TYPE=q8_0, CONTEXT_LENGTH=32768, MAX_LOADED_MODELS=2, HOST=0.0.0.0:11434;
      `OLLAMA_NUM_PARALLEL` TIDAK diset. Uji empiris lewat terowongan: 4 panggilan paralel 2,1 dtk = berurutan
      2,2 dtk -> serial. Mengubahnya = buat ulang container dengan `-e OLLAMA_NUM_PARALLEL=2` (pastikan volume
      model, awas memori GPU: konteks x paralel); keputusan atasan, server dipakai bersama. Belum dilakukan.
    - **Tugas 2 (PaddleOCR `tube_exp_date`, jalur label_anchor)**: set lama 15 OK (9 cocok DB, 6 beda DB)
      + hold-out 3 OK (semua cocok DB). Waktu median 5,8 / mean 7,0 / p90 12 dtk di laptop CPU (LLM
      default: 2,6 / 5,0 / 10,9 dtk; kasus OK LLM 26 dtk). **Cek visual 6 OK beda DB set lama:** 4277
      (210929; cetakan memang 210929, sama dgn LLM, bukan salah baca); **5 FALSE OK sejati**: 4673
      (baca 190028, cetakan 190428), 3980 (130423, cetakan 130428), 6993 (201031, DB 291031; garis hitam),
      7677 (051129, cetakan 091129; batang hitam menutup digit pertama), 1388 (131020 vs DB 131029; garis
      hitam). Confidence PaddleOCR pada false OK tinggi (0,91-0,98) -> confidence TIDAK menangkapnya.
      Perbandingan foto sama: Paddle OK 18 vs LLM OK 11; irisan 3; Paddle-saja 15 (10 cocok DB,
      5 false), LLM-saja 8 (semua benar). Jadi PaddleOCR menambah OK tapi TIDAK aman sebagai penentu
      sendiri (5 false OK / 176 vs LLM 0), dan di hold-out (3 benar dari 19) LLM 0 - sampel terlalu kecil
      untuk menyimpulkan.
    - **Jawaban singkat:** PaddleOCR bukan lebih akurat (lebih banyak OK tapi 5 false OK), sedikit lebih
      lambat di kasus LOW, jauh lebih cepat di kasus OK; belum aman jadi sumber ke-5 yang menaikkan OK.
      Satu aturan yang TERLIHAT aman tapi post-hoc & kecil: "LLM LOW dengan nilai non-null yang SAMA persis
      dgn OK Paddle" -> +2 OK di set lama (2144, 6425, keduanya benar, 0 false), +0 di hold-out. Kandidat
      untuk divalidasi di set lebih besar, BELUM diterapkan. Aturan "OK hanya bila LLM OK DAN Paddle sepakat"
      hanya menyisakan 3 OK (4277, 6518, 6567) - lebih aman tapi tidak menaikkan OK.
    - Foto kunci: garis/batang hitam masih penyebab utama (5 dari 6 OK beda DB Paddle bergaris atau
      digit ambigu). Tidak ada perbaikan software di sini; foto ulang tetap solusi.
    - Data hasil: `testdata_private/tube/final_old_{base,s2a,par4}.json`, `final_new_base.json`,
      `paddle_{old,new}.json` (gitignored).

## Infra: CI/CD pilot ke VPS (jangan re-discover ini lagi)

Ini jalur **terpisah** dari deploy manual on-prem `.24` di atas — dipakai
khusus buat latihan praktik DevOps ke sebuah **Proxmox VPS pilot**
(`vision-service` di VM 104, node `pve1` — 4 vCPU / 4GB RAM / 32GB disk,
Ubuntu 26.04, network `vmbr0` dengan Proxmox VM-level firewall AKTIF
`firewall=1`). **GitHub tetap source of truth**; GitLab cuma dipakai buat
CI/CD-nya. Detail lengkap ada di README.md section "Pilot via GitLab CI/CD
ke VPS" — di sini cuma gotcha yang jangan sampai di-re-discover.

1. **`torch`/`torchvision` di `requirements.txt` HARUS pakai
   `--extra-index-url https://download.pytorch.org/whl/cpu`** (baris
   pertama file). Konfirmasi: `torch.__version__` di venv Windows yang
   sudah tested = `"2.14.0+cpu"` (no CUDA runtime bundled), tapi `pip show`/
   `pip freeze` cuma nunjukin `"2.14.0"` polos — versi CPU vs CUDA-enabled
   sama-sama diberi nomor versi yang sama, dibedakan cuma dari index
   mana yang dipakai install. Tanpa extra-index-url ini, `pip install
   torch==2.14.0` di Linux x86_64 (VPS-nya) resolve ke build CUDA-enabled
   default, narik beberapa `nvidia-*-cu12` package (total bisa beberapa GB)
   yang percuma di VPS tanpa GPU dan bisa menghabiskan disk 32GB yang
   memang udah pas-pasan.
2. **VPS pakai Python 3.11 dari deadsnakes PPA, BUKAN python3 default
   Ubuntu 26.04** — `ops/provision_vps.sh` install ini eksplisit. Alasan:
   `venv/` yang sudah divalidasi di repo ini Python 3.11 (lihat item
   `venv/` di tabel atas); default python3 Ubuntu 26.04 kemungkinan lebih
   baru dan belum tentu ada wheel `paddleocr`/`paddlepaddle` buat versi itu.
3. **Branch `production` sengaja dipisah dari `main`/`v1.0.0`** — isinya
   sama (dibuat dari situ), tapi satu-satunya fungsinya jadi pemicu
   pipeline pilot ini. `.github/workflows/mirror-to-gitlab.yml` cuma
   push-mirror branch ini (bukan `--all`/tags) ke GitLab, dan
   `.gitlab-ci.yml` di-gate `CI_COMMIT_BRANCH == "production"` — push ke
   `main`/`v1.0.0` TIDAK PERNAH trigger deploy ke VPS. Kerja harian tetap
   di `main`/`v1.0.0`; begitu mau benar-benar di-pilot-kan, merge/push ke
   `production`.
4. **Push-mirror dari GitHub Actions, bukan fitur "pull mirror" bawaan
   GitLab** — GitLab punya fitur native buat pull-mirror + auto-trigger
   pipeline, tapi itu gated di sebagian tier berbayar. Solusinya:
   `.github/workflows/mirror-to-gitlab.yml` di sisi GitHub yang push
   manual ke GitLab pakai token (`GITLAB_PROJECT_URL` + `GITLAB_TOKEN`,
   secrets sudah di-set di GitHub repo settings) — begitu GitLab nerima
   push biasa, `.gitlab-ci.yml` standar yang trigger, gratis di tier
   manapun.
5. **GitLab Runner terdaftar LANGSUNG DI VPS itu sendiri** (shell
   executor, tag **`vps-aiocr`** — harus persis, itu yang dicari
   `.gitlab-ci.yml`), bukan runner terpisah yang SSH masuk. Konsekuensi:
   job CI jalan sebagai user `gitlab-runner` yang dibuat otomatis oleh
   paket resmi `gitlab-runner`.
6. **Sudoers cuma mengizinkan SATU script privileged**
   (`/usr/local/bin/vision-service-deploy.sh`, sumbernya `ops/deploy.sh`),
   bukan wildcard command apapun. Alasannya: kalau CI job (misal dari MR
   yang di-compromise) cuma bisa jalanin satu script yang isinya
   root-controlled, permukaan serangnya jauh lebih kecil dibanding kasih
   `gitlab-runner` sudo bebas ke `systemctl`/`rsync`/dll satu-satu.
7. **Deploy SELALU smoke-test (`tests/test_ocr.py`) di venv yang baru
   di-install DULU, baru restart `systemctl restart vision-service`** —
   kalau smoke test gagal, script exit non-zero SEBELUM service yang
   sedang jalan disentuh sama sekali. Deploy yang rusak gak pernah
   mematikan service yang masih OK, sejalan dengan prinsip "jangan
   nebak, flag ke manusia" yang sama dipakai di logic OCR-nya.
8. **Firewall itu 2 LAYER independen di VM ini, bukan cuma 1** — NIC VM
   104 dibuat dengan `firewall=1` (Proxmox punya firewall sendiri di
   level Datacenter/VM), terpisah total dari `ufw` di dalam Ubuntu-nya
   (`ops/configure_firewall.sh`). Kalau port kelihatan "diblok" padahal
   ufw udah di-allow, cek juga rule di Proxmox UI — salah satu layer
   block tetap bikin gagal walau layer lainnya udah benar. **UPDATE
   2026-09-28**: di provisioning nyata pertama, ternyata layer Proxmox
   TIDAK memblokir apa-apa (traffic tetap tembus walau ufw masih
   `inactive`) — jadi 2-layer ini benar ada, tapi jangan asumsikan
   Proxmox layer pasti restriktif; selalu test tembus/tidaknya di kedua
   layer secara terpisah, jangan cuma asumsi dari `firewall=1` doang.
9. **`git archive` dari Windows dengan `core.autocrlf=true` (default installer
   Git for Windows) diam-diam convert LF→CRLF di file hasil export** —
   walau blob git-nya sendiri LF bersih dan working tree checkout lokal
   juga LF. Ini bikin `ops/provision_vps.sh` gagal total di VPS pertama
   kali (`$'\r': command not found`, `set: pipefail: invalid option
   name`) padahal file di repo/working-tree kelihatan normal. Relevan
   kalau transfer repo ke VPS lewat `git archive`+`scp`/tar (bukan
   `git clone` native di VPS-nya) dari mesin Windows. Fix: `git -c
   core.autocrlf=false archive --format=tar <branch> -o out.tar` —
   jangan pakai `git archive` polos buat keperluan ini di Windows.
10. **`opencv-python` (paket non-headless yang dipinning di
    `requirements.txt`) butuh `libGL.so.1` saat import**, yang gak ada
    di Ubuntu server minimal (`ImportError: libGL.so.1: cannot open
    shared object file`). Ini OS-level dependency, BUKAN masalah versi
    Python package — jangan ganti `opencv-python` jadi
    `opencv-python-headless` di `requirements.txt` (itu mengubah paket
    yang sudah pinned/tested), cukup install lib sistemnya:
    `apt-get install -y --no-install-recommends libgl1 libglib2.0-0`.
    Belum ditambahkan ke `provision_vps.sh`'s apt package list — kalau
    VPS di-rebuild dari nol, gotcha ini bakal muncul lagi sampai
    ditambahkan ke situ.
11. **Default CPU type Proxmox ("kvm64"/qemu64 baseline) sama sekali
    gak punya AVX/AVX2** (`/proc/cpuinfo` flags mentok di `sse2`/`pni`)
    — wheel CPU PaddlePaddle (kemungkinan torch juga) diasumsikan
    minimal AVX buat kernel BLAS/oneDNN-nya, jadi begitu smoke test
    jalan, crash `Illegal instruction (core dumped)` — BUKAN error
    Python biasa, gampang ke-salah-diagnosis sebagai masalah dependency.
    Fix HARUS di Proxmox, bukan di dalam VM: UI → VM 104 → Hardware →
    Processor → **Type = "host"** (aman di setup single-node `pve1` ini,
    gak ada concern live-migration ke CPU fisik lain), lalu **Stop+Start
    penuh** (reboot dari dalam OS TIDAK cukup — model CPU QEMU cuma
    ke-apply ulang kalau prosesnya di-spawn dari awal). Confirmed:
    `/proc/cpuinfo` abis fix nunjukin CPU asli (`Intel Xeon Silver
    4410Y`) dengan `avx`/`avx2`/`sse4_2`.
12. **Traffic dari jaringan kantor ke subnet `100.100.160.x` (termasuk
    VPS pilot ini) lewat overlay/VPN privat, BUKAN IP publik internet
    biasa mesin kantor.** IP publik yang kelihatan dari layanan macam
    `ifconfig.me` (di kasus ini `36.64.64.202`) SAMA SEKALI BUKAN IP
    yang dilihat VPS — source IP asli yang sampai ke VPS itu
    `10.10.162.98` (kelihatan dari `/var/log/auth.log`: `Accepted
    publickey for ... from <ip>`, atau banner "Last login ... from"
    di SSH client). Kalau mau nulis firewall allowlist buat IP kantor,
    SELALU cek source IP asli lewat auth.log/`who`/`ss` di VPS-nya
    dulu, jangan percaya IP publik dari layanan "what's my ip" buat
    jaringan overlay kayak gini. Allowlist yang settled buat port 8000:
    `10.10.162.0/24` (VLAN kantor) + `100.100.160.0/24` (segmen server
    internal); SSH (22) tetap terbuka ke semua IP.
13. **`ops/configure_firewall.sh`'s loop penghapusan rule lama (buat
    idempotent re-run) ada bug** — kalau dijalankan ulang saat masih
    ada rule port 8000 lama, bisa gagal di tengah jalan dengan
    `ERROR: Invalid syntax` (nomor rule ufw geser tiap kali satu
    rule dihapus, tapi ekstraksi nomor lewat `grep`+`sed` di loop-nya
    gak selalu re-sync dengan benar). Workaround yang dipakai: hapus
    manual `sudo ufw --force delete <N>` per nomor rule (mulai dari
    nomor TERBESAR dulu supaya nomor lain gak ikut geser), baru
    tambahkan ulang rule allow/deny-nya. Belum di-patch di script-nya
    sendiri — kalau perlu re-run `configure_firewall.sh` lagi dengan
    allowlist baru, cek dulu apakah bug ini masih muncul.

**Status implementasi saat ini (2026-09-28, UPDATE akhir sesi)**: VPS pilot
(VM 104) **SUDAH fully provisioned dan live**. `provision_vps.sh` sudah
dijalankan sukses, GitLab Runner sudah diregister (tag `vps-aiocr`, shell
executor) dan status **Online** di GitLab UI. Sebuah pipeline yang sempat
pending ~52 menit (dari push `production` sebelumnya, nunggu runner) auto-
jalan begitu runner online — `deploy` job-nya sempat GAGAL di percobaan
pertama karena 2 gotcha baru (#10 libGL dan #11 CPU AVX di atas), setelah
di-fix manual di VPS lalu job di-retry dari GitLab UI, **end-to-end deploy
BERHASIL** (`validate`+`deploy` sukses, `vision-service` systemd `active
(running)`, `/api/health` respond `{"status":"ok"}`). Firewall ufw sudah
dikonfigurasi (lihat gotcha #12): port 8000 dibatasi ke `10.10.162.0/24` +
`100.100.160.0/24`, SSH terbuka ke semua. Testing manual via `index.html`
lokal (arahkan field "Backend URL" ke `http://100.100.160.31:8000`)
confirmed jalan. **Belum dikerjakan**: auth/HTTPS di API (memang belum
dalam scope pilot ini, lihat catatan "local prototype" di atas), dan
5 gotcha baru di atas (#9-13) belum di-patch ke script-nya sendiri
(`configure_firewall.sh` bug, `provision_vps.sh` belum nambah
`libgl1`/`libglib2.0-0` ke apt list).

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
      tetap). Confirmed fix di sample tube asli user.
- [x] `label_anchor` diperluas 3x lagi (2026-09-24) dari tes ke lebih banyak
      sample foto real (lihat item 11): digit-fallback buat tutup tube tanpa
      kata "EXP" sama sekali, dukungan format tanggal titik + tahun 4-digit
      buat label cetak, dan ambang kepercayaan deteksi tube yang lebih ketat
      khusus jalur ini biar gak salah crop ke background. Regression-tested
      ke 32 foto di `images/`, nggak ada yang rusak.
- [x] Known gap Pond's/Elsheskin (baris EXP+MFD berdempetan) RESOLVED
      (2026-09-24, lihat item 12): tambah `find_zoom_retry_match()` sebagai
      last-resort fallback (crop+upscale+re-OCR band kecil di sekitar piece
      label yang ketemu tapi tanpa digit) plus `group_zoomed_pieces_into_lines()`
      buat cluster hasil re-OCR itu per baris fisik. Clustering-nya sempat 2x
      salah desain (range-overlap chain-merge, lalu fixed-seed yang salah
      klaim piece digit ke baris MFD padahal harusnya baris EXP) sebelum
      settled ke sequential adjacent-gap grouping. Confirmed fix di sample
      asli, regression-tested ke 32 foto - 31 lainnya hasilnya identik,
      cuma Pond's yang berubah dari `LABEL_NOT_FOUND` jadi resolved
      (`FORMAT_OK`, `LOW_CONFIDENCE`, capped confidence 0.84 by design).
- [x] `find_digit_fallback_match()` vs `find_zoom_retry_match()` urutan
      dibalik (2026-09-24, lihat item 13) — user lapor sample tube Pond's UV
      Protect lain (EXP di crimp + MFD tercetak di body, satu foto) sempat
      confidently return tanggal MFD sebagai kalau itu EXP, karena
      digit_fallback (buta konteks, gak tahu ada literal "EXP") kebagian
      giliran duluan dan nemu 1 kandidat non-ambiguous sebelum zoom_retry
      (yang di-anchor ke piece "EXP" asli) sempat coba. Fix: zoom_retry
      sekarang dicoba duluan setiap kali ada piece yang literally mengandung
      label — digit_fallback tetap last-resort kalau zoom_retry juga gagal.
      Regression-tested ke file PNG asli sample yang dilaporkan: hasil
      berubah dari confidently-wrong (`140624`/MFD, `FORMAT_OK`) jadi
      `DIGIT_FALLBACK_AMBIGUOUS` (both `140627`/EXP dan `140624`/MFD ketemu,
      di-flag ambigu bukan ditebak) — sesuai prinsip minimalkan false PASS.
      Opsi lanjutan (exclude kandidat dari piece berlabel lain seperti "MFD")
      belum dikerjakan, nunggu keputusan eksplisit user.
- [x] Field-level pilot gating (2026-09-25) — `/api/field-types` sekarang
      return `pilot_ready` per field_type (`wo_number`/`batch_no` = `false`,
      di `field_patterns.json`; default `true` kalau field gak set flag ini).
      `index.html` filter dropdown-nya berdasarkan flag ini by default, ada
      checkbox "Show all field types (dev/testing only)" buat nampilin field
      yang belum matang saat testing lokal. Ini bagian dari persiapan pilot
      internal terbatas (deploy scope masih terbatas ke `tube_emboss_default`
      + `tube_exp_date` yang sudah di-regression-test, BUKAN full production
      architecture — item Laravel/React/kontrak final di bawah tetap belum
      dikerjakan).
- [x] Fix false-positive digit_fallback (2026-09-25, lihat gotcha #15) —
      `find_digit_fallback_match()` sekarang skip piece yang teksnya utuh
      sudah valid sebagai format field_type LAIN (`other_format_configs`),
      supaya kode 10-char `tube_emboss_default` gak ke-cannibalize jadi
      kandidat EXP cuma karena 6 digit pertamanya kebetulan valid tanggal.
      Regression-tested ke 35 foto (`tools/diagnose_exp_date_folder.py`,
      script baru) — cuma 4 false-positive yang berubah, 31 lainnya identik.
      3 foto real baru (`ponds_uvprotect_exp140627_mfd140624_closeup.jpg`,
      `ponds_uvprotect_exp140627_wideshot_lowres.jpg`,
      `ponds_uvmiracle_default_only_310826qhb8.jpg`) ditambahkan ke `images/`
      sebagai fixture tetap.
- [x] Diagnosis 2 laporan "trouble" dari real sample (2026-09-25, lihat
      gotcha #14) — dua-duanya dikonfirmasi BUKAN bug: sharpened-pass yang
      nyisipin digit palsu di depan kode (correctly di-cap LOW_CONFIDENCE
      karena disagreement raw-vs-sharpened) dan kasus 3-digit EXP yang
      literally gak kebaca di semua pass (correctly LABEL_NOT_FOUND karena
      gak ada kandidat lengkap). Juga ketemu: catatan "belum dikerjakan" di
      gotcha #13 soal exclude kandidat MFD dari digit_fallback itu STALE —
      fiturnya (`DIGIT_FALLBACK_EXCLUDED_LABELS`) ternyata sudah ada dari
      commit `92ac7bd`, cuma dokumentasinya belum di-sync.
- [x] Diagnosis 2 foto real-sample "gagal" lain dari user (2026-09-28, lihat
      gotcha #17) — keduanya dikonfirmasi BUKAN bug: satu klip fisik yang
      dipakai buat motret ternyata menutupi salah satu digit emboss-nya
      sendiri (LABEL_NOT_FOUND yang benar, gak ada info digit di framenya),
      satu lagi sample yang sama persis dengan kasus di gotcha #14 (3 digit
      EXP gak kebaca akibat glare foil), re-test hasilnya identik/konsisten.
- [x] Field_type `tube_mfd_date` (2026-09-28, lihat gotcha #18) — baca MFD
      print di body sebagai pendamping EXP crimp yang sering gak kebaca.
      Plus fix `digit_fallback` (exclusion label data-driven + flag
      `digit_fallback: false` per field). `pilot_ready: true`.
- [x] Kalkulator EXP = MFD + shelf-life (bulan) client-side di `index.html`
      (2026-09-28, gotcha #18) — alat testing, API Vision Service gak berubah
- [x] Bundle integrasi Laravel (`integrations/laravel-ipc/`, 2026-09-28) —
      `HttpVisionClient` (multipart ke `/api/analyze`), `VisionEvaluator`
      (PASS hanya kalau status OK + format valid + cocok `ipc_batches.exp_date`;
      apa pun lainnya REVIEW; FAIL hanya kalau confident tapi beda), EXP =
      MFD + `master_products.shelf_life_months`, tabel audit `ipc_logs`,
      `php artisan vision:ping`, hook React. Diverifikasi di SALINAN `ipc_app`
      (repo asli tidak disentuh): 255 test lulus (18 baru), Pint + tsc lulus,
      request asli ke VPS OK. Gotcha yang ketemu dari request asli: kalau
      label_anchor gagal, response TIDAK punya key `extracted_date_code` dan
      `raw_ocr_text` = semua teks di foto (460+ char) — jangan pakai
      `raw_ocr_text` sebagai nilai kode kecuali `format_valid` true. Gotcha
      verifikasi lokal: vendor `ipc_app` butuh PHP ≥ 8.4, tapi `pdo_sqlite` di
      PHP 8.5 Laragon diblokir Application Control policy → test dijalankan
      dengan MySQL instance sementara (datadir di scratchpad, port 3399).
- [x] Startup AI Cross-Check tahap 1 sebagian (2026-11-11, lihat gotcha #19):
      `llm_extract.py` + `tools/evaluate_label_bulk.py` + konsensus
      multi-ukuran, terukur di 14 foto.
- [x] Startup AI Cross-Check tahap 1, 2, 6 selesai di repo ini (2026-11-11,
      gotcha #19): `label_bulk` + `im_document` terdaftar di `main.py`
      (35 foto masing-masing, 0 false OK; label_bulk 17 OK, IM 10 OK),
      `tube_batch_code`, halaman `/startup-check`. Belum di-commit.
- [x] `coding_print` non-tube (2026-11-11, gotcha #20): `llm_extract.analyze_coding_print`,
      terdaftar di `main.py`, slot 5 di `/startup-check` (kolom manual tetap
      cadangan), `tools/evaluate_coding_print.py`. Belum di-commit.
- [x] Slot Primary Coding tunggal di `/startup-check` (2026-11-12, gotcha #21):
      tube/non-tube dibedakan otomatis lewat `packaging_type` (>= 3/4 view),
      tube boleh 2 foto. Belum di-commit.
- [x] `tube_exp_llm` (2026-11-13, gotcha #22): YOLO crop (Python) + transkripsi
      Ollama, cascade 3-dari-4 sumber; 11 OK / 175 foto (0 false OK terlihat),
      hold-out 19 foto 0 OK; `num_ctx` dihapus. Belum di-commit, belum di-deploy,
      belum dipakai `startup_check.html`. Aturan BELUM tervalidasi di set lebih besar.
- [x] Kecepatan `tube_exp_llm` + ukur PaddleOCR di set tube (2026-11-13, gotcha #23): 2 view dan paralel
      ditolak (false OK / tidak lebih cepat); PaddleOCR 18 OK tapi 5 false OK -> bukan penentu. Env
      `TUBE_EXP_VIEW_SIDES`/`TUBE_EXP_PARALLEL` ditambah (default tidak berubah). Belum di-commit.
- [x] Persiapan deploy ke server .151 (2026-11-13, BELUM di-commit/deploy/diuji build): `Dockerfile` (python:3.11-slim,
      libgl1, model PaddleOCR diunduh saat build), `.dockerignore`, `docker-compose.prod.yml` (port 8083, env file
      `VISION_ENV_FILE`, `mem_limit` 5g), `.gitlab-ci.yml` ditulis ulang (runner tag `server105`, rsync ke
      `VISION_DEPLOY_PATH`, build -> `up --wait` -> smoke curl; variabel CI: `VISION_DEPLOY_PATH`, `VISION_PORT`,
      `VISION_ENV_FILE`). `main.py`: env `LOAD_PADDLEOCR` (default 1; 0 = PaddleOCR tidak dimuat, jalur OCR -> 503),
      jalur OCR kini lewat threadpool + satu kunci `_OCR_LOCK` (antrean tetap serial, event loop bebas), env
      `VISION_API_KEY` opsional (header `X-API-Key` untuk /api/* kecuali health; halaman uji tidak dijaga). Jalur lama
      systemd/VM 104 (`ops/*.sh`, `vps-aiocr`) tidak dipakai lagi oleh CI. Prasyarat server: `/etc/vision-service/production.env`
      (`OLLAMA_URL=http://100.100.160.26:11434`), Ollama .26 terjangkau dari .151, runner `server105` aktif untuk project ini.
- [ ] **Akurasi tube masih rendah (±6%)** — penghambat = garis hitam menimpa satu digit
      crimp (gotcha #22), bukan resolusi. Opsi: panduan foto ulang, kebijakan REVIEW di
      ipc_app, ukur PaddleOCR (`tube_exp_date`) di server, uji kurangi view/paralel
      untuk kecepatan, validasi di set tube lebih besar. Kode depan 10 char (batch)
      belum disentuh.
- [ ] `coding_print` sisa: user cek item `_verify` di `coding_print_truth.json`
      + 2 foto tak pasti (2468, 847); resolusi unggah foto di ipc_app SUDAH
      dinaikkan (2026-11-11, ipc_app commit `99a0db3`: kamera ideal 1920x1080,
      simpan maks 1600 px, JPEG 0.85) — belum dicek di tablet nyata; daftarkan
      di `ipc_app/config/vision.php`
- [ ] **SAAT INTEGRASI ke ipc_app baca `docs/ai-learning-plan.md` bagian "PENGINGAT SAAT
      INTEGRASI"** (advisory saja, development dulu, vision:ping, catat selisih, ukur ulang).
- [x] Eval sampel acak vs DB (2026-11-12, `tools/evaluate_vs_db.py`, rencana
      "AI makin pintar" di `docs/ai-learning-plan.md`): label_bulk 63/150 OK,
      im_document 21/150 OK; view 640 px DIPUTUSKAN dipertahankan; 1 kemungkinan
      false OK (foto 1505). Foto produksi baru sudah disalin ke `testdata_private/`.
- [ ] Startup AI Cross-Check sisa: putuskan view 640 px, user cek 2 item
      `_verify`, selidiki 5 foto WhatsApp ERROR, kunci jawaban IM lebih
      banyak, daftar field_type baru di `ipc_app/config/vision.php`
      (tahap 3–5 di ipc_app)
- [ ] Integrasi bundle ke `ipc_app` sungguhan (copy + 3 edit manual +
      migrate + isi shelf-life per SKU + UI di halaman stage) — dikerjakan user
- [ ] Input `shelf_life_months` di form/import Master Produk `ipc_app`
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
- [x] Push ke GitHub — sudah dilakukan sejak beberapa commit lalu, remote
      `origin` = `github.com/bagus-chalil/vision-service`
- [x] CI/CD pilot infra ke VPS terpisah (2026-09-28, lihat section "Infra:
      CI/CD pilot ke VPS" di atas untuk detail/gotcha) — `.gitlab-ci.yml`,
      `.github/workflows/mirror-to-gitlab.yml`, `ops/provision_vps.sh`,
      `ops/deploy.sh`, `ops/vision-service.service`,
      `ops/configure_firewall.sh`, fix `requirements.txt` (extra-index-url
      CPU-only torch). Branch `production` dibuat & di-push, GitHub secrets
      (`GITLAB_PROJECT_URL`, `GITLAB_TOKEN`) sudah di-set, mirror workflow
      confirmed jalan.
- [x] Provisioning VPS pilot itu sendiri (VM 104, `pve1`) — DONE 2026-09-28.
      `ops/provision_vps.sh` sukses jalan, GitLab Runner terdaftar (tag
      `vps-aiocr`, shell executor, status Online), ufw dikonfigurasi
      (`10.10.162.0/24` + `100.100.160.0/24` → port 8000, SSH terbuka ke
      semua). 2 gotcha baru ketemu & di-fix di proses ini (libGL + CPU
      AVX Proxmox, lihat Infra gotcha #10-11) — sisanya (#9, #12, #13) di
      Infra section di atas.
- [x] End-to-end deploy pilot (push `production` → GitLab CI → VPS restart
      service via `ops/deploy.sh`) — DONE 2026-09-28. Pipeline yang sempat
      pending lama auto-jalan begitu runner online, `deploy` job gagal di
      percobaan pertama (root cause: gotcha libGL + CPU AVX), setelah
      di-fix manual lalu job di-retry dari GitLab UI dan **sukses**.
      `vision-service` systemd `active (running)`, `/api/health` OK,
      testing manual via `index.html` (Backend URL → `http://100.100.160.31:8000`)
      confirmed jalan.

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
