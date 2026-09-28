# Vision Service — OCR Test Tool (Prototype)

Local testing tool untuk uji akurasi PaddleOCR pada dokumen Label Bulk/FG dan WI
Filling/Packing, sebelum kita define Fixed ROI per field. **Ini bukan production
code** — tidak ada auth, HTTPS, atau request_id/API contract final.

## Struktur project

```
main.py                       FastAPI backend. /api/analyze = 1 entrypoint, dispatch otomatis ke
                               generic pipeline (field_patterns.json) atau tube_emboss pipeline
                               (emboss_format_patterns.json) berdasarkan field_type yang dikirim
tube_emboss_pipeline.py       Sub-pipeline: YOLO tube localize + crimp OCR + format validasi
field_patterns.json           Config regex per field_type generic (no localization) - edit manual, no restart
emboss_format_patterns.json   Config block-based (day/month/year/batch/mfg_code) - field_type di sini
                               otomatis lewat tube_emboss pipeline, bukan generic
index.html                    Frontend testing UNIFIED - 1 dropdown field_type untuk semua tipe,
                               otomatis pakai pipeline yang benar per tipe (lewat /api/analyze)
tube_emboss.html              Frontend testing standalone khusus tube emboss sub-pipeline (dev/debug)
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
pip install -r requirements.txt
```

`requirements.txt` adalah hasil `pip freeze` dari venv yang sudah tested
(termasuk `ultralytics` untuk YOLO tube detector, yang sebelumnya tidak
tercatat di sini). **Pakai file ini, jangan `pip install paddleocr
paddlepaddle ...` versi longgar** — paddleocr 3.x API-nya beda total dari
2.x (lihat `CLAUDE.md` gotcha #1), jadi instalasi tanpa pin versi berisiko
resolve ke versi yang tidak kompatibel di server lain.

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

1. Pilih **Field type** di dropdown — ini gabungan dari `field_patterns.json`
   dan `emboss_format_patterns.json`. Frontend tidak perlu tahu pipeline mana
   yang dipakai; backend (`/api/analyze`) yang menentukan otomatis dari
   field_type yang dipilih.
2. Upload foto dari file, atau klik **Use camera** untuk ambil foto langsung
   (simulasi kamera HP).
3. Klik **Run OCR**.
4. Hasil tampil beda bentuk tergantung pipeline field_type-nya:
   - **Generic** (`generic`, `wo_number`, `batch_no`, dst — whole-image OCR,
     tanpa localization): tabel semua text region yang terdeteksi, kolom
     **Validation** (`FORMAT_OK` / `FORMAT_MISMATCH` / `NOT_APPLICABLE`),
     kolom **Status** gabungan (`OK` / `LOW_CONFIDENCE` / `FORMAT_MISMATCH`),
     bounding box per region di atas gambar.
   - **Tube emboss** (`tube_emboss_default`, dst — YOLO localize + crop dulu
     baru OCR): satu hasil tunggal (raw_ocr_text, confidence, format_valid,
     status), bounding box tube + garis crop, plus preview crop tube & crimp
     di bawah gambar.
5. Kalau ada catatan ROI/format mismatch, muncul di panel kuning di bawah
   tabel hasil.

## Menambah / edit field type

Ada 2 config, pilih sesuai kebutuhan field-nya:

- **`field_patterns.json`** — generic, whole-image OCR + regex, TANPA
  localization. Cocok untuk field yang teksnya sudah cukup terisolasi di
  foto (misal WO number di form). Dibaca ulang setiap request, **tidak
  perlu restart backend**. Struktur satu entry:

  ```json
  "wo_number": {
    "label": "Work Order Number",
    "description": "WO + 6 digits, e.g. WO123456",
    "pattern": "^WO\\d{6}$",
    "expected_length": 8,
    "roi_hint": "Catatan opsional, muncul di UI saat FORMAT_MISMATCH terjadi pada field ini."
  }
  ```

  - `pattern` — regex Python (`re.fullmatch`, harus match keseluruhan teks).
    Set `null` kalau field ini tidak perlu validasi format (selalu
    `NOT_APPLICABLE`, hanya pakai confidence seperti biasa).
  - `roi_hint` — catatan yang tampil **hanya saat mismatch terjadi**.
    Set `null` kalau tidak ada catatan.

- **`emboss_format_patterns.json`** — field yang butuh localization dulu
  sebelum OCR (foto berisi banyak elemen lain, bukan cuma teksnya). Saat ini
  cuma tube cap emboss (`tube_emboss_default`, lewat YOLO + crop di
  `tube_emboss_pipeline.py`). Field type yang key-nya ada di file ini
  **otomatis** di-route ke pipeline itu oleh `/api/analyze` — tidak perlu
  pengaturan tambahan di frontend. Kalau nanti WI/exp date juga butuh
  localization sendiri (Fixed ROI + homography), pipeline barunya didaftarkan
  di sini juga, dengan pola yang sama.

- Teks hasil OCR **tidak pernah** di-strip/trim otomatis oleh sistem, di
  kedua config — dicocokkan apa adanya ke pattern. Kalau mismatch, itu tetap
  harus direview manusia, bukan ditebak/dipotong otomatis.

## Deploy checklist (manual, Windows/on-prem — mis. `.24`)

1. `git clone` repo ini — `models/tube_detector_v1/best.pt` (YOLO tube
   detector) sudah ikut ter-commit, jadi tidak perlu download model
   terpisah.
2. `pip install -r requirements.txt` di venv baru (lihat Setup di atas) —
   jangan install versi longgar, terutama untuk `paddleocr`/`paddlepaddle`.
3. **Sebelum** menjalankan `main.py` penuh, jalankan dulu
   `python tests/test_ocr.py` di mesin target. Workaround oneDNN
   (`enable_mkldnn=False`, lihat Known issues di bawah) sudah di-hardcode di
   `main.py`, tapi CPU server bisa beda instruction set dari mesin dev ini —
   pastikan PaddleOCR bisa load & inference dulu sebelum dianggap siap
   pilot.
4. Set environment variable `VISION_SERVICE_ALLOWED_ORIGINS` ke origin
   frontend yang benar-benar dipakai (comma-separated kalau lebih dari
   satu), sebelum start uvicorn:
   ```powershell
   $env:VISION_SERVICE_ALLOWED_ORIGINS = "http://100.100.160.23"
   ```
   Kalau env var ini tidak di-set, CORS default terbuka (`*`) dan service
   akan print warning di log saat startup. Ini bukan pengganti auth/HTTPS
   (belum ada, lihat Prinsip desain) — cuma mengurangi permukaan serang
   selama pilot.
5. Batasi akses jaringan ke port service (default 8000) hanya dari IP yang
   memang butuh, pakai `ops\configure_firewall.ps1`:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\ops\configure_firewall.ps1 -AllowedIPs "100.100.160.23"
   ```
6. Jalankan service via `ops\run_service.ps1` (supervisor, auto-restart) dan
   daftarkan sebagai Scheduled Task via `ops\register_tasks.ps1` supaya
   survive reboot/crash tanpa perlu remote manual.

Ini semua masih pilot-scope (LAN internal, field `tube_emboss_default` +
`tube_exp_date` saja) — bukan production architecture penuh (Laravel/React/
kontrak API final masih belum dikerjakan, lihat `CLAUDE.md`).

## Pilot via GitLab CI/CD ke VPS (jalur latihan DevOps, terpisah dari `.24`)

Ini jalur **kedua**, terpisah dari deploy manual ke `.24` di atas — dipakai
untuk latihan praktik CI/CD ke sebuah Proxmox VPS kosongan (4 vCPU / 4GB RAM
/ 32GB disk, Ubuntu 26.04). **GitHub tetap source of truth**; GitLab cuma
jadi tempat pipeline jalan.

Alur: push/merge ke branch **`production`** di GitHub →
`.github/workflows/mirror-to-gitlab.yml` push-mirror *khusus branch ini* ke
project GitLab → `.gitlab-ci.yml` di GitLab jalan otomatis (di-gate ke
`CI_COMMIT_BRANCH == "production"`) → job `deploy` dieksekusi oleh GitLab
Runner yang terdaftar **di VPS itu sendiri** (bukan runner terpisah yang SSH
masuk) → `ops/deploy.sh` sync kode + install deps + smoke test + restart
service.

Branch `production` sengaja dipisah dari `main`/`v1.0.0` — isinya sama
persis (dibuat dari situ), tapi satu-satunya fungsinya cuma jadi pemicu
pipeline pilot ini. Kerja harian tetap di `main`/`v1.0.0` seperti biasa;
begitu ada perubahan yang mau benar-benar di-pilot-kan ke VPS, merge/push ke
`production`.

Kenapa **native (venv + systemd), bukan Docker**: image Docker untuk stack
ini (torch + paddlepaddle + ultralytics + opencv) gampang tembus 3–4GB, plus
overhead Docker daemon sendiri — terlalu berisiko di RAM 4GB/disk 32GB.
Revisit kalau VM di-upgrade nanti.

**Gotcha penting**: `requirements.txt` sekarang menyertakan
`--extra-index-url https://download.pytorch.org/whl/cpu` di baris atas.
Tanpa ini, `pip install torch==2.14.0` di Linux x86_64 resolve ke build
CUDA-enabled (beberapa GB paket `nvidia-*-cu12` yang gak kepakai sama
sekali di VPS tanpa GPU ini) alih-alih build `+cpu` yang sudah divalidasi di
Windows — bisa menghabiskan disk 32GB percuma. Jangan hapus baris itu.

### Setup sekali di VPS (sebelum pipeline pertama jalan)

1. Buat project GitLab (mirror dari `bagus-chalil/vision-service`), simpan
   URL + Project Access Token (scope `write_repository`) sebagai GitHub
   Actions secrets di repo GitHub: `GITLAB_PROJECT_URL`
   (`https://gitlab.com/<namespace>/<project>`, dengan `https://`) dan
   `GITLAB_TOKEN`. *(Sudah dibuat: `GITLAB_PROJECT_URL` →
   `https://gitlab.com/cosmaxidn/vision-service`, `GITLAB_TOKEN` → token.)*
2. Clone repo ini ke VPS (sementara, buat bootstrap saja), lalu jalankan
   sekali sebagai root:
   ```bash
   git clone https://github.com/bagus-chalil/vision-service.git /root/bootstrap
   cd /root/bootstrap
   sudo ./ops/provision_vps.sh
   ```
   Script ini idempotent (aman dijalankan ulang) — install Python 3.11,
   swap 4GB, user service `visionsvc`, GitLab Runner, systemd unit, dan
   sudoers rule yang **cuma** mengizinkan runner menjalankan satu script
   privileged (`/usr/local/bin/vision-service-deploy.sh`), tidak ada
   command lain yang di-`sudo`-kan ke runner.
3. Ikuti langkah manual yang di-print di akhir `provision_vps.sh`: register
   runner (`sudo gitlab-runner register --tag-list vps-aiocr ...`) pakai
   token dari GitLab, jalankan `ops/configure_firewall.sh <allowed-ip>`, dan
   **cek juga firewall Proxmox di level VM** (NIC VM ini `firewall=1`) —
   ufw saja tidak cukup kalau Proxmox-nya sendiri masih block.
4. Deploy pertama kali manual (sebelum pipeline ada history):
   `sudo /usr/local/bin/vision-service-deploy.sh /root/bootstrap`.

Setelah itu, push/merge ke branch `production` di GitHub → otomatis sampai
ke VPS. Detail desain (kenapa 1 script privileged, kenapa smoke test jalan
sebelum restart service, dll) ada di komentar masing-masing file
`ops/*.sh`.

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
