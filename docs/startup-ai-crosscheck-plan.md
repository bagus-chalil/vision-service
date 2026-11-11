# Startup AI Cross-Check — Rencana & Requirement

Status: **perencanaan** (belum ada kode). Dibuat 2026-11-11.
Cakupan: vision-service (Python, dokumen ini) + ipc_app (Laravel, `C:\laragon\www\new_trial_validation_app\ipc_app`).

## 1. Tujuan
Saat Startup Check, user memotret **Label Bulk**, **IM**, dan **coding tube/produk**. Sistem membaca foto-foto itu, menghitung EXP dan batch yang seharusnya dengan rumus, lalu membandingkannya. Kalau tidak sesuai, user mendapat popup saat lanjut ke Filling/Packing.

## 2. Pembagian tanggung jawab (tidak boleh dilanggar)
| Komponen | Tugas | Dilarang |
|---|---|---|
| vision-service | Membaca foto (PC LLM untuk Label Bulk/IM, PaddleOCR untuk tube). Mengembalikan teks apa adanya + keterbacaan + confidence | Menghitung, memutuskan PASS/FAIL, menebak/memperbaiki nilai |
| ipc_app | Parsing, hitung EXP, bandingkan, putuskan PASS/FAIL/REVIEW, gate, popup, audit | Memperbaiki nilai hasil baca otomatis |

`field_type` selalu dikirim dari luar (langkah workflow), tidak pernah ditebak dari isi foto.

## 3. Alur
1. Label Bulk → `no_batch`, `mixing_date` (field "Mixing Date" saja).
2. IM → umur simpan per kemasan ("N Tahun dari Tanggal Mixing"), disimpan **dalam bulan** (tahun × 12).
3. ipc_app hitung: EXP diharapkan = Mixing Date + umur simpan (bulan), batch diharapkan = 3 huruf setelah `/`.
4. Pembanding: hasil baca tube (EXP & batch, dua foto terpisah; non-tube dari foto coding), nilai yang diketik user di Startup Check, hasil hitungan.
5. Keputusan PASS/FAIL/REVIEW di ipc_app → popup + gate di server.

## 4. Requirement fungsional
- F1. Field foto baru `label_bulk` di Startup Check (kamera langsung, tanpa file picker). `im_number` sudah ada.
- F2. Batch = tepat 3 huruf setelah `/` pada No Batch (`26W137/MJB` → `MJB`). Bukan tepat satu kandidat → REVIEW.
- F3. Abaikan "Exp. Date" tulisan tangan di Label Bulk (expiry bulk 1–3 bulan) dan stiker timestamp.
- F4. Umur simpan IM selalu tahun → konversi ke bulan. Kemunculan Bottle dan Unit Box berbeda → REVIEW. Tampilan di UI: bulan ÷ 12.
- F5. Keputusan: **PASS** semua cocok & terbaca; **FAIL** terbaca jelas tapi tidak cocok; **REVIEW** tidak terbaca/ambigu/confidence rendah/tanggal tidak ada/AI tidak tersedia/EXP ≤ Mixing Date.
- F6. Gate (server): FAIL memblok masuk Filling/Packing. REVIEW dan "AI tidak tersedia" boleh lanjut hanya dengan alasan tertulis yang tercatat (siapa, kapan, alasan).
- F7. Popup memuat nilai diharapkan, nilai terbaca, dan saran (foto ulang / koreksi data).
- F8. Audit setiap pemeriksaan dan override. Gate tidak berlaku mundur (feature flag; batch yang sudah lewat Startup tidak terkena).

## 5. Requirement non-fungsional
- Near-zero false PASS; ragu → REVIEW. Nilai OCR/LLM tidak pernah di-strip/dikoreksi.
- Hanya tabel baru (`ipc_startup_ai_checks`) + satu field foto; tanpa migrasi destruktif.
- Satu pemanggilan ±25–30 dtk, layanan memproses satu per satu → pola sinkron dengan spinner (seperti OCR sekarang); php-fpm/nginx timeout ≥120 dtk. Queue hanya bila terasa lambat.
- LLM mati/timeout → status "tidak dapat diverifikasi", bukan PASS.
- LLM hanya di jaringan kantor; foto tidak keluar jaringan.

## 6. Aturan tanggal (kalender sungguhan)
- Tambah bulan memakai kalender (bukan 30 hari/365 hari): Jan 31, Feb 28/29, Apr 30, dst.
- EXP = hari yang sama + N bulan. `ExpDateCalculator` ipc_app sudah begitu; diperluas tesnya.
- Hari yang tidak ada di bulan tujuan → **REVIEW**, popup menampilkan kandidat "hari terakhir bulan". Tidak pernah PASS otomatis sampai QC memutuskan aturan akhir bulan.
- Tahun 4 digit (Label Bulk) dan 2 digit (tube) dinormalisasi sebelum dibandingkan (format DDMMYY).

| Mixing | Umur simpan | Hasil |
|---|---|---|
| 08/09/2026 | 24 bln | `080928` |
| 29/02/2028 | 12 bln | 29/02/2029 tidak ada → REVIEW (kandidat 28/02/2029) |
| 29/02/2028 | 48 bln | `290232` (kabisat, ada) |
| 31/01/2026 | 1 bln | REVIEW (kandidat 28/02/2026) |
| 31/03/2026 | 1 bln | REVIEW (kandidat 30/04/2026) |
| 28/02/2027 | 12 bln | `280228` |

## 7. Model LLM
- Server: Ollama `0.40.13`, container `ollama`, saat ini terikat `127.0.0.1:11434` (belum bisa diakses dari luar server → urusan integrasi DevOps).
- Model: **`qwen3.8:27b`** (arsitektur qwen35, 27.3B, Q4_K_M, context 262144, capabilities: **vision**, thinking, tools). Terkonfirmasi lewat `ollama show`.
- **Jangan** pakai `cosmax-image-prompt:latest` (basis sama, tapi system prompt untuk pembuat prompt gambar) maupun `gpt-oss:20b` (teks saja).
- Parameter bawaan model `temperature 1` → **wajib di-override per request**: `temperature 0`, output JSON, thinking dimatikan atau `low` (default `medium` memperlambat). Kecepatan sebenarnya harus diukur di tahap 1.
- Prompt: jawab JSON saja, field yang tidak yakin = `null`, jangan mengarang teks yang tidak ada di foto, satu kali retry bila JSON rusak.

## 8. Kontrak data (vision-service → ipc_app)
```
label_bulk:  { no_batch_raw, mixing_date_raw, kode_bulk, legible:{...}, confidence, status }
im_document: { im_number, shelf_life:[{packaging, value, unit, raw_text}], legible, confidence, status }
```
Field_type baru: `label_bulk`, `im_document` (LLM), `tube_batch_code` (PaddleOCR, sisi batch tube, mis. `CDA`). Ketiganya didaftarkan juga di `ipc_app/config/vision.php`.

## 9. Tahapan
| # | Tahap | Selesai bila |
|---|---|---|
| 0 | Kumpulkan data uji: ±30 Label Bulk, ±10 IM bervariasi + kunci jawaban. Sampel awal ada di `images/sources/` (IM.jpg, Label Bulk*.jpg, Label FG, WI). Sumber tambahan: foto produksi `C:\Users\cosmax\Downloads\ipc-attachments` (7088 jpg, `{batch_id}/{stage}/{hash}.jpg`; 1772 di `startup/`). **Nama file di-hash, jenis foto (field_label) hanya ada di tabel `ipc_attachments` DB `ipc_system`** → pemetaan lewat query read-only. Foto Label Bulk belum punya field sendiri di produksi; kandidatnya `filling/date_bulk` (perlu dicek manual), foto IM = `startup/im_number`. | Set uji terpilih + kunci jawaban |
| 1 | vision-service: `label_bulk` + skema JSON + script evaluasi di `tools/` + ukur kecepatan | Akurasi per field terukur, tanpa tebakan |
| 2 | vision-service: `im_document` (aturan Bottle vs Unit Box) | Idem |
| 3 | ipc_app: `StartupCrossCheck` (parser, hitung EXP bulan, keputusan) + Pest dengan tabel §6 | Semua kasus tepi lulus |
| 4 | ipc_app: foto `label_bulk`, tabel `ipc_startup_ai_checks`, panel hasil | Hasil tampil di Startup Check |
| 5 | ipc_app: gate server + popup + override beralasan + audit | Test per flow (full / filling / packing_fg) |
| 6 | vision-service: `tube_batch_code`, dihubungkan ke tahap 3 | Cocok dengan foto sampel |
| 7 | Uji lapangan terbatas (feature flag, 1–2 operator) | Tidak ada false PASS, waktu bisa diterima |

## 10. Risiko
- Foto IM kecil/terpotong → baca umur simpan paling berisiko.
- LLM 27B bisa lambat; antrean satu-per-satu menumpuk saat banyak line.
- Halusinasi LLM → mitigasi: validasi silang dengan nilai ketikan user + aturan `null` bila ragu.
- Data uji awal terlalu sedikit untuk mengukur akurasi.

## 11. Keputusan terbuka
1. ~~QC: aturan akhir bulan~~ **Diputuskan 2026-11-11 (user):** karena umur simpan selalu tahun penuh, satu-satunya hari yang bisa tidak ada di bulan tujuan adalah **29 Februari** (kabisat) → REVIEW. Sisanya deterministik, tidak perlu aturan tambahan. Aturan umum "hari tidak ada → REVIEW" tetap dipertahankan sebagai pengaman.
2. **DevOps:** cara vision-service mengakses Ollama (binding `127.0.0.1` → lewat caddy/port terbuka), saat integrasi.
3. Tahap 0: siapa yang mengumpulkan foto uji dan kunci jawabannya.
4. Apakah foto IM perlu panduan pengambilan di UI (zoom ke bagian umur simpan).
