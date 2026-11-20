# Prompt: integrasi `coding_formula` ke ipc_app

Salin seluruh isi di bawah garis ke sesi Claude Code yang dibuka di
`C:\laragon\www\new_trial_validation_app\ipc_app`.

---

## Tugas

Ganti pembacaan AI untuk **Primary Coding** di Startup Check (foto `coding_primary` + foto kedua opsional
`coding_primary_2`) dari jalur lama (`coding_print` + `tube_exp_date` + `tube_exp_llm`) ke field_type baru
Vision Service **`coding_formula`**. Field type baru ini mencari batch dan EXP menurut formula coding,
di mana pun letaknya di foto (emboss crimp, inkjet di badan tube, kotak label, tutup). Pada tube, KEDUA
foto dicari batch dan EXP, lalu hasilnya digabung di Vision Service.

## Baca dulu sebelum mengubah apa pun

1. `CLAUDE.md` ipc_app dan README deploy-nya. Catat perubahan di bagian Status-nya setelah selesai.
2. `config/vision.php` (`llm_field_types`, `startup_ai.fields`, `tube_field_types`, `tube_llm_field_type`).
3. `app/Actions/Vision/AnalyzeStartupPhoto.php`, `QueueStartupAiReads.php`, `ReadStartupPhoto.php`,
   `RecordStartupAiFeedback.php`, `StartupAiStatus.php`.
4. `app/Services/Vision/StartupVisionEvaluator.php` (terutama `codingPrint()` dan `tubeLlmItem()`),
   `HttpVisionClient.php`, `VisionResult.php`.
5. `resources/js/pages/startup-check/edit.tsx` dan tes `tests/Feature/StartupAi*Test.php`.
6. Referensi kontrak di repo vision-service: `C:\laragon\www\vision-service\CLAUDE.md` gotcha #24,
   `coding_formula.py` (docstring paling atas = sintaks formula), `coding_formulas.json` (katalog contoh).

## Aturan yang tidak boleh dilanggar

- AI tetap **ADVISORY**. Operator tetap mengisi manual. Jangan auto-isi toggle Conform/Not Conform, dan
  jangan melonggarkan `VisionEvaluator`.
- Keputusan PASS/FAIL/REVIEW hanya ada di Laravel. Vision Service hanya mengembalikan bacaan + status.
- Apa pun yang bukan `status: "OK"` pada field yang dibutuhkan berarti **REVIEW**, tidak pernah PASS.
  `CONFLICT`, `LOW_CONFIDENCE`, `ERROR`, dan field yang tidak ada semuanya masuk REVIEW.
- Nilai ditampilkan persis seperti dibaca (`value`), tidak pernah di-strip atau dikoreksi.
- Kerjakan di branch `development`, bukan `production`.
- Daftarkan `coding_formula` di `config/vision.php` hanya setelah `php artisan vision:ping` menunjukkan
  field_type itu ada di server Vision Service. Kalau belum ada, berhenti dan laporkan: vision-service perlu
  di-deploy dulu, karena kodenya belum di-commit/deploy saat prompt ini ditulis.
- Buat migrasi baru, jangan mengubah migrasi lama. Jangan menyentuh data produksi.

## Kontrak API Vision Service

`POST {VISION_SERVICE_URL}/api/analyze`, multipart:

| Field | Isi |
|---|---|
| `file` | foto coding pertama (wajib) |
| `file2` | foto kedua (opsional; tube = sisi lain) |
| `field_type` | `coding_formula` |
| `coding_formula` | kosong atau `auto` = mode AUTO (semua formula katalog dicoba). Atau JSON array template, mis. `["{BATCH3}","EXP {MMM} {YY}"]`. Atau kunci katalog (`label_exp_ddmmyy_init`, dst.) |
| `request_id` | UUID (opsional) |

Header `X-API-Key` bila `VISION_API_KEY` diset di server, sama seperti field_type LLM lain.

Formula `auto` mewajibkan batch + exp. Formula SKU mewajibkan semua dari batch/exp/mfg yang ada di
formulanya. Formula yang tidak valid → HTTP **400** `{"status":"ERROR","error":"FORMULA_INVALID: ..."}`.
Ollama mati → HTTP 200 `{"status":"ERROR","error":"LLM_UNAVAILABLE: ..."}`.

Contoh respons (2 foto tube, mode auto):

```json
{
  "status": "OK",
  "formula_source": "auto",
  "required_fields": ["batch", "exp"],
  "fields": {
    "batch": {"status": "OK", "value": "PLB", "photos": [1], "matched_text": "PLB",
              "views": [{"status": "FOUND", "value": "PLB"}], "required": true},
    "exp":   {"status": "OK", "value": "181129", "date": {"year": 2029, "month": 11, "day": 18},
              "granularity": "day", "parts": {"day": "18", "month": "11", "year2": "29"},
              "matched_text": "EXP DATE 181129 TQ", "photos": [0], "required": true},
    "init":  {"status": "OK", "value": "TQ", "required": false}
  },
  "photos": [ {"status": "LOW_CONFIDENCE", "fields": {}, "views_read": [{"side": 1024, "lines": ["..."]}]} ],
  "confidence": 0.95, "engine_used": "ollama", "model": "qwen3.8:27b",
  "attempts": 6, "elapsed_sec": 7.1, "field_type": "coding_formula", "pipeline": "llm"
}
```

Field yang konflik antar foto:
`{"status": "CONFLICT", "value": null, "per_photo": [{"photo": 0, "status": "OK", "value": "OKB"}, {"photo": 1, "status": "OK", "value": "AKB"}]}`.

**EXP harus dibandingkan memakai `fields.exp.date` sesuai `granularity`, bukan memakai string `value`:**
- `granularity = "day"` → banding tanggal penuh dengan `ipc_batches.exp_date`.
- `granularity = "month"` → banding bulan + tahun saja.

Bentuk cetakan bisa `181129`, `27.10.29`, `EXP NOV 29`, `10 2029`, `112028` (MMYYYY), `17112029`. Semua bentuk
itu sudah diurai ke `date` oleh Vision Service, jadi jangan mem-parse `value` lagi di PHP.

Batch dibandingkan `fields.batch.value` dengan 3 huruf setelah `/` pada `ipc_batches.no_batch`. Ada batch
yang tercetak panjang (mis. `26Y070X`). Kalau panjangnya bukan 3, tampilkan saja sebagai info dan REVIEW,
jangan dipaksa cocok.

Waktu: rata-rata ±2,5 detik per foto, kasus OK ±4 detik per foto, 2 foto ±8–10 detik. `llm_timeout` yang ada
(150 dtk) sudah cukup.

## Yang perlu dikerjakan

1. **Formula per produk (opsional, default auto).** Tambah kolom nullable `coding_formula` (JSON / text) di
   `master_products`, lewat migrasi baru. Isi null berarti kirim `auto`. Tambahkan input di form/import
   Master Produk bila sederhana. Kalau tidak sederhana, cukup kolom + seeder kosong, lalu catat sebagai TODO.
   Jangan memaksa admin mengisi.
2. **Config.** Tambah `coding_formula` ke `llm_field_types`. Arahkan `startup_ai.fields.coding_primary` ke
   `coding_formula`. Foto `coding_primary_2` tidak lagi dibaca sendiri, tapi dikirim sebagai `file2` pada
   panggilan yang sama. Jadikan jalur lama (`tube_field_types`, `tube_llm_field_type`) bisa dimatikan lewat
   config/env dengan default MATI untuk `coding_formula`. Jangan hapus kodenya dulu, supaya bisa rollback
   lewat env.
3. **Client.** `HttpVisionClient` harus bisa mengirim `file2` dan `coding_formula`. Pertahankan perilaku
   panggilan field_type lain.
4. **Antrean/aksi.** `QueueStartupAiReads` / `ReadStartupPhoto` / `AnalyzeStartupPhoto`: satu panggilan
   `coding_formula` per batch untuk Primary Coding, berisi foto pertama + foto kedua (bila ada) + formula
   produk (atau `auto`). Kalau foto kedua baru diunggah belakangan, panggil ulang dengan kedua foto.
   Jangan memanggil dua kali secara terpisah.
5. **Evaluator.** Tambahkan cabang `coding_formula` di `StartupVisionEvaluator`:
   - item `coding_batch` dan `coding_exp` dengan aturan di atas;
   - `CONFLICT` tampilkan nilai per foto dengan catatan "foto 1 dan foto 2 terbaca beda - cek manual";
   - `formula_source = "auto"` tampilkan sebagai info "formula otomatis (SKU belum punya formula)";
   - tube dan non-tube memakai aturan yang sama. Kebijakan lama "tube selalu REVIEW" (`codingPrint()`)
     **jangan diubah diam-diam**: pertahankan sebagai default lewat config (mis.
     `vision.startup_ai.tube_force_review = true`) dan laporkan supaya user memutuskan.
6. **Log & feedback.** Simpan respons ke `ipc_logs` seperti field_type LLM lain (termasuk `formula_source`
   dan selisih AI vs input operator). `RecordStartupAiFeedback` / `StartupAiStatus` harus mengenali
   `coding_formula` untuk `coding_batch`/`coding_exp`.
7. **UI** (`startup-check/edit.tsx`): tampilkan batch & EXP hasil AI, sumber foto (foto 1/2), status, dan
   catatan CONFLICT/auto. Teks berbahasa Indonesia, gaya sama seperti item AI yang sudah ada.
8. **Tes** (Pest/PHPUnit, Vision Service di-fake, jangan memanggil server nyata):
   - OK cocok → PASS-advisory;
   - EXP granularity `month` cocok bulan/tahun walau hari beda;
   - beda nilai → FAIL/REVIEW sesuai aturan yang ada;
   - `CONFLICT` → REVIEW;
   - `LOW_CONFIDENCE` → REVIEW;
   - `ERROR` / 400 → REVIEW dan tidak crash;
   - `file2` ikut terkirim;
   - formula produk null → `auto`;
   - jalur lama tidak dipanggil saat `coding_formula` aktif.

   Jalankan seluruh test suite + Pint + `tsc`. Gotcha lokal: `pdo_sqlite` di PHP 8.5 Laragon diblokir
   Application Control. Kalau tes sqlite gagal karena itu, pakai instance MySQL sementara seperti
   integrasi sebelumnya (lihat Status ipc_app).
9. **Verifikasi nyata (bila server terjangkau):** `php artisan vision:ping`, lalu satu permintaan asli
   dengan foto coding non-tube dan satu pasang foto tube. Laporkan respons mentahnya.

## Laporan akhir yang diminta

Ringkas dalam Bahasa Indonesia:
- file yang diubah/dibuat;
- migrasi;
- flag config/env baru beserta default-nya;
- hasil test suite;
- hasil `vision:ping` / permintaan nyata;
- keputusan yang perlu dibuat user, minimal: (a) apakah tube tetap dipaksa REVIEW, (b) siapa yang mengisi
  formula per SKU;
- daftar hal yang sengaja belum dikerjakan.
