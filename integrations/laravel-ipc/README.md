# Integrasi Vision Service → Laravel IPC (`ipc_app`)

Bundle ini berisi kode **sisi Laravel** untuk memanggil Vision Service (OCR) dari `ipc_app`,
lalu memutuskan PASS / FAIL / REVIEW dan mencatat audit di tabel `ipc_logs`.

> **Tidak ada file Python yang dipindah ke `ipc_app`.** Vision Service tetap jalan di
> servernya sendiri (pilot: `http://100.100.160.31:8000`). Laravel cuma memanggilnya lewat
> HTTP. Yang di-copy ke `ipc_app` hanya isi folder `files/` di bawah ini.

Sudah diverifikasi 2026-09-28 terhadap `ipc_app` branch `production`:
- seluruh test suite `ipc_app` lulus (255 test, termasuk 18 test baru dari bundle ini), di
  MySQL 8.4 + PHP 8.5;
- Pint lulus, hook TypeScript lulus `tsc --strict`;
- `php artisan vision:ping` dan request multipart asli ke VPS berhasil (foto Pond's UV Protect:
  MFD `140624` + 36 bulan = `140627`).

---

## Pembagian tugas

| Siapa | Tugasnya |
|---|---|
| **Vision Service** (Python, VPS) | Baca foto → kembalikan nilai OCR apa adanya + confidence + `format_valid`. **Tidak pernah** memutuskan PASS/FAIL. |
| **Laravel** (`ipc_app`, bundle ini) | Tentukan `field_type` dari langkah kerja, kirim foto, bandingkan hasil dengan data batch (`ipc_batches.exp_date`, shelf-life SKU), putuskan **PASS / FAIL / REVIEW**, simpan audit ke `ipc_logs`. |

---

## Isi bundle

Struktur `files/` sama persis dengan root `ipc_app`. **Semuanya file baru**, jadi tidak ada
file lama yang tertimpa.

```
files/
├── config/vision.php                         URL, timeout, field_type yang diizinkan, aturan per field
├── app/Services/Vision/
│   ├── VisionClient.php                      interface (bisa di-fake di test)
│   ├── HttpVisionClient.php                  POST /api/analyze (multipart)
│   ├── VisionResult.php                      hasil baca OCR (bukan keputusan)
│   ├── VisionEvaluator.php                   ← logika keputusan PASS / FAIL / REVIEW
│   ├── VisionEvaluation.php                  hasil keputusan + alasan
│   └── ExpDateCalculator.php                 EXP = MFD + shelf-life (bulan)
├── app/Actions/Vision/AnalyzeVisionField.php simpan foto → panggil Vision → evaluasi → tulis ipc_logs
├── app/Http/Controllers/VisionController.php endpoint JSON
├── app/Http/Requests/AnalyzeVisionRequest.php validasi upload
├── app/Models/IpcLog.php
├── app/Console/Commands/VisionPing.php       php artisan vision:ping
├── routes/vision.php
├── database/migrations/
│   ├── 2026_09_28_100000_add_shelf_life_months_to_master_products_table.php
│   └── 2026_09_28_100001_create_ipc_logs_table.php
├── resources/js/hooks/use-vision-analyze.ts  hook React untuk memanggil endpoint
└── tests/
    ├── Feature/VisionAnalyzeTest.php         14 test (Http::fake, tanpa perlu VPS)
    └── Unit/ExpDateCalculatorTest.php
```

---

## Langkah integrasi

### 1. Copy isi `files/` ke root `ipc_app`

PowerShell (dari mesin yang punya kedua repo):

```powershell
robocopy "C:\laragon\www\vision-service\integrations\laravel-ipc\files" "C:\laragon\www\new_trial_validation_app\ipc_app" /E
```

Bash:

```bash
cp -rn vision-service/integrations/laravel-ipc/files/. ipc_app/
```

### 2. Tiga edit manual di file yang sudah ada

**a. `app/Providers/AppServiceProvider.php`**, tambahkan binding di `register()`:

```php
use App\Services\Vision\HttpVisionClient;
use App\Services\Vision\VisionClient;

public function register(): void
{
    $this->app->bind(PdfRenderer::class, BrowsershotPdfRenderer::class);
    $this->app->bind(VisionClient::class, HttpVisionClient::class);
}
```

**b. `routes/web.php`**, tambahkan setelah `require __DIR__.'/batches.php';`:

```php
require __DIR__.'/vision.php';
```

**c. `app/Models/MasterProduct.php`**, tambahkan `shelf_life_months` ke `$fillable` dan `casts()`:

```php
protected $fillable = [
    'fg_code',
    'product_name',
    'shelf_life_months',
    'is_active',
];

protected function casts(): array
{
    return [
        'is_active' => 'boolean',
        'shelf_life_months' => 'integer',
    ];
}
```

### 3. `.env`

```env
VISION_SERVICE_URL=http://100.100.160.31:8000
VISION_SERVICE_TIMEOUT=90
VISION_SERVICE_CONNECT_TIMEOUT=5
```

Tambahkan juga ke `.env.example` supaya developer lain tahu.

### 4. Migrate

```bash
php artisan migrate
php artisan storage:link   # kalau belum pernah, supaya image_url di response bisa dibuka
```

### 5. Isi shelf-life per SKU

Kolom `master_products.shelf_life_months` masih kosong untuk semua produk, dan form master
produk belum punya input untuk kolom ini. Sementara isi lewat tinker atau SQL:

```bash
php artisan tinker
>>> App\Models\MasterProduct::where('fg_code', 'FG-XXXX')->update(['shelf_life_months' => 36]);
```

Produk tanpa shelf-life tetap bisa di-scan, tapi hasil `tube_mfd_date`-nya selalu **REVIEW**
(EXP tidak bisa dihitung).

### 6. Cek koneksi dari server Laravel

Jalankan **di server Laravel** (`.23`), bukan di laptop, karena firewall VPS hanya menerima
port 8000 dari `10.10.162.0/24` dan `100.100.160.0/24`:

```bash
php artisan vision:ping
```

Output yang benar:

```
Vision Service: http://100.100.160.31:8000
Health: ok
  [ada]   tube_exp_date
  [ada]   tube_mfd_date
  [ada]   tube_emboss_default
```

### 7. Jalankan test

```bash
php artisan test --filter=Vision
php artisan test --filter=ExpDateCalculator
```

Test tidak butuh VPS (semua pakai `Http::fake()`).

### 8. Pakai di halaman React

Contoh di `resources/js/pages/packing-check/edit.tsx` untuk foto `primary_coding_batch_exp`:

```tsx
import { useVisionAnalyze } from '@/hooks/use-vision-analyze';

const vision = useVisionAnalyze(batch.id);

// dipanggil setelah QC pilih/ambil foto:
await vision.analyze(file, 'packing', 'tube_exp_date');

// tampilkan hasil:
{vision.processing && <p>Membaca kode... (±30 detik)</p>}
{vision.error && <p className="text-red-600">{vision.error}</p>}
{vision.result && (
    <div>
        <b>{vision.result.decision}</b> ({vision.result.decision_reason})
        <div>Terbaca: {vision.result.ocr_value ?? '-'}</div>
        {vision.result.computed_value && <div>EXP dihitung dari MFD: {vision.result.computed_value}</div>}
        <div>Seharusnya: {vision.result.expected_value ?? '-'}</div>
    </div>
)}
```

`field_type` **selalu ditentukan oleh halaman/langkah kerja**, bukan dipilih QC dan bukan
ditebak dari foto. Saran pemetaan (perlu dikonfirmasi ke QC):

| Halaman / foto | field_type |
|---|---|
| Packing → `primary_coding_batch_exp` (emboss crimp) | `tube_exp_date` |
| Packing → foto body tube yang ada tulisan "MFD" | `tube_mfd_date` |
| Finished → `exp_date` | `tube_exp_date` |
| Filling → `image_tube` | `tube_emboss_default` (belum ada aturan, jadi selalu REVIEW) |

---

## Endpoint

| Method | URL | Keterangan |
|---|---|---|
| `POST` | `/batches/{batch}/vision/analyze` | multipart: `photo`, `stage` (startup/filling/packing/finished), `field_type`. Auth + `can:update,batch` (Staff/Admin). Balikan JSON. |
| `GET` | `/batches/{batch}/vision/logs` | riwayat scan batch itu (terbaru dulu) |

Contoh response:

```json
{
  "decision": "REVIEW",
  "decision_reason": "Confidence OCR rendah - cek manual.",
  "field_type": "tube_mfd_date",
  "ocr_value": "140624",
  "computed_value": "140627",
  "expected_value": "140627",
  "vision_status": "LOW_CONFIDENCE",
  "confidence": 0.84,
  "format_valid": true,
  "error_reason": null,
  "image_url": "http://.../storage/ipc-attachments/12/vision/<uuid>.jpg"
}
```

---

## Aturan keputusan (`VisionEvaluator`)

KPI: false PASS mendekati nol. Urutan pengecekannya:

| Kondisi | Keputusan |
|---|---|
| Vision Service error / tidak bisa dihubungi / foto rusak | **REVIEW** (foto ulang) |
| `format_valid` bukan `true` (misal `LABEL_NOT_FOUND`) | **REVIEW** |
| `field_type` tidak punya aturan di `config('vision.rules')` | **REVIEW** |
| Data pembanding kosong (`ipc_batches.exp_date` kosong, shelf-life SKU kosong) | **REVIEW** |
| MFD + shelf-life jatuh di tanggal yang tidak ada (misal 31 + 1 bulan → Februari) | **REVIEW**, tidak digeser ke tanggal lain |
| Confidence rendah (`LOW_CONFIDENCE`), **walaupun nilainya cocok** | **REVIEW** |
| Confidence OK + format valid + **sama** dengan EXP batch | **PASS** |
| Confidence OK + format valid + **beda** dengan EXP batch | **FAIL** |

Aturan per `field_type` (di `config/vision.php`):
- `tube_exp_date` → `batch_exp_date`: EXP hasil OCR harus sama dengan `ipc_batches.exp_date` (DDMMYY).
- `tube_mfd_date` → `mfd_plus_shelf_life`: MFD hasil OCR + `master_products.shelf_life_months`
  harus sama dengan `ipc_batches.exp_date`.

Pada kondisi REVIEW, `expected_value` / `computed_value` tetap diisi kalau bisa dihitung,
supaya QC bisa lihat apa yang diharapkan sistem.

> Catatan realistis: di sample yang ada sekarang, hasil `tube_mfd_date` selalu
> `LOW_CONFIDENCE`, jadi keputusannya akan REVIEW (dengan EXP hitungan tetap ditampilkan).
> Ini memang disengaja, bukan bug.

---

## Tabel `ipc_logs`

Satu baris per scan. Kolom penting: `request_id` (UUID yang sama dengan yang dikirim ke
Vision Service, bisa dicocokkan dengan `logs/fallback_cases.jsonl` di VPS), `image_path`,
`vision_status`, `confidence`, `format_valid`, `ocr_value` (kode bersih),
`raw_ocr_text` (semua teks mentah, untuk audit), `error_reason`, `decision`,
`decision_reason`, `expected_value`, `computed_value`, `vision_response` (JSON lengkap),
`created_by`.

Foto scan disimpan di `storage/app/public/ipc-attachments/{batch}/vision/{request_id}.jpg`,
**bukan** sebagai `IpcAttachment`, jadi tidak mengubah foto "terkini" yang dipakai halaman
stage maupun laporan cetak.

---

## Catatan production

- **Lambat dan antre.** Satu scan sekitar 25–30 detik, dan Vision Service memproses request
  satu per satu. Kalau 3 stasiun scan bersamaan, stasiun ketiga bisa menunggu sekitar 90 detik.
  Tampilkan status "sedang membaca" di UI.
- **PHP `max_execution_time`.** Controller memanggil `set_time_limit(timeout + 30)`. Kalau
  server memakai php-fpm dengan `request_terminate_timeout`, atau ada proxy (nginx
  `fastcgi_read_timeout`), naikkan juga ke ≥ 120 detik.
- **Vision Service belum punya auth/HTTPS.** Aman selama hanya bisa diakses dari jaringan
  internal (sudah dibatasi firewall VPS). Jangan buka port 8000 ke publik.
- **Kontrak API.** Ini masih memakai `/api/analyze` (multipart) yang ada sekarang. Kontrak final
  `/api/vision/analyze` (base64 + `request_id`) belum ada. Kalau nanti berubah, cukup ubah
  `HttpVisionClient`.

## Menambah field_type / aturan baru

1. Pastikan `field_type` sudah ada di Vision Service (`/api/field-types`, cek dengan `vision:ping`).
2. Tambahkan ke `config('vision.field_types')`.
3. Kalau perlu dibandingkan dengan data IPC, tambahkan aturan di `config('vision.rules')` dan
   case baru di `VisionEvaluator::evaluate()`. Tanpa aturan, hasilnya selalu REVIEW (aman).

## Belum dikerjakan

- Komponen UI yang dipasang langsung di halaman stage (baru hook + contoh di atas).
- Input `shelf_life_months` di form Master Produk / import Excel.
- Otomatis mengisi toggle Conform/Not Conform dari hasil PASS/FAIL (sengaja belum; tetap QC
  yang memutuskan di form).
