# Rencana "AI makin pintar" (Startup AI Cross-Check)

Dibuat 2026-11-12. Catatan keputusan dan rencana; belum ada yang diimplementasikan.

## Fakta dasar
- Qwen (Ollama) tidak punya memori; tiap request dari nol. Memori Claude / CLAUDE.md
  hanyalah teks yang disuntikkan ke prompt, modelnya tidak berubah.
- Model tidak boleh memperbarui dirinya otomatis tiap hari. KPI = nol false PASS;
  satu koreksi salah (DB / operator salah ketik, sering ketemu di eval 2026-11-12)
  bisa merusak akurasi tanpa ketahuan. Setiap perubahan WAJIB diukur ulang di set
  uji tetap sebelum dipakai.

## Opsi (urut dari yang aman)
1. **Koreksi tervalidasi -> prompt/contoh.** Pola salah baca yang terbukti (6 vs 8
   pada foto buram, format penulisan batch) jadi aturan/contoh di prompt. Ukur ulang.
2. **Referensi master data.** Daftar nomor IM / nama produk / format batch valid per
   SKU disuntikkan sebagai kandidat. Tetap hanya flag, tidak pernah koreksi otomatis.
3. **Fine-tune dari foto terkoreksi.** Satu-satunya yang benar-benar "belajar", tapi
   melanggar aturan "tanpa training" saat ini. Hanya bila user setuju dan data sudah
   ribuan.

## Langkah
- Tahap berdampingan 2-4 minggu: AI baca, operator tetap isi manual, selisih tercatat
  di `ipc_logs` (ipc_app).
- Dari selisih, kumpulkan kasus yang TERBUKTI AI salah (dicek mata, bukan hanya
  dibandingkan DB) -> opsi 1 dan 2 -> ukur ulang `tools/evaluate_*.py` /
  `tools/evaluate_vs_db.py`.
- Ukur ulang dengan foto resolusi baru (kamera ipc_app 1600 px) sebelum AI boleh
  menggantikan input manual.

## Temuan eval 2026-11-12 (sampel acak 150 foto/jenis, vs DB)
- label_bulk: 63 OK (42%); 1 kemungkinan false OK (foto 1505, 06/10 terbaca 08/10,
  semua view sepakat, foto buram). View 640 px DIPERTAHANKAN: tanpa view itu foto 863
  lolos salah.
- im_document: 21 OK (14%); 0 false OK yang ketemu. Hambatan = kualitas foto
  (umur simpan tak terbaca di 52/129 LOW).
- DB sering salah/beda format dari foto (mis. "CKB 26X003", "26X015" vs foto
  "26X165"): jangan jadikan DB kunci mutlak.

## PENGINGAT SAAT INTEGRASI KE ipc_app (jangan lupa)
- AI hanya ADVISORY: operator tetap mengisi manual; jangan auto-isi toggle
  Conform/Not Conform; jangan longgarkan `VisionEvaluator`.
- Kerjakan di branch `development` dulu, bukan `production`.
- Daftarkan field_type di `config/vision.php` HANYA setelah `php artisan vision:ping`
  menunjukkan field_type itu ada di server.
- Catat selisih AI vs input operator di `ipc_logs` (bahan "belajar", lihat Langkah).
- Kumpulkan kasus AI-salah yang DICEK MATA, bukan hanya dibanding DB (DB sering salah).
- Ukur ulang dengan foto resolusi baru (kamera 1600 px) sebelum AI boleh
  menggantikan input manual; ukur ulang tiap prompt/aturan diubah.
- Blokir infra: VPS belum bisa menjangkau Ollama (.26 hanya 127.0.0.1) dan masih
  kode lama; deploy vision-service terbaru dulu.
- Jenis kemasan (tube/non-tube): kolom master_products + Qwen `packaging_type`
  sebagai pengecek; hanya coding_primary yang bisa tube.
