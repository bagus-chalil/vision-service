# Panduan foto tube (usulan untuk ipc_app) — 2026-11-12

Dasar: baseline 175 foto tube produksi (lihat CLAUDE.md, folder `testdata_private/tube/`).

## Temuan
- Semua 175 foto set ini ber-resolusi 640 px sisi terpanjang (unggahan lama, sebelum ipc_app commit `99a0db3`
  yang menaikkan ke 1600 px). Resolusi TIDAK bisa dibandingkan di set ini -> perlu foto baru untuk mengukur.
- 84/175 foto NO_SIGNAL: kebanyakan tube kecil di foto lebar (crimp hanya ratusan piksel).
- Garis hitam (sensor mark) tercetak di crimp semua tube, di tengah baris EXP; di foto dengan garis hanya 1/128
  lolos OK pada baca bebas. Model bisa menebak digit di bawah garis -> hasil di posisi itu tidak boleh dipercaya.

## Panduan operator (usulan)
1. Dekatkan kamera ke crimp (ujung pipih tube) sampai baris EXP memenuhi ±1/2 lebar frame; jangan foto seluruh tube.
2. Miringkan tube ±15-30 derajat atau geser sudut pandang supaya garis hitam tidak persis menimpa digit; boleh 2 foto
   dengan posisi garis berbeda (halaman uji sudah mendukung 2 foto, hasil per-posisi tidak digabung per karakter).
3. Hindari silau (foil emas/mengkilap): ubah sudut cahaya; pastikan fokus (tap-to-focus pada emboss).
4. Jangan ada klip/jari menutupi baris emboss.
5. Satu tube per foto bila bisa (ada tube lain/kertas di frame menambah noise).

## Yang perlu dilakukan sebelum diterapkan
- Kumpulkan ~50 foto tube baru dengan panduan ini (resolusi 1600 px), ukur ulang dengan
  `tools/baseline_tube_exp.py` + `tools/proto_masked_exp.py`, bandingkan dengan baseline lama.
- Putuskan dengan ipc_app (tim lain): teks panduan di UI + overlay bingkai di kamera.
