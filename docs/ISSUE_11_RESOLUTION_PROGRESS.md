# Issue #11 — implementasi resolusi dan benchmark

Status 18 November 2026: **kode tersedia; eksperimen nyata menunggu review data dan hasil issue #10**. Review tetap dijadwalkan di akhir sesuai arahan pengguna. PR ditumpuk di atas PR #15.

Yang sudah diimplementasikan:

- Input640/768/960 tervalidasi pada factory detektor; default dan identitas bundle640 lama tetap dipertahankan. Profil resolusi berbeda memiliki identitas/cache terpisah dan input size tidak diganti melalui setter.
- Audit resize/rotasi/kompresi demo, Android, gateway Go dan worker Python berdasarkan source lokal. Profil Android ditambahkan sebagai pendekatan codec terpisah, dengan batas interpretasinya.
- Runner memerlukan receipt kandidat beku dan snapshot dataset yang cocok. Manifest benchmark hanya memuat validation; foto bersama direkonsiliasi tanpa double count dan kebocoran lintas dataset/sesi diperiksa.
- Pengukuran CPU batch1 dengan warmup≥5, pengulangan≥3 serta urutan profil diselang-seling. 960 hanya dijalankan setelah768 layak; kontrol640/768 diukur ulang pada stage960.
- Laporan kualitas per kelas/ukuran objek, AP tiap komponen, p50/p95 detektor, native memori proses dan HTTP opsional yang terpisah. Cache mengikat input/model/shape/environment; snapshot diperiksa lagi pada akhir run.
- Gate kualitas, dukungan, regresi dan budget p95 1.25× kontrol640; tidak ada aktivasi atau promosi otomatis.

Tes mencakup bbox pada gambar landscape/portrait serta rotasi EXIF, penolakan orig_shape yang salah, skema wire88 label, identitas bundle640 yang tetap sama, invalid input size, pemisahan warmup, interleaving pada byte identik, pencatatan ukuran sumber/terkirim, bucket GT yang melintasi batas ukuran prediksi, receipt/input yang berubah, shared-photo annotation conflicts, kebocoran sesi lintas model, 960 bersyarat, sparse/NaN/regresi codec, pengukuran native memori dan konteks/auth HTTP.

Penerimaan yang masih menunggu:

1. Review/build/verifier data #8/#9 dan keputusan kandidat/seed nyata #10.
2. Perbandingan640/768 pada kandidat yang sudah dibekukan; pengukuran960 bila lolos gate.
3. Hasil nyata AP/kualitas per kelas/ukuran/kondisi, latency dan memori pada CPU yang sebanding.
4. HTTP dari endpoint profil yang sesuai dan verifikasi memakai capture nyata; pendekatan codec bukan bukti kesamaan hasil kamera.
5. Pilihan ukuran berdasarkan hasil tersebut, kalibrasi dan evaluasi akhir sebelum aktivasi pada issue #12.

Tidak ada training, inference model nyata, emulator, worker eksperimen atau deployment dijalankan pada tahap implementasi ini. `model.json`, konfigurasi worker aktif dan input640 tetap dipertahankan. Issue #11 tetap terbuka sampai penerimaan nyata selesai.

Lihat [panduan benchmark](RESOLUTION_EXPERIMENT_GUIDE.md) dan [rencana preflight](../evaluations/issue11-resolution-plan.json).
