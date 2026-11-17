# Issue #8 — persiapan dan kekurangan data

Status 17 November 2026: **implementasi tooling dan pemulihan baseline selesai; penerimaan data baru belum terpenuhi**. Issue harus tetap terbuka sampai data baru, review independen, penggabungan final, dan verifikasi output selesai.

## Hasil nyata

- Branch implementasi dibuat dari main `feb61aa` tanpa menimpa branch pekerjaan sebelumnya.
- Dataset Broad Waste v4 dipulihkan dari backup lokal: 3.893 foto, 2.228 train penuh, 943 train aktif, 975 validation, 690 test.
- Manifest sebelum/sesudah relokasi tetap SHA-256 `846f9a112db657c0352582acaa25aa935eed32ea18ec6a6a626373d790816c8a`. Hanya root absolut daftar split dan path YAML yang direlokasi. File backup tidak dimutasi.
- Verifier dataset memeriksa semua checksum gambar, kesesuaian label, geometri, sampling aktif, kelompok dan pHash antar-split. Semua lulus, dengan toleransi tepi historis DrinkingWaste yang tetap dilaporkan.
- Dua bobot baseline cocok dengan SHA-256 pada `model.json`. Seluruh 117 foto replay juga dipulihkan dan cocok dengan manifest beku.
- Sebanyak 48 kandidat TACO yang belum masuk v4 tersedia dalam antrean review lokal, dengan contact sheet serta keputusan review historis dipertahankan. Kandidat mencakup 8 bbox cardboard, 40 wrapper, 18 paper, dan 18 drink_carton; satu foto bisa memuat beberapa kelas.
- Semua kandidat tersebut masih **REVIEW_REQUIRED_NOT_TRAINING_DATA**, dengan nol foto tambahan yang disetujui. Foto yang pernah diperiksa tidak diklaim sebagai holdout baru.

## Implementasi

`scripts/prepare_tuning_data.py` menyediakan empat perintah:

1. `restore` memverifikasi sumber dan merelokasi dataset tanpa mengubah manifest, anotasi, split atau pilihan train aktif. Output yang sudah ada ditolak.
2. `stage-review` memulihkan kandidat yang belum masuk baseline beserta bbox dan riwayat review; tidak mengubah keputusan lama menjadi persetujuan.
3. `audit` memeriksa checksum/pHash nyata, batas bbox, taksonomi lengkap, bukti review manusia, status izin/atribusi, perlindungan data lama, benturan sesi/kelompok, serta kuota data baru. Gagal kesiapan menghasilkan exit code 2 dan laporan defisit.
4. `build` hanya menerima intake yang lulus audit, mempertahankan train aktif legacy, menambahkan data baru, dan memisahkan test legacy dari holdout baru. Jalankan verifier dataset pada hasil build sebelum training.

Lihat [panduan anotasi dan perintah](TUNING_DATA_ANNOTATION_GUIDE.md), [protokol evaluasi beku](../examples/tuning-evaluation-protocol.json), serta [rencana lima tahap](YOLO_TUNING_PLAN.md).

## Kekurangan yang terukur

| Kelas | Tambahan foto train disetujui / minimum | Validation baru | Holdout baru |
|---|---:|---:|---:|
| cardboard | 0 / 150 | 0 / 50 instance, 0 / 20 kelompok | 0 / 50 instance, 0 / 20 kelompok |
| plastic_wrapper | 0 / 100 | 0 / 50 instance, 0 / 20 kelompok | 0 / 50 instance, 0 / 20 kelompok |
| paper | 0 / 100 | 0 / 50 instance, 0 / 20 kelompok | 0 / 50 instance, 0 / 20 kelompok |
| drink_carton | 0 / 100 | 0 / 50 instance, 0 / 20 kelompok | 0 / 50 instance, 0 / 20 kelompok |

Kelompok benda dan sesi yang terhubung dihitung satu unit independen. Satu foto bisa berisi beberapa kelas; target tidak dapat dijumlahkan untuk menyimpulkan jumlah foto independen. Tidak ada pemenuhan kuota dengan duplikasi, augmentasi foto lama, atau relabel split legacy.

RealWaste tetap berada pada jalur riset terpisah karena perbedaan ketentuan lisensi dan review manusia independen yang masih tertunda. Dataset lama yang dipulihkan tidak membuktikan kelengkapan anotasi baru atau kemampuan di kamera lapangan.

## Environment dan verifikasi

Environment riset yang tersedia memakai Python 3.13.1, Ultralytics 8.4.183, NumPy 2.2.5, dan Pillow 11.2.1. Inference CPU memakai Torch 2.14.1/OpenCV 4.13.0.92. Environment training terpisah memakai Torch 2.9.1+cu128. GPU yang terdeteksi RTX 4070 Laptop 8.188 MiB, driver 610.62; tidak ada training dijalankan pada tahap ini. Ruang bebas saat inventaris sekitar 340 GB.

Laporan yang disimpan:

- [Verifikasi dataset baseline](../evaluations/issue8-baseline-data-verification.json).
- [Inventaris environment dan hash](../evaluations/issue8-environment.json).
- [Kekurangan data baru](../evaluations/issue8-new-data-readiness.json).
- [Replay baseline yang diukur ulang](../evaluations/issue8-baseline-replay.md) dan [pemeriksaan reproduksi](../evaluations/issue8-baseline-reproduction.json).

Replay ulang memakai 117 foto, dua profil input, dan tiga pengulangan pada CPU delapan thread. Semua checksum input dan metrik agregat pada setiap model/profil cocok dengan laporan lama: precision/recall/F1 gabungan standar 67,8%/43,7%/53,2%, dengan 97 TP, 46 FP, dan 125 FN; F1 redup sintetis 51,6%. Waktu CPU standar sekarang median 512,1 ms/p95 538,2 ms, berbeda dari laporan historis. Ini tidak menunjukkan perubahan kualitas atau percepatan; ukur baseline dan kandidat berdampingan pada kondisi sebanding untuk keputusan performa berikutnya.

Tes Windows yang dapat dijalankan lulus; tes deployment yang mengimpor `fcntl` memerlukan Linux. Unit test audit memakai stub fungsi pHash agar dependency riset OpenCV tidak perlu ditambahkan ke lock service/CI. Penghitungan pHash nyata tetap dijalankan pada intake di environment riset. CI Linux pada PR menjadi pemeriksaan suite lengkap. Tidak ada emulator atau perubahan model produksi.

336 tes kompatibel Windows dan Ruff lulus. [CI Linux pada commit kode `6674064`](https://github.com/waste-not-app/waste-not-ai/actions/runs/44117653326) juga lulus, termasuk deployment recovery, pemeriksaan kontrak, audit dependency dan bobot. Lihat checks PR untuk status head terbaru setelah lampiran laporan ini.

## Syarat penutupan

- Review manusia independen terhadap kandidat/anotasi dan foto tambahan dengan izin yang sesuai.
- Data dari benda/sesi baru untuk validation dan holdout sampai dukungan minimum terpenuhi.
- Audit intake lulus, dataset final dibangun, dan verifier output lulus.
- Hasil baseline replay serta CI dicatat. Akurasi kandidat baru baru dinilai pada tahap training berikutnya.

Sampai syarat ini terpenuhi, perubahan ini merupakan fondasi yang dapat ditinjau dan diuji; issue #8 belum selesai penuh.
