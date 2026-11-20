# Ukuran input dan biaya CPU — issue #11

Eksperimen penerimaan memakai checkpoint yang dibekukan dari issue #10 dan validation yang sudah direview. [Pilot #11](ISSUE_11_RESOLUTION_PROGRESS.md) mencatat hasil sementara dari bobot rilis dan validation legacy.

## Audit jalur gambar

Audit berikut berdasarkan source lokal, bukan pengujian kamera/APK yang sedang dipasang:

| Jalur | Pemrosesan sebelum YOLO |
|---|---|
| Demo browser (`web/demo.js`) | Capture/upload dinormalisasi ke canvas, sisi panjang maksimal 1600; JPEG kualitas 90 |
| Android (`ScanCamera.kt`, `ScanPhotoCache.kt`) | Target CameraX 1280×960 dengan fallback; decoder memakai inSampleSize pangkat dua sampai sisi panjang ≤1600; rotasi diterapkan; disimpan sebagai JPEG kualitas 85 |
| Gateway Go (`modules/ai/service.go`) | Decode dengan auto-orientation lalu encode JPEG kualitas 90, tanpa resize tambahan pada jalur yang diaudit; diteruskan ke `/internal/v1/detect` |
| Worker Python (`media.py`) | JPEG/PNG ≤5 MiB, sisi ≤4096, total ≤12 juta piksel; EXIF transpose, RGB dan pembersihan metadata; tanpa resize tambahan |
| Detektor | Ultralytics letterbox dengan input square640/768/960; bbox `xyxyn` dikembalikan relatif terhadap gambar decode/orientasi asli |

Sumber Android/Go berada pada sibling workspace `waste-not-android` dan `waste-not-backend`. Tidak ada perubahan dibuat pada kedua repo tersebut.

Replay mempertahankan JPEG90 demo dan simulasi redup0.55, serta menambahkan pendekatan Android JPEG85→gateway JPEG90. Sampling/codec Pillow bukan implementasi BitmapFactory/encoder Go, sehingga profil Android tidak mengklaim kesamaan piksel dengan kamera nyata. Capture nyata yang telah direview tetap dibutuhkan pada evaluasi akhir.

Setiap input mencatat ukuran file gambar sumber yang tersedia, ukuran terkirim dan checksum byte yang dipakai. Profil yang dibandingkan pada resolusi berbeda memakai byte identik. Menaikkan input YOLO tidak memulihkan detail yang telah hilang akibat sampling/kompresi sebelumnya.

## Parameter runtime dan identitas

`YoloDetector(..., input_size=640|768|960)` dan `load_detector(..., input_size=...)` menyediakan profil eksperimen. Setiap profil memakai instance tersendiri; input size tidak dapat diganti melalui setter tanpa membuat instance/identitas baru. Semua caller worker/demo/CLI lama tetap default640. `model.json` serta konfigurasi produksi tidak diubah.

Identitas bundle square640 yang dirilis tetap `broad-coco-b0f633193403`. Ukuran lain mendapat versi komponen berbeda dan input size masuk identitas bundle, sehingga cache/kalibrasi 640 tidak dipakai sebagai profil 768/960. Label gabungan tetap 88 dan skema respons tetap sama. Prediksi wajib merujuk ukuran gambar decode; SDK yang melaporkan orig_shape berbeda ditolak. Wrapper memakai koordinat normalized-original SDK, tanpa melakukan inverse-letterbox kedua kali.

Skrip replay historis `benchmark_camera_models.py` menerima `--input-size` dan `--warmup`. Replay tersebut tetap pemeriksaan regresi legacy dan tidak dipakai untuk memilih pemenang validation baru.

## Prasyarat kandidat dan dataset

`benchmark_resolutions.py` menerima receipt issue #10 untuk masing-masing komponen:

- `VALIDATION_STABLE_PENDING_RESOLUTION_REVIEW` untuk kandidat yang lulus tiga seed; checksum file harus sama dengan kandidat terpilih.
- `KEEP_BASELINE` atau `KEEP_BASELINE_UNSTABLE` untuk mempertahankan komponen awal terdaftar; checksum harus cocok dengan `model.json`.

Receipt harus menyatakan holdout belum dibuka dan cocok dengan seluruh snapshot dataset/input awal. Receipt stability sekarang membawa snapshot/config tersebut agar dapat diverifikasi tahap berikutnya.

Dataset snapshot memeriksa train/validation dan metadata perlindungan split seperti issue #10; tidak membuka test.txt, gambar atau label test. Benchmark menyusun manifest **VALIDATION_ONLY** dari kedua validation set. Foto bersama tidak dijumlahkan dua kali; konflik anotasi pada scope yang sama ditolak. Kelompok, sesi, hash identik dan pHash lintas split kedua model diperiksa kembali. Toleransi tepi DrinkingWaste legacy dinormalisasi hanya pada GT salinan untuk scoring; dataset asal tidak dimutasi.

## Menjalankan eksperimen

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONIOENCODING = 'utf-8'
python scripts/benchmark_resolutions.py plan --output evaluations/issue11-resolution-plan.json

# Setelah review/build/verifier dan seleksi issue #10 benar-benar lengkap:
python scripts/benchmark_resolutions.py run --broad-receipt artifacts/training/broad-stability.json --general-receipt artifacts/training/general-stability.json --broad-weights PATH_FROZEN_BROAD_PT --general-weights PATH_FROZEN_GENERAL_PT --output artifacts/resolution/experiment-01
```

Dataset default adalah output baru issue #8/#9. Gunakan `--broad-data` dan `--general-data` bila namanya berbeda. Jika pilihan komponen mempertahankan baseline, gunakan JSON keputusan yang sesuai sebagai receipt. Selalu gunakan output baru; hasil lama tidak ditimpa. File stage disimpan sebelum pengukuran AP/HTTP sehingga kegagalan lanjutan tidak menghilangkan hasil stage yang telah selesai.

Urutan pengukuran:

1. Periksa hash candidate/receipt/dataset sebelum model dimuat. Inference CPU dengan thread tetap (default8), batch1.
2. Muat profil640 dan768, keluarkan sedikitnya lima warmup per profil dari timer, lalu jalankan sedikitnya tiga pengulangan. Urutan profil dirotasi per gambar/profil input/pengulangan.
3. Laporkan precision/recall/F1 per kelas, bucket ukuran objek, p50/p95 waktu `detect`, native RSS serta AP validation tiap komponen. Persiapan JPEG, loading model, AP dan HTTP berada di luar timer detektor.
4. 960 hanya diuji bila768 lolos pemeriksaan kualitas/dukungan/CPU. Stage960 mengukur ulang640 dan768 bersama960 agar kontrol waktu tetap sebanding pada stage yang sama.
5. AP Broad/general dilaporkan sendiri-sendiri dengan YAML train/val tanpa test, conf0.001, IoU NMS0.7, batch1. AP dua model tidak dirata-ratakan untuk mengklaim mAP gabungan.
6. Verifikasi kembali semua snapshot/hash setelah pengukuran. Cache terikat manifest, byte input, model/profil, environment dan protokol.

Bucket objek memakai area normalized GT yang tetap lintas resolusi: kecil <0.01, medium <0.1, besar ≥0.1. TP/FN mengikuti ukuran GT; unmatched FP mengikuti ukuran prediksi. Ini definisi eksperimen dan bukan bucket COCO berbasis luas piksel. Simulasi redup/profil codec tidak menambah jumlah foto independen.

## Memori dan HTTP

Memori diukur sebagai native RSS/peak RSS proses, memakai Windows GetProcessMemoryInfo atau `/proc/self/status` pada Linux. Semua profil yang dibandingkan resident pada proses yang sama; angka tersebut bukan biaya memori terisolasi masing-masing model/resolusi. Platform tanpa provider ditandai unavailable. Pengukuran isolated-process diperlukan bila batas memori per profil menjadi syarat deployment.

HTTP opsional dan diukur terpisah. Sediakan endpoint worker khusus tiap profil dengan weights/identitas yang sama; worker default640 tidak dapat dianggap mewakili768. Endpoint dapat memakai instance detektor yang diinjeksikan ke `worker.create_app`, dengan `load_detector(..., input_size=...)`. Tidak ada worker baru dijalankan otomatis oleh tool ini.

```powershell
python scripts/benchmark_resolutions.py run ... --http-endpoint 640=http://127.0.0.1:8100/internal/v1/detect --http-endpoint 768=http://127.0.0.1:8101/internal/v1/detect
```

Token diambil dari `AI_SERVICE_TOKEN`, tidak disimpan pada report; redirect dimatikan dan endpoint remote wajib HTTPS. Ukuran gambar, model version dan model SHA response diverifikasi. Bila960 ikut terukur, semua endpoint yang diperlukan harus disediakan. Tanpa endpoint, HTTP berstatus `NOT_MEASURED` dan tidak diberi angka buatan. Report HTTP adalah waktu upload/server/response termasuk inference, bukan perkiraan overhead hasil pengurangan dua median. Biaya HTTP tidak dipakai menggantikan budget CPU lokal.

## Keputusan dan status saat ini

Ukuran lebih besar hanya diteruskan bila macro-F1 enam kelas fokus membaik, micro-F1 keseluruhan tidak menurun, dukungan tiap kelas fokus ≥50, kelas/bucket dengan dukungan ≥50 tidak turun >2 poin F1, dan p95 CPU ≤1.25× kontrol640 pada stage/input yang sama. Profil redup/Android juga diperiksa untuk regresi dan budget. Pilihan tetap640 bila syarat tidak lulus.

Report selalu `promoted=false`, `holdout_opened=false`, dan input aktif tetap640. Pilihan eksperimen belum boleh diaktifkan sebelum kalibrasi serta evaluasi akhir issue #12. Adaptasi fine-tuning768 bila diperlukan merupakan eksperimen training terpisah dengan lineage checkpoint baru, bukan bagian dari perbandingan inference-only ini.

[Preflight eksperimen penerimaan](../evaluations/issue11-resolution-plan.json) menunggu receipt yang cocok dengan dataset hasil review. Metrik pilot tersedia di [laporan JSON](../evaluations/issue11-resolution-pilot.json).
