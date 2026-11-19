# Kalibrasi kardus dan pendamping COCO11s

7 November 2026. Evaluasi kalibrasi selesai dan threshold kardus 0,65 ditolak setelah hasil test menurun. Konfigurasi demo v4 sekarang memakai pendamping COCO11s dengan threshold kardus 0,50. Pada 117 foto benchmark yang sama, F1 gabungan meningkat **48,0% menjadi 53,2%**, dan ponsel benar meningkat **6 menjadi 12 dari 23 objek**.

## Hasil kalibrasi kardus

Threshold 0,65 dipilih dari hasil validation v4 sebelum memeriksa test dan replay kamera. Evaluasi menerapkan kandidat profil melalui fungsi review yang sama dengan runtime, dalam proses evaluasi terisolasi. Profil dibatasi pada checksum, versi dan taksonomi checkpoint square640 v4 serta gabungan v4/COCO11n. Evaluasi private decision juga menguji batas 0,64 dan 0,65.

| Kardus | Threshold 0,50 | Kandidat 0,65 |
|---|---:|---:|
| Validation benar / salah deteksi | 17 / 23 | 16 / 14 |
| Validation F1 | 37,4% | 39,5% |
| Test benar / salah deteksi | 11 / 8 | 8 / 6 |
| Test F1 | 44,9% | 36,4% |
| F1 gabungan pada benchmark standar | 48,0% | 47,4% |

Validation memuat 51 anotasi kardus dan test memuat 30. Pada test, kehilangan tiga deteksi benar melebihi pengurangan dua salah deteksi. Kandidat berstatus `REJECTED_ON_TEST` dan konfigurasi aktif memakai threshold 0,50. Unit test memastikan kandidat evaluasi terisolasi dari runtime. [Hasil evaluasi](../../evaluations/cardboard-v4-calibration.json), [kandidat profil](../../evaluations/cardboard-v4-threshold-candidate.json).

Replay menggunakan prediksi yang tersimpan dan memeriksa bahwa skor awal dapat direproduksi. Cache validation/test menyimpan prediksi mulai confidence 0,50, sehingga mencakup semua prediksi yang diperlukan untuk kenaikan threshold ini. Latency inference tidak diukur ulang dalam eksperimen threshold.

## Pemilihan model umum

Setelah hasil threshold ditolak dan unduhan foto baru gagal, checkpoint COCO11s yang tersedia lokal diperiksa sebagai pendamping. Pemilihan memakai peningkatan F1 ponsel dan remote pada validation dengan threshold 0,50. Semua 80 label COCO dipertahankan.

| Hasil model umum | COCO11n | COCO11s |
|---|---:|---:|
| Ponsel benar validation | 2 / 19 | 8 / 19 |
| Salah deteksi ponsel validation | 0 | 0 |
| Remote benar validation | 1 / 9 | 2 / 9 |
| Salah deteksi remote validation | 0 | 0 |
| Ponsel benar test | 5 / 18 | 7 / 18 |
| Salah deteksi ponsel test | 3 | 3 |
| Remote benar test | 2 / 9 | 4 / 9 |
| Salah deteksi kulkas test | 1 | 4 |
| mAP50 test seluruh kelas yang memiliki GT | 55,2% | 63,1% |

Validation terdiri dari 43 foto dan test 42 foto. Checkpoint COCO11s dipilih setelah hasil validation, kemudian dievaluasi pada test dan benchmark kamera. Hasil ini mengukur perubahan kapasitas model pendamping. Pengujian ini menggunakan bobot pretrained COCO11s yang tersedia lokal. [Perbandingan model umum dan provenance dataset](../../evaluations/general-coco11s-comparison.json).

## Benchmark gabungan

Benchmark mempertahankan 117 foto, 222 anotasi, manifest, JPEG90, square640, threshold 0,50 dan matching IoU 0,50. Ketiga mode diuji dengan tiga pengulangan untuk setiap foto/profil input pada CPU delapan thread.

Sebagian foto test Broad Waste juga menjadi sumber benchmark, sehingga hasil test dan replay memiliki overlap. Jumlah foto independen tidak dijumlahkan antara kedua laporan tersebut.

| Input standar | v4 + COCO11n | v4 + COCO11s |
|---|---:|---:|
| Precision | 65,6% | 67,8% |
| Recall | 37,8% | 43,7% |
| F1 | 48,0% | 53,2% |
| Objek benar | 84 | 97 |
| Salah deteksi dalam cakupan anotasi | 44 | 46 |
| Ponsel benar / dukungan | 6 / 23 | 12 / 23 |
| Salah deteksi ponsel | 4 | 3 |
| Remote benar / dukungan | 5 / 37 | 9 / 37 |
| Kardus benar / dukungan | 5 / 15 | 5 / 15 |
| Salah deteksi kardus | 5 | 5 |
| Median CPU | 154 ms | 211 ms |

Pada simulasi redup, F1 meningkat 48,4% menjadi 51,6%. Pendamping lebih besar meningkatkan recall bersama tambahan biaya CPU dan sedikit tambahan salah deteksi total. Runtime benchmark memakai Python 3.13.1, Torch 2.14.1 CPU dan Ultralytics 8.4.183. Variasi kondisi eksekusi dapat memengaruhi waktu, dan pemeriksaan singkat lain juga berjalan pada sebagian pengukuran. [Benchmark lengkap](../../evaluations/camera-replay-v1-v4-coco11s-comparison.md), [data per foto](../../evaluations/camera-replay-v1-v4-coco11s-comparison.json).

Foto ponsel biru pengguna diperiksa setelah pemilihan model melalui endpoint demo nyata. Foto tersebut masih diprediksi remote, dengan skor 64,2%, dibanding 87,8% pada pendamping lama. Foto ini menjadi diagnosis terpisah dan belum menjadi data training atau benchmark utama. Hasil agregat yang lebih baik belum menyelesaikan kasus ponsel tersebut.

## Data ponsel dan langkah pengambilan foto

Pencarian sumber publik menghasilkan kandidat metadata, tetapi unduhan asli Wikimedia membalas HTTP 403 dan enam kandidat Flickr membalas HTTP 502. Ada **nol foto training baru** yang berhasil diperoleh pada upaya ini. Setiap kandidat masih memerlukan pemeriksaan lisensi per foto, screening visual dan anotasi bbox. [Audit akuisisi](../../evaluations/phone-rear-v2-acquisition.json).

Tambahan yang diperlukan adalah foto kamera baru dengan izin penggunaan untuk latihan, terutama ponsel tampak belakang dan tiga kamera. Sertakan tampak depan, sudut miring, genggaman tangan dan variasi pencahayaan, serta remote sebagai pembanding. Identitas benda fisik dan sesi pengambilan perlu dicatat agar pembagian train/validation/test menghindari duplikasi adegan. Foto benchmark dan screenshot diagnosis tetap dikelola sebagai evaluasi.

## Mencoba dan memverifikasi

```powershell
.\demo-camera-v4.bat
```

Launcher membuka konfigurasi eksperimen terbaru pada `http://127.0.0.1:8100`, dengan versi `broad-coco-b0f633193403`. Endpoint konfigurasi dan deteksi sudah diperiksa melalui HTTP untuk 88 label gabungan, threshold kardus 0,50, checksum model umum, foto kardus yang bisa dipilih dan pembersihan sesi. Kamera fisik baru belum diuji. [Pemeriksaan demo](../../evaluations/v4-coco11s-demo-verification.json).

Bobot pendamping berada di `artifacts/models/yolo11s.pt`, dengan SHA-256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`. Semua dugaan tetap memerlukan konfirmasi pengguna. Dataset publik, dukungan kelas kecil dan benchmark yang sudah diamati selama pengembangan membuat hasil ini tetap eksploratif.

```powershell
# Pengumpulan prediksi test diperlukan sekali jika cache belum tersedia.
.\.venv-ai-train\Scripts\python.exe scripts/evaluate_cardboard_calibration.py --collect-test
# Replay validation, test dan benchmark melalui kandidat threshold yang terisolasi.
.\.venv-ai\Scripts\python.exe scripts/evaluate_cardboard_calibration.py
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
.\.venv-ai\Scripts\python.exe scripts/benchmark_camera_models.py --v3 artifacts/training/broad-waste-v4-focus-yolo11s/weights/best.pt --coco artifacts/models/yolo11s.pt --waste-name v4 --general-name coco11s --repeats 3 --output evaluations/camera-replay-v1-v4-coco11s-comparison.json
```

Seluruh 250 tes Python lulus. Pengujian baru mencakup batas threshold kandidat, identitas artifact, validasi private decision dan isolasi kandidat yang ditolak dari konfigurasi aktif.
