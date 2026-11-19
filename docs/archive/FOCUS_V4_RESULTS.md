# Hasil training fokus kardus dan ponsel

[Pembaruan setelah evaluasi kalibrasi](V4_CALIBRATION_AND_COCO11S.md) mengganti pendamping pada launcher v4 menjadi COCO11s dengan port 8100. F1 benchmark gabungan menjadi 53,2%. Dokumen ini mencatat eksperimen training sebelumnya dengan pendamping COCO11n.

7 November 2026. Training dan benchmark ulang selesai. Gabungan Broad Waste v4 dengan COCO awal memperoleh F1 **48,0%**, dibanding **43,0%** pada gabungan v3 sebelumnya. Kardus yang terdeteksi benar bertambah dari **2 menjadi 5 dari 15 objek**. Dua percobaan fine-tuning model umum belum meningkatkan deteksi ponsel dan remote pada validation, sehingga komponen umum yang dipakai adalah checkpoint COCO awal.

## Benchmark yang sama

Benchmark menggunakan 117 foto publik, 222 anotasi, square 640, JPEG90, confidence review 0,50 dan matching IoU 0,50. Manifest dan input tetap identik dengan benchmark sebelumnya. Seluruh 234 pasangan foto/profil COCO memberikan skor yang identik pada kedua pengujian. [Laporan v3](../../evaluations/camera-replay-v1-comparison.md), [laporan v4](../../evaluations/camera-replay-v1-v4-focus-comparison.md).

| Gabungan pada input standar | v3 + COCO | v4 + COCO |
|---|---:|---:|
| Precision | 60,7% | 65,6% |
| Recall | 33,3% | 37,8% |
| F1 | 43,0% | 48,0% |
| Objek benar | 74 | 84 |
| Salah deteksi dalam cakupan anotasi | 48 | 44 |
| Objek terlewat | 148 | 138 |
| Kardus benar / dukungan | 2 / 15 | 5 / 15 |
| Salah deteksi kardus | 1 | 5 |
| Precision kardus | 66,7% | 50,0% |
| Ponsel benar / dukungan | 6 / 23 | 6 / 23 |
| Remote benar / dukungan | 5 / 37 | 5 / 37 |
| Pasangan label bertentangan | 14 | 18 |

Pada simulasi redup, F1 gabungan meningkat **43,8% menjadi 48,4%**. Hasil agregat meningkat bersama bertambahnya salah deteksi kardus dan hipotesis yang bertentangan. Recall kardus gabungan masih 33,3%. Kamera fisik baru, kondisi cahaya nyata dan kelas lain memerlukan pengujian lanjutan.

Median waktu CPU pada pengujian v4 adalah 154 ms untuk gabungan, dengan p95 203 ms. Kontrol COCO yang identik juga berubah dari median 87 ms pada pengujian sebelumnya menjadi 49 ms pada pengujian ini. Perubahan waktu tersebut menunjukkan variasi kondisi eksekusi. Angka ini belum membuktikan percepatan dari training. Runtime benchmark memakai Python 3.13.1, Torch 2.14.1 CPU, Ultralytics 8.4.183 dan delapan thread.

## Training kardus

Dataset `broad-waste-v4` mempunyai 943 foto train aktif, 975 validation dan 690 test. Kardus train aktif bertambah dari 29 menjadi 93 anotasi. Training melanjutkan checkpoint v3 dengan freeze 10, AdamW lr 0,0001, batch 16, square 640, seed 42 dan batas 30 epoch. Early stopping selesai setelah **22 epoch**, checkpoint terbaik pada **epoch 12**, dengan waktu training CSV 319 detik. GPU yang dipakai adalah RTX 4070 Laptop.

Pada 690 foto test yang identik bagi kedua checkpoint, mAP50 meningkat **33,53% menjadi 36,50%**, sedangkan mAP50:95 meningkat **21,87% menjadi 24,46%**. Dataset test ini mencakup tambahan foto v4, sehingga angkanya perlu dibandingkan dengan evaluasi v3 pada dataset v4 yang sama. [Perbandingan test dan validation](../../evaluations/broad-waste-v4-focus-yolo11s-20261107-comparison.json).

Pada validation, kardus benar meningkat 7 menjadi 17 dari 51 anotasi dan F1 kardus meningkat 22,6% menjadi 37,4%. Salah deteksi kardus meningkat 4 menjadi 23. Precision cup validation juga menurun 78,7% menjadi 72,0%, sementara precision dan recall kaleng meningkat. Kandidat tetap berstatus eksperimen.

Bobot tersedia di `artifacts/training/broad-waste-v4-focus-yolo11s/weights/best.pt`, dengan SHA-256 `3fc8b7a0baf192b6a88554ce83b24c3ac4ffbd4551e5ee2109a81b68a608462f`.

## Percobaan ponsel

Paket `general-confusion-v1` memakai 535 foto train, 43 validation dan 42 test dengan 80 label COCO asli. Semua anotasi COCO pada foto terpilih dipertahankan. Tambahan mencakup 54 foto ponsel Open Images serta 14 contoh negatif kardus. Riwayat COCO train pada pretraining dan keterbatasan screening dijelaskan dalam [catatan dataset](DATASET_V4_FOCUS.md).

| Percobaan | Freeze | Laju belajar | Epoch selesai / terbaik | Ponsel benar pada validation | Remote benar pada validation |
|---|---:|---:|---:|---:|---:|
| COCO awal | — | — | — | 2 / 19 | 1 / 9 |
| Fine-tuning awal | 10 | 0,0001 | 14 / 4 | 2 / 19 | 1 / 9 |
| Kepala deteksi | 23 | 0,00003 | 11 / 1 | 2 / 19 | 1 / 9 |

Percobaan kedua memakai warmup bias 0,00003 dan warmup satu epoch untuk membatasi perubahan fitur. Percobaan ini dipicu hasil validation percobaan pertama. Aturan pemilihan mensyaratkan peningkatan F1 ponsel atau remote pada confidence 0,50, tanpa penurunan salah satunya. Kedua kandidat belum memenuhi syarat tersebut. Checkpoint dipilih dari validation, kemudian konfigurasi gabungan terpilih menjalani benchmark kamera. [Keputusan pemilihan](../../evaluations/focus-v4-training-selection.json), [percobaan awal](../../evaluations/general-confusion-v1-yolo11n-comparison.json), [percobaan kepala deteksi](../../evaluations/general-confusion-v1-head-yolo11n-comparison.json).

Foto ponsel biru pengguna diperiksa secara lokal sebagai diagnosis terpisah setelah pemilihan checkpoint. COCO awal dan gabungan v4 masih memberi label remote dengan skor 87,8%. Kandidat fine-tuning awal dan kandidat kepala deteksi juga memberi label remote dengan skor 67,1% dan 85,8%. Foto ini belum masuk dataset training atau benchmark utama. Foto ponsel modern tampak belakang, terutama tiga kamera dan variasi sudut genggaman, masih menjadi kekurangan data yang relevan.

## Mencoba dan reproduksi

```powershell
.\demo-camera-v4.bat
```

Launcher memilih gabungan v4 dengan COCO awal pada `http://127.0.0.1:8099`. Versi gabungannya adalah `broad-coco-0b06663cfd8d`. Semua dugaan tetap memerlukan konfirmasi pengguna.

Pemeriksaan melalui HTTP demo nyata lulus untuk konfigurasi 88 label, identitas checkpoint, upload foto kardus publik, hasil kardus yang dapat dipilih dan pembersihan sesi. Pengujian ini memakai foto dari benchmark. Pengambilan gambar langsung dari kamera fisik belum diuji. [Hasil pemeriksaan demo](../../evaluations/focus-v4-demo-verification.json).

```powershell
.\.venv-ai-train\Scripts\python.exe scripts/train_broad_waste.py --data data/local/broad-waste-v4 --name broad-waste-v4-focus-yolo11s --initial-model artifacts/training/broad-waste-v3-batch2-yolo11s/weights/best.pt --compare-model artifacts/training/broad-waste-v3-batch2-yolo11s/weights/best.pt --freeze 10 --lr0 0.0001 --epochs 30
.\.venv-ai-train\Scripts\python.exe scripts/train_general_focus.py
.\.venv-ai-train\Scripts\python.exe scripts/train_general_focus.py --name general-confusion-v1-head-yolo11n --freeze 23 --lr0 0.00003 --warmup-bias-lr 0.00003 --warmup-epochs 1
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
.\.venv-ai\Scripts\python.exe scripts/benchmark_camera_models.py --v3 artifacts/training/broad-waste-v4-focus-yolo11s/weights/best.pt --waste-name v4 --general-name coco --repeats 3 --output evaluations/camera-replay-v1-v4-focus-comparison.json
```

Gunakan nama run baru untuk mengulang training. Parameter lengkap, results CSV, checkpoint, notices dan log tersimpan lokal di `artifacts/training/`. Training memakai Torch 2.9.1 CUDA 12.8. Pengujian Python mencakup 246 tes yang lulus, termasuk penjagaan taksonomi dan split pada trainer model umum. Sumber publik, screening Codex, ukuran kelas kecil dan penggunaan benchmark publik yang sudah diamati membuat hasil ini tetap eksploratif.
