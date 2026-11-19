# Kandidat broad v3: hasil training dan perbandingan

7 November 2026. Kandidat YOLO11s telah selesai dilatih dan dievaluasi. Pada **test yang sama, 677 gambar dan sepuluh kelas**, mAP50 naik dari **0,2828 menjadi 0,3518** (+6,90 poin persentase); mAP50:95 naik dari **0,1875 menjadi 0,2278**. Ini hasil eksploratif publik, bukan akurasi lapangan. Model/default launcher lama tidak diganti.

## Eksperimen dan data

[Dataset v3 batch kedua](DATASET_V3.md) memiliki 3.798 gambar: 3.699 baseline ditambah 99 foto yang lolos screening awal dari 147 unduhan. Empat puluh delapan foto ditunda. Flickr membalas HTTP 429 pada akuisisi lanjutan, sehingga request dihentikan. Semua data mentah, bobot dan log tetap lokal serta diabaikan Git.

Training menggunakan 901 gambar unik yang dipilih dari 2.173 gambar train, 948 val, dan 677 test. Seluruh contoh kelas minoritas dipertahankan; gambar mayoritas dikurangi tanpa menghapus bbox lain dalam satu gambar. Target 300 instance training per kelas belum tercapai. Penambahan foto dan penyeimbangan gambar terjadi bersama-sama, sehingga eksperimen ini tidak memisahkan pengaruh masing-masing intervensi.

Konfigurasi: YOLO11s pretrained COCO, transfer baris bottle/cup yang sama dengan v2, freeze 10, AdamW lr 0,0003, seed 42, square 640, batch 16, maksimum 30 epoch, patience 10. Training selesai **28 epoch**, best epoch **18** berdasarkan mAP50:95 validation; waktu CSV terakhir 421,016 detik. GPU: RTX 4070 Laptop. Angka epoch diambil dari CSV training, bukan callback progres terakhir.

- Bobot baru: `artifacts/training/broad-waste-v3-batch2-yolo11s/weights/best.pt`.
- SHA-256: `cd8db14b189a8195278694aac07bbf4ae972467c663c1f6aebb72fc5b7bc71f0`.
- Pembanding broad v2 SHA-256: `a3f34225287ad8dd01fe9082488db3f9c40dea5b7a90b8a19f1e6bc77d69a069`; hash setelah eksperimen masih sama.
- Manifest dataset: `78e78d1fd37806b5179cab2434602994755755fbd396dbcd3acd382802b20022`.

## Perbandingan yang setara

AP test kedua checkpoint dihitung pada dataset/split/taksonomi yang sama, conf awal 0,001, NMS IoU 0,70, max_det 50, square 640. Jangan membandingkan langsung agregat ini dengan angka v2 tanggal 20 Oktober yang memakai test 656 gambar dan sembilan kelas dengan GT. [Perbandingan lengkap](../../evaluations/broad-waste-v3-batch2-yolo11s-20261107-comparison.json), [report kandidat](../../evaluations/broad-waste-v3-batch2-yolo11s-20261107.json).

| Test identik | Broad v2 | Kandidat v3 |
|---|---:|---:|
| mAP50 | 28,28% | 35,18% |
| mAP50:95 | 18,75% | 22,78% |

Precision/recall operasional di bawah dihitung terpisah pada 948 gambar validation, confidence 0,50, matching IoU 0,50, NMS 0,70 dan square 640 yang identik untuk kedua checkpoint.

| Kelas | Precision v2 | Precision v3 | Recall v2 | Recall v3 | Dukungan GT val |
|---|---:|---:|---:|---:|---:|
| bottle | 79,07% | 87,48% | 81,48% | 74,82% | 691 |
| can | 92,16% | 82,95% | 51,65% | 58,79% | 182 |
| cup | 76,47% | 78,74% | 55,52% | 58,01% | 281 |
| plastic_bag | Tidak ada prediksi | 0% | 0% | 0% | 8 |
| paper | Tidak ada prediksi | 0% | 0% | 0% | 4 |
| cardboard | 50,00% | 33,33% | 11,11% | 22,22% | 9 |
| drink_carton | Tidak ada prediksi | Tidak ada prediksi | 0% | 0% | 2 |
| plastic_wrapper | 0% | 12,50% | 0% | 10,00% | 10 |
| food_container | Tidak ada prediksi | 0% | 0% | 0% | 7 |
| bottle_cap | Tidak ada prediksi | 22,22% | 0% | 25,00% | 8 |

Precision botol membaik dengan penurunan recall; recall kaleng naik dengan penurunan precision; kedua ukuran cup meningkat. Dukungan kelas langka terlalu sedikit untuk kesimpulan kuat dan banyak yang tetap tidak menghasilkan TP pada cutoff 0,50. Precision tanpa prediksi disimpan sebagai null, bukan dianggap 100%.

Gate kegunaan kelas umum dari eksperimen sebelumnya tetap digunakan tanpa penyesuaian setelah melihat hasil: bottle P≥0,85/R≥0,60 **lulus**; can P≥0,80/R≥0,65 **gagal pada recall**; cup P≥0,80/R≥0,40 **gagal pada precision**. Gate ini bukan release gate. Tidak ada promosi otomatis, perubahan threshold, atau klaim bahwa sepuluh kelas sudah andal. Semua kelas broad masih memakai profil `broad-experimental-01` dan memerlukan review/material confirmation.

## Mencoba kandidat

Dari root repo:

```powershell
.\demo-camera-v3.bat
```

Launcher ini secara eksplisit memilih checkpoint eksperimen v3 dan square 640. Setelah laporan kamera tidak mengenali ponsel, launcher v3 juga menjalankan model COCO pendamping melalui `--general-model`; [rincian cakupan kamera](CAMERA_COVERAGE.md). Angka evaluasi di dokumen ini hanya milik checkpoint v3 tunggal, bukan hasil gabungan. Untuk menguji v3 tunggal gunakan CLI `demo --model .../best.pt --input-shape square` tanpa flag pendamping dan tanpa `AI_GENERAL_MODEL_PATH`. `demo-camera-broad.bat` tetap memilih v2; launcher/model default lain tetap tersedia. Kamera fisik belum diuji pada eksperimen ini.

Reproduksi perbandingan AP pembanding terhadap report kandidat yang sudah selesai:

```powershell
.\.venv-ai-train\Scripts\python.exe scripts/compare_broad_candidate.py --report evaluations/broad-waste-v3-batch2-yolo11s-20261107.json --previous artifacts/training/broad-waste-v2-yolo11s/weights/best.pt
```

Script memeriksa hash manifest, daftar split dan bobot pembanding sebelum mengevaluasi. Reproduksi training ada pada [catatan dataset](DATASET_V3.md); gunakan nama run baru karena script menolak menimpa run lama.

## Verifikasi dan batas

204 tes Python lulus; 34 artifact kontrak cocok; 44/44 skenario rekomendasi fixture lulus tanpa panggilan Gemini. Dataset 3.798 gambar lolos pemeriksaan hash/label/split dan pemeriksaan pHash/kelompok lintas split. Inferensi nyata melalui CLI CPU pada cup validation `COCO2017-42735` berhasil mendeteksi cup (confidence 0,9194), memakai checksum baru/square 640 dan tetap UNCERTAIN dengan konfirmasi material. Ini smoke kompatibilitas runtime, bukan pengukuran akurasi lapangan atau latency stabil; timing panggilan pertama mencakup warm-up.

COCO train2017 pernah dilihat bobot pretrained, test Drinking Waste lama sudah pernah diperiksa, dan kelompok nama file/pHash merupakan proksi. Screening tambahan dilakukan Codex pada contact sheet, bukan sertifikasi ahli. Belum tersedia holdout kamera lokal dengan benda/sesi independen dan negatif nyata. Langkah berikutnya adalah memperbanyak data kelas yang masih lemah dan menguji kamera nyata sebelum memilih checkpoint rilis.
