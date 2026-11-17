# Eksperimen fine-tuning terkontrol — issue #10

Trainer Broad Waste dan general sekarang berbagi runner yang memakai **validation saja**. Dataset review issue #8/#9 belum lengkap dan review manusia ditunda ke akhir sesuai arahan pengguna, sehingga tidak ada training nyata yang dijalankan pada tahap implementasi ini. Model produksi tetap memakai v4 + COCO11s.

## Prasyarat dan perlindungan holdout

Gunakan dataset final yang dibangun setelah review melalui preparer issue #8/#9. Builder menyimpan `readiness.json` dengan status lulus, checksum manifest dataset dan checksum intake. Setelah build, jalankan verifier dataset seperti panduan data. Output lama yang belum memiliki bukti terikat checksum perlu dibangun ulang ke output baru; jangan mengisi status lulus secara manual.

Runner memeriksa taksonomi, daftar train/val, hash gambar dan label aktif, kesesuaian bbox, kelompok, serta pHash lintas split. Manifest split test hanya dipakai untuk pemeriksaan identitas/duplikasi metadata; **test.txt, gambar test dan label test tidak dibuka**. Hash manifest membekukan identitas keseluruhan dataset.

Untuk Ultralytics, runner membuat `selection-data.yaml` tersendiri yang hanya memuat train dan val. Training, final validation internal trainer, AP serta metrik operasional semuanya memakai file tersebut. Skrip ini tidak menyediakan mode evaluasi test. Evaluasi holdout kandidat beku dikerjakan terpisah pada issue #12.

Bobot awal Broad memakai checksum v4 dari `model.json`; general memakai checksum COCO11s. File lokal harus tersedia dan cocok sebelum import model. Taksonomi checkpoint dan jumlah layer freeze diperiksa sebelum training. Nama run harus aman dan belum pernah dipakai; tidak ada overwrite/resume training otomatis.

## Parameter dan pencatatan

Parameter CLI mencakup `--imgsz`, `--batch`, `--workers`, `--patience`, `--seed`, `--freeze`, `--lr0`, `--lrf`, `--nbs`, `--warmup-epochs`, `--warmup-bias-lr`, `--close-mosaic`, `--mosaic`, `--degrees`, `--scale`, HSV, translate, dan probabilitas flip. Nilai nonfinite/rentang salah ditolak; ukuran input harus kelipatan stride 32. Eksperimen mendukung CPU atau satu indeks CUDA, bukan distributed training.

Kontrol awal: square640, batch16, nominal batch64, AdamW, lr0 0.0001, cosine LR/lrf 0.01, maksimal 30 epoch, patience10, seed42, freeze10. Augmentasi kontrol mempertahankan mosaic0.5, close-mosaic6, rotasi15, scale0.35, HSV 0.015/0.7/0.4, tanpa flip vertikal. Profil ringan opsional memakai mosaic0.2, rotasi5, scale0.2. Blur/JPEG khusus tidak ditambahkan sebelum kebutuhan dari data nyata dinilai; transform default yang aktif dan versi package terkait dicatat saat training.

Setiap run menyimpan config lengkap, hash input/bobot, bukti checkpoint, kurva `results.csv`, progress dan `epochs.jsonl`, pipeline augmentasi, environment/perangkat, cache prediksi validation dan laporan per komponen. Effective batch dicatat dari batch serta gradient accumulation yang benar-benar dipakai, termasuk perubahan warmup. Jika trainer mengubah batch, kandidat tidak boleh dibandingkan sebelum kontrol yang sebanding dijalankan ulang.

`best_epoch` berasal dari callback `on_model_save` ketika `trainer.fitness == trainer.best_fitness`, sesuai perilaku versi Ultralytics 8.4.183 yang dipatok. Hash checkpoint sebelum optimizer stripping dan hash final dipisahkan. Puncak satu kolom CSV tidak dipakai untuk menebak epoch `best.pt`.

## Urutan B0–B3

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONIOENCODING = 'utf-8'
python scripts/manage_tuning_experiments.py plan --output evaluations/issue10-experiment-plan.json
```

Perintah `plan` tidak melatih model atau mengimpor Torch/Ultralytics. Run yang prasyaratnya belum tersedia dicatat `DEFERRED_DATA_OR_BASELINE`; ini bukan hasil eksperimen. Ada delapan entri (B0–B3 untuk dua model), dengan anggaran awal enam training run karena B0 hanya evaluasi.

Setelah review, build dan verifier selesai, contoh untuk Broad:

```powershell
python scripts/train_broad_waste.py --stage B0 --name issue10-broad-b0-seed42 --evaluate-only artifacts/verified-model-bundle/broad-waste-v4-focus-yolo11s.pt
python scripts/train_broad_waste.py --stage B1 --name issue10-broad-b1-seed42 --freeze 10
python scripts/train_broad_waste.py --stage B2 --name issue10-broad-b2-seed42 --freeze 0

python scripts/manage_tuning_experiments.py choose --report artifacts/training/issue10-broad-b1-seed42/report.json --report artifacts/training/issue10-broad-b2-seed42/report.json --output artifacts/training/broad-b1-b2-selection.json

$chosenExperiment = Get-Content artifacts/training/broad-b1-b2-selection.json -Raw | ConvertFrom-Json
python scripts/train_broad_waste.py --stage B3 --name issue10-broad-b3-seed42 --lr0 0.00003 --freeze $chosenExperiment.selected_config.freeze --b3-selection artifacts/training/broad-b1-b2-selection.json
```

B3 hanya diizinkan setelah pilihan validation B1/B2 tersedia dan cocok dengan dataset, bobot awal serta seluruh config pemenang (selain lr0 yang turun menjadi 0.00003). Jika kedua kandidat gagal, status tetap `KEEP_BASELINE`; diagnosis data/kurva sebelum memperluas pencarian. B1/B2/B3 tidak menerima `--evaluate-only`. B0 wajib mengevaluasi bobot awal tanpa training.

Untuk general, gunakan `train_general_focus.py`, nama run `issue10-general-...`, dan bobot B0 `artifacts/verified-model-bundle/yolo11s.pt`. Dataset default kedua trainer adalah output baru issue #8/#9. Jalur dataset dapat diberikan dengan `--data`, tetapi prasyarat review/checksum tetap berlaku. `--plan-only` melakukan preflight tanpa training.

## Seleksi dan kestabilan seed

`choose` menerima report validation dari cabang model yang sama. Hash dataset/label/bobot awal serta imgsz/batch/nbs/seed harus sama. Macro-F1 kelas fokus dipakai untuk urutan kandidat; penurunan precision/recall fokus >2 poin atau F1 kelas kontrol dengan dukungan ≥50 >2 poin menolak kandidat. Kelas fokus memerlukan dukungan ≥50. Untuk general, kesalahan dua arah phone/remote juga tidak boleh bertambah. Hasil tanpa kenaikan tetap mempertahankan baseline.

AP dilaporkan untuk setiap komponen pada validation. Nilai mAP Broad dan general tidak dirata-ratakan untuk mengklaim mAP detektor gabungan. Kenaikan F1 di JSON memakai skala 0–1; 0.05 berarti lima poin persentase. Keputusan tahap ini hanya untuk eksperimen berikutnya, bukan promosi rilis.

Setelah memilih satu kandidat per cabang, jalankan resep pemenang dengan seed17 dan seed73 memakai `--stage extra`, nama run baru, dan config lain yang sama. Gunakan kembali report pemenang seed42. Lalu:

```powershell
python scripts/manage_tuning_experiments.py stability --selection artifacts/training/broad-selection.json --report PATH_SEED17_REPORT --report PATH_PRIMARY_SEED42_REPORT --report PATH_SEED73_REPORT --output artifacts/training/broad-stability.json
```

`stability` memerlukan tepat tiga seed, bukti training/config/input yang sama, dan kandidat seed42 yang sudah dibekukan. Kandidat stabil berstatus `VALIDATION_STABLE_PENDING_RESOLUTION_REVIEW`, tetap `promoted=false` dan `holdout_opened=false`. Run augmentasi tambahan dan dua pengulangan seed per cabang berada di luar enam training run awal; catat biaya dan alasan setiap run.

## Status implementasi saat ini

[Rencana run yang terukur](../evaluations/issue10-experiment-plan.json) mencatat prasyarat yang belum tersedia. B0/B1/B2/B3, uji augmentasi, serta pengulangan seed belum dijalankan karena dataset akhir masih menunggu review. Tidak ada skor kandidat atau keputusan pemenang yang dibuat-buat. CI memverifikasi alur kode dengan model tiruan dan dataset sintetis; CI hijau tidak membuktikan model membaik.
