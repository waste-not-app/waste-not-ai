# Dataset fokus kardus dan ponsel

7 November 2026. Dua paket dataset eksperimen sudah disiapkan mengikuti dua komponen detector gabungan. Pemilihan kelas mengacu pada benchmark replay kamera, dengan recall kardus 2/15, ponsel 6/23 dan remote 5/37. Foto rekomendasi pribadi pengguna tidak dimasukkan ke dataset.

## Paket kardus

`data/local/broad-waste-v4/data.yaml` memakai sepuluh kelas Broad Waste. Dataset berisi 3.893 foto, termasuk 95 foto tambahan yang disetujui melalui screening. Tambahan berasal dari 79 foto TACO dan 16 foto Open Images yang dikonfirmasi sebagai kardus. Label generik Open Images `Box` diperiksa bahannya sebelum dipetakan ke `cardboard`.

| Ukuran | v3 batch kedua | v4 fokus |
|---|---:|---:|
| Seluruh foto | 3.798 | 3.893 |
| Foto train penuh | 2.173 | 2.228 |
| Foto train aktif | 901 | 943 |
| Foto validation | 948 | 975 |
| Foto test | 677 | 690 |
| Anotasi kardus train aktif | 29 | 93 |
| Anotasi kardus validation | 9 | 51 |
| Anotasi kardus test | 15 | 30 |

Seluruh 3.798 foto lama mempertahankan hash gambar, label dan split. `train.txt` berisi pilihan latihan aktif, sedangkan `train-full.txt` menyimpan daftar penuh. Jumlah anotasi kardus berbeda dari jumlah foto karena satu foto bisa memuat beberapa objek.

## Paket ponsel dan pembanding

`data/local/general-confusion-v1/data.yaml` memakai urutan 80 kelas COCO yang sama dengan model umum saat ini. Dataset berisi 620 foto, terdiri dari 552 foto COCO 2017, 54 foto ponsel Open Images dan 14 foto kardus Open Images sebagai contoh negatif terverifikasi. Seluruh anotasi COCO pada foto terpilih dipertahankan, termasuk objek di luar tiga kelas fokus.

| Ukuran | Train | Validation | Test |
|---|---:|---:|---:|
| Foto | 535 | 43 | 42 |
| Anotasi ponsel | 322 | 19 | 18 |
| Anotasi remote | 204 | 9 | 9 |
| Anotasi kulkas | 79 | 11 | 11 |
| Foto negatif kardus | 14 | 0 | 0 |

Sebanyak 72 dari 80 kelas memiliki anotasi latihan. Foto dari kelas lain berfungsi sebagai replay saat fine-tuning. Foto negatif kardus mempunyai file label kosong dengan status `VERIFIED_NEGATIVE` serta persetujuan review khusus. Keempat belas foto tersebut juga menjadi contoh positif kardus pada paket Broad Waste dan seluruhnya ditempatkan pada train di kedua paket.

Ponsel tampak belakang yang ditandai saat screening berjumlah 15 foto train, satu validation dan satu test. Ponsel modern dengan tiga kamera belakang masih kurang terwakili. Unduhan tambahan Wikimedia Commons ditolak HTTP 403 dan foto dari sumber tersebut belum masuk paket ini. Pengambilan foto kamera fisik baru masih diperlukan untuk kasus Samsung biru yang dilaporkan pengguna.

## Akuisisi dan screening

| Sumber calon | Foto diperiksa | Disetujui | Ditunda |
|---|---:|---:|---:|
| TACO kardus | 83 | 79 | 4 |
| COCO pembanding | 587 | 552 | 35 |
| Open Images ponsel dan Box | 134 | 70 | 64 |

Total ada 701 foto tambahan unik pada kedua paket. Empat belas foto dipakai oleh kedua model dalam split train yang sama. Foto ambigu seperti tablet/PDA, GPS, gambar iklan, remote yang meragukan, kotak berbahan lain dan duplikat adegan ditunda. Keputusan per foto, alasan, reviewer, waktu dan checksum tersimpan pada snapshot review dan manifest.

Screening dilakukan Codex melalui contact sheet beranotasi dan pemeriksaan judul sumber untuk foto yang ambigu. Review manusia independen masih diperlukan. Anotasi Open Images berfokus pada kelas target dan belum diaudit secara menyeluruh untuk kelengkapan semua 80 kelas COCO. Dataset diberi status `EXPERIMENTAL`.

## Pemisahan evaluasi

Benchmark `camera-replay-v1.json` mempertahankan SHA-256 `fa83d698c132e6bdda33a2c7cffcf8e501001b40b62e4d92eb0d704bedf72aaa`. Data tambahan tidak mempunyai kecocokan pHash berjarak paling banyak lima bit dengan benchmark. Sebanyak 47 foto test lama Broad Waste memang menjadi sumber benchmark dan tetap berada pada test.

Pemeriksaan antarpaket juga menguji hash identik, kelompok proksi dan pHash dekat. Hasilnya tidak memiliki konflik split. Kelompok berdasarkan metadata dan pHash masih merupakan pendekatan terhadap adegan fisik yang benar-benar independen. [Laporan audit kebocoran](../evaluations/focus-v4-leakage-audit.json).

Foto COCO train 2017 sudah tersedia saat pretraining model COCO. Penggunaannya di sini merupakan penyeimbangan ulang untuk fine-tuning. Foto COCO val disisihkan untuk validation/test eksperimen fine-tuning setelah mengecualikan benchmark, tetapi validation COCO juga umum digunakan dalam pengembangan checkpoint asal. Tambahan Open Images memberikan sumber foto berbeda. Angka evaluasi berikutnya perlu menyebutkan riwayat sumber ini.

## Sumber dan hak penggunaan

Foto TACO tambahan memakai lisensi CC BY 2.0 yang diperiksa per foto melalui metadata Flickr. Foto COCO terpilih mempunyai metadata lisensi CC BY atau CC BY-SA. Foto Open Images tambahan mempunyai metadata lisensi CC BY 2.0. URL asli, pembuat, lisensi dan provenance label tercatat per foto pada `manifest.json`. Kedua paket menyertakan `DATASET_NOTICES.md`. [COCO](https://cocodataset.org/#download), [TACO](https://tacodataset.org/), [Open Images](https://storage.googleapis.com/openimages/web/download.html).

## Verifikasi dan penggunaan

```powershell
.\.venv-ai\Scripts\python.exe scripts/verify_broad_data.py --data data/local/broad-waste-v4 --output evaluations/focus-v4-cardboard-verification.json
.\.venv-ai\Scripts\python.exe scripts/verify_broad_data.py --data data/local/general-confusion-v1 --output evaluations/focus-v4-general-verification.json
.\.venv-ai\Scripts\python.exe scripts/prepare_focus_datasets.py audit
.\.venv-ai\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Kedua verifier lulus pemeriksaan file gambar, checksum, geometri label, hak sumber dan daftar split. Pembulatan tepi pada 15 anotasi DrinkingWaste lama memakai toleransi asli 0,002. Seluruh 245 tes Python lulus. [Verifikasi kardus](../evaluations/focus-v4-cardboard-verification.json), [verifikasi model umum](../evaluations/focus-v4-general-verification.json).

Untuk akuisisi dan build ulang, `prepare_focus_datasets.py` menyediakan tahap `acquire-cardboard`, `acquire-general`, `acquire-openimages`, `build-cardboard`, `build-general` dan `audit`. Cache sumber serta review berada di `data/local/downloads/focus-v4/`. Builder meminta folder output baru agar snapshot hasil sebelumnya terjaga. Snapshot sumber, dataset lokal dan gambar berada dalam jalur yang diabaikan Git, sehingga clone baru memerlukan proses akuisisi dan review ulang atau salinan paket lokal.

[Training dan benchmark ulang telah selesai](FOCUS_V4_RESULTS.md). Trainer Broad Waste dipakai untuk paket sepuluh kelas dan `train_general_focus.py` disediakan untuk paket 80 kelas COCO. F1 benchmark gabungan v4 dengan COCO awal meningkat dari 43,0% menjadi 48,0%. Kedua kandidat fine-tuning model umum belum meningkatkan deteksi ponsel/remote pada validation. Kasus ponsel biru pengguna masih perlu ditangani.
