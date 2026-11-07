# Perluasan dan penyeimbangan dataset v3

Training kandidat batch kedua sudah selesai: 28 epoch, best epoch 18. Pada test identik mAP50 meningkat 28,28%→35,18%; kandidat tetap eksperimen. [Hasil lengkap dan cara mencoba](BROAD_WASTE_V3_RESULTS.md).

## Pembaruan batch kedua

7 November 2026. Terminal sudah dapat mengakses jaringan. Akuisisi bertambah dari 58 menjadi **147 foto** sebelum Flickr membalas HTTP 429; request ke sumber tersebut dihentikan. Seluruh 147 foto telah melalui screening awal: **99 disetujui dan 48 ditunda**. Dibanding batch pertama, ada 49 foto tambahan yang lolos. Dataset baru berada di `data/local/broad-waste-v3-batch2/`; batch pertama tetap utuh.

Dataset batch kedua memiliki **3.798 gambar**, split penuh **2.173 train / 948 val / 677 test**, serta 901 gambar training aktif yang dipilih tanpa duplikasi. Seluruh 3.699 record baseline mempertahankan gambar dan penempatan split aslinya. Tambahan publik dibagi ulang melalui komponen kelompok/pHash terhadap baseline; split tambahan tidak dijanjikan identik dengan batch pertama. Belum ada model yang dilatih pada batch pertama. Snapshot akuisisi dan review disimpan di dalam dataset agar audit tidak bergantung pada cache yang akan berubah pada unduhan berikutnya.

| Kelas langka | Training baseline | Training batch pertama | Training batch kedua |
|---|---:|---:|---:|
| plastic_bag | 6 | 12 | 20 |
| paper | 7 | 13 | 17 |
| cardboard | 17 | 29 | 29 |
| drink_carton | 3 | 5 | 10 |
| plastic_wrapper | 22 | 40 | 47 |
| food_container | 6 | 12 | 28 |
| bottle_cap | 23 | 28 | 31 |

Target 300 instance training per kelas masih belum tercapai. Angka screening merupakan pemeriksaan Codex pada contact sheet beranotasi, bukan sertifikasi ahli manusia. Foto meragukan tidak dipaksakan masuk; keputusan dan hash tersimpan di `review-snapshot.json`. [Report build batch kedua](../evaluations/broad-waste-v3-batch2-build.json) dan [verifikasi dataset nyata](../evaluations/broad-waste-v3-batch2-verification.json). Verifikasi mencakup hash/label/split aktif dan memastikan tidak ada kelompok proksi atau pHash dekat yang melintasi split.

Seluruh **204 tes Python lulus** setelah penambahan dukungan dataset eksplisit pada training, validasi bahwa YAML menunjuk dataset/split yang dipilih, serta perbandingan checkpoint dengan taksonomi yang sama. CLI training tidak lagi diam-diam memakai v1 ketika menjalankan eksperimen data baru. Parameter `--compare-model` menilai checkpoint lama pada validation yang identik dengan kandidat. Profil confidence atau model default tidak dipromosikan otomatis.

Training kandidat YOLO11s selesai pada GPU RTX 4070 Laptop dengan konfigurasi yang sama dengan kandidat broad v2 (COCO initial weights, transfer baris bottle/cup, freeze 10, lr 0,0003, seed 42, maksimum 30 epoch dan early stopping). Pembeda utama adalah dataset dan pilihan gambar training. Hasil dan batas peningkatan dilaporkan pada tautan di atas. Reproduksi dengan nama run baru:

```powershell
.\.venv-ai-train\Scripts\python.exe scripts/train_broad_waste.py --data data/local/broad-waste-v3-batch2 --name broad-waste-v3-batch2-yolo11s --initial-model artifacts/models/yolo11s.pt --seed-known-head --freeze 10 --lr0 0.0003 --epochs 30 --compare-model artifacts/training/broad-waste-v2-yolo11s/weights/best.pt
```

Log/progres berada di `artifacts/training/broad-waste-v3-batch2-yolo11s-training.log` dan folder run bernama sama. Evaluasi tersimpan terpisah dari laporan dataset; hasil model tidak disimpulkan dari jumlah gambar.

## Riwayat batch pertama

7 November 2026. **Status: dataset eksperimen v3 berhasil dibuat.** Jalur terminal tetap gagal DNS, tetapi browser internal berhasil mengunduh 58 foto baru dengan lisensi CC BY 2.0 yang diperiksa pada halaman masing-masing foto. Sebanyak 50 lolos screening visual awal, 8 ditunda. Dataset lokal menjadi **3.749 gambar**; seluruh 3.699 gambar dan penempatan split baseline tetap utuh. Model belum dilatih ulang. Laporan: [acquisition](../evaluations/broad-waste-v3-acquisition.json), [build](../evaluations/broad-waste-v3-build.json), [verifikasi](../evaluations/broad-waste-v3-verification.json).

## Hasil batch pertama

Lokasi dataset: `data/local/broad-waste-v3/data.yaml`. Split penuh: **2.142 train / 940 val / 667 test**. Daftar training seimbang memilih 879 gambar unik dari 2.142 gambar train; `train-full.txt` tetap menyediakan seluruh training. Pemilihan gambar utuh mengurangi botol dari 1.773 menjadi 600 instance dan cup dari 1.334 menjadi 936, dengan semua contoh kelas minoritas dipertahankan. Cup dapat melebihi batas 600 karena muncul bersama kelas lain yang masih memerlukan contoh.

| Kelas langka | Training lama | Training v3 aktif |
|---|---:|---:|
| plastic_bag | 6 | 12 |
| paper | 7 | 13 |
| cardboard | 17 | 29 |
| drink_carton | 3 | 5 |
| plastic_wrapper | 22 | 40 |
| food_container | 6 | 12 |
| bottle_cap | 23 | 28 |

Seluruh sepuluh kelas kini mempunyai ground truth pada val dan test; karton minuman yang sebelumnya kosong masing-masing memiliki dua instance. Jumlah tersebut masih terlalu kecil untuk klaim kualitas kelas yang kuat. **Target 300 instance training per kelas belum tercapai.** Kenaikan jumlah data bukan bukti kenaikan akurasi model.

Screening dilakukan Codex pada lima contact sheet beranotasi, bukan sertifikasi ahli manusia atau audit setiap objek pada resolusi asli. ID yang ditunda: `1257`, `1205`, `1239`, `1323`, `866`, `1192`, `1078`, `1495`. Alasannya tersimpan per foto pada `data/local/downloads/broad-waste-v3/review.json`, termasuk dugaan objek kertas yang belum dianotasi pada `1192`, objek sangat kecil, atau adegan padat yang perlu pemeriksaan lebih rinci. Anotasi penulis tidak ditebak ulang. Seluruh 50 foto yang disetujui lolos pemeriksaan komponen split dan duplikat; tidak ada penolakan split tambahan pada batch ini.

## Audit kebutuhan

Target awal eksperimen adalah 300 instance training per kelas, bukan jaminan kecukupan data atau akurasi. Instance adalah bounding box, bukan jumlah foto unik. Metadata utama TACO yang sudah tersedia secara lokal menghasilkan 825 kandidat gambar yang belum masuk baseline dan memuat kelas langka.

| Kelas | Instance training lama | Kekurangan menuju 300 | Kandidat instance baru TACO, sebelum kurasi dan split |
|---|---:|---:|---:|
| bottle | 1.761 | 0 | 243 |
| can | 453 | 0 | 93 |
| cup | 1.333 | 0 | 74 |
| plastic_bag | 6 | 294 | 87 |
| paper | 7 | 293 | 97 |
| cardboard | 17 | 283 | 185 |
| drink_carton | 3 | 297 | 42 |
| plastic_wrapper | 22 | 278 | 718 |
| food_container | 6 | 294 | 58 |
| bottle_cap | 23 | 277 | 260 |

Kolom terakhir adalah batas atas dari anotasi kandidat, **bukan tambahan yang sudah diperoleh**. Hak foto, unduhan, pemeriksaan visual, deduplikasi, dan alokasi val/test akan menguranginya. Satu gambar TACO (`549`) ditolak karena bbox melewati batas piksel; bbox tidak dikoreksi dengan tebakan. TACO sendiri tidak cukup untuk mencapai target semua kelas, bahkan sebelum kurasi. Sumber tambahan atau foto sendiri tetap dibutuhkan terutama untuk karton minuman, kantong, kertas, dan wadah makanan.

## Sumber dan keputusan pemakaian

- [Repo penulis TACO](https://github.com/pedropro/TACO) menyediakan anotasi utama dalam format COCO; seluruh bbox kelas target pada satu foto dipertahankan. Anotasi unofficial tidak dipakai.
- Rilis dataset [Zenodo 3587843](https://zenodo.org/records/3587843) tersedia melalui tautan DOI resmi penulis dan mendeklarasikan CC BY 4.0 untuk rilis tersebut. Metadata lisensi per foto tidak seluruhnya rinci. Proses ini tetap mengikuti audit baseline: periksa halaman Flickr per foto dan simpan lisensi/atribusi spesifik; deklarasi arsip/toolkit tidak mengganti pemeriksaan tersebut.
- Label klasifikasi gambar tidak diubah menjadi bbox seluruh foto. Label umum `plastic` atau `container` dari sumber lain tidak langsung diasumsikan kantong plastik atau wadah makanan.

## Melanjutkan proses

Dari root repo, gunakan environment AI yang sudah terpasang:

```powershell
# Hitung kandidat/kekurangan dari metadata lokal; tanpa jaringan.
.\.venv-ai\Scripts\python.exe scripts/expand_broad_waste.py --stage plan

# Audit hak foto dan unduh secara berurutan, maksimum 100 request per run.
.\.venv-ai\Scripts\python.exe scripts/expand_broad_waste.py --stage acquire --max-requests 100
```

Cache/progres berada pada `data/local/downloads/broad-waste-v3/`. Run berikutnya melanjutkan hasil yang sudah berhasil; tidak menimpa dataset lama. HTTP 429 atau kegagalan jaringan menghentikan run tanpa retry berulang. Snapshot manifest baseline dan anotasi sumber dikunci dengan hash. Run acquisition terminal terhambat dengan `URLError: [Errno 11001] getaddrinfo failed`. Browser internal dipakai sebagai jalur unduhan yang berhasil; akses otomatis aplikasi Chrome ditolak, sehingga aplikasi itu tidak dipakai.

Importer `--stage import-browser --browser-import PATH_JSON` menerima daftar file yang benar-benar diunduh beserta `source_id`, `local_path`, `source_page`, `creator_page`, `license_url`, `author`, `checked_at`, dan `download_url` yang diperiksa pada DOM browser. Batch di sesi ini tersimpan lokal sebagai `browser-import.json`. Importer memvalidasi identitas foto terhadap anotasi, URL gambar/lisensi, rasio ukuran dan decoder, serta checksum. Impor ulang file yang sama tidak menggandakan record. Download tetap berstatus PENDING hingga screening visual selesai.

Setelah gambar benar-benar tersedia, periksa bbox pada resolusi cukup, kecocokan kelas, dan apakah semua objek target telah dianotasi. Buat JSON review lokal per gambar yang benar-benar diperiksa, misalnya:

```json
[
  {
    "source_id": "ID_TACO_YANG_SUDAH_DIPERIKSA",
    "raw_sha256": "HASH_FOTO_DARI_ACQUISITION_JSON",
    "decision": "APPROVED",
    "reviewer": "NAMA_PENINJAU_SEBENARNYA",
    "reviewed_at": "TANGGAL_REVIEW_ISO"
  }
]
```

Contoh di atas adalah format, bukan bukti review. Gambar tanpa approval spesifik beserta hash tidak masuk build.

```powershell
.\.venv-ai\Scripts\python.exe scripts/expand_broad_waste.py --stage build --review data/local/downloads/broad-waste-v3/review.json
```

Dataset v3 sekarang sudah ada, sehingga perintah build itu akan menolak menimpanya. Untuk batch berikutnya, gunakan output versi baru melalui `--output data/local/broad-waste-v3-batch2`, sesudah memperbarui review gambar baru. Evaluasi konsistensi dataset ini dapat diulang tanpa jaringan:

```powershell
.\.venv-ai\Scripts\python.exe scripts/verify_broad_data.py --data data/local/broad-waste-v3 --output artifacts/broad-waste-v3-verification.json
```

Build menolak output yang sudah ada dan menolak membuat dataset bila belum ada gambar baru yang lolos. Gambar baseline disalin tanpa kompresi ulang. Split lama dikunci; kelompok proksi TACO, hash identik dan pHash menghubungkan gambar baru. Kandidat yang menghubungkan dua split lama ditolak. Proksi filename/pHash belum membuktikan independensi benda fisik/sesi kamera.

`data.yaml` menunjuk `train.txt` yang memilih gambar utuh secara deterministik, dengan batas awal 600 instance per kelas mayoritas. Seluruh gambar kelas di bawah batas diprioritaskan. Semua bbox dalam satu gambar tetap disimpan, sehingga kelas yang muncul bersama dapat melewati batas. Tidak ada duplikasi foto atau penghitungan augmentasi sebagai data baru. `train-full.txt` menyimpan pilihan seluruh training untuk perbandingan. Validation/test tidak diseimbangkan ulang. Manifest membedakan jumlah penuh dan jumlah training aktif.

Dataset yang berhasil dibangun tetap berstatus EXPERIMENTAL. Target tiap kelas mungkin belum terpenuhi. Penambahan data tidak dianggap peningkatan model sampai training dan evaluasi terpisah dilakukan.

## Verifikasi sesi ini

Sepuluh tes dataset lulus, termasuk konflik split transitif, duplikat, gambar minoritas dengan beberapa kelas, bbox sumber invalid, penghentian DNS/429, cache dari snapshot berbeda, penolakan foto tanpa review, build lengkap memakai gambar sintetis fixture, dan impor browser dengan pemeriksaan identitas serta resume tanpa duplikasi. Fixture bukan tambahan data training. Hash manifest baseline sesudah pekerjaan identik dengan sebelum pekerjaan. Suite AI keseluruhan pada sesi sebelumnya berhenti memberi keluaran dan diinterupsi; kelulusannya tidak diklaim dari pengujian dataset ini.

Verifikasi dataset nyata lulus untuk seluruh 3.749 gambar: hash gambar, geometri/isi label, daftar training aktif/full, jumlah instance per split, kelompok proksi dan pHash lintas split. Tidak ditemukan kebocoran kelompok/near-duplicate antar split. Lima belas pembulatan tepi bbox pada sumber Drinking Waste lama tetap memakai toleransi audit awal 0,002; label tidak diubah diam-diam. Pemeriksaan tambahan membandingkan 3.699 record baseline dan memastikan seluruh hash gambar serta penempatan split lama identik.
