# Audit FPI-Det dan RealWaste

7 November 2026. Audit selesai pada 60 sampel visual dari masing-masing sumber. RealWaste menghasilkan **15 foto kardus dengan bbox manual** untuk batch riset Broad Waste v5. Arsip FPI-Det berisi **22.879 foto dan pasangan labelnya** berhasil diperoleh dan seluruh gambar sudah melalui pemeriksaan hash/pHash. Sampel FPI disimpan dengan label native dan status audit, karena masih ada masalah format, duplikasi split dan cakupan anotasi untuk training COCO 80 kelas.

## RealWaste

Sampel diambil secara deterministik dari 461 foto pada folder Cardboard di repositori penulis, commit `156e8e9154eeb163df30fcfd767f55b19633bb06`. Seluruh 60 unduhan cocok dengan checksum Git blob penerbit. Screening contact sheet diikuti pemeriksaan resolusi penuh dan anotasi pixel XYXY pada foto yang disetujui.

| Hasil | Jumlah |
|---|---:|
| Foto diunduh dan disaring | 60 |
| Foto mendapat bbox kardus manual | 15 |
| Foto ditunda | 45 |
| Duplikat persis dalam sampel | 0 |
| Pasangan pHash dekat dalam sampel | 0 |
| Kecocokan foto baru dengan dataset/benchmark yang dilindungi | 0 |

Alasan penundaan mencakup karton minuman yang memerlukan pemetaan `drink_carton`, kemasan makanan dengan taksonomi yang meragukan, bahan yang belum jelas, serta foto yang masih memerlukan review bbox lengkap. `Cardboard_133` tampak seperti wadah plastik meskipun berasal dari folder Cardboard. Foto tersebut ditunda. Empat puluh lima foto yang ditunda tidak semuanya dinilai salah label.

Dataset hasil berada di `data/local/broad-waste-v5-realwaste-research/data.yaml`.

| Ukuran | Hasil |
|---|---:|
| Seluruh foto | 3.908 |
| Train penuh / validation / test | 2.243 / 975 / 690 |
| Train aktif | 958 |
| Train aktif v4 yang dipertahankan | 943 |
| Foto baru pada train aktif | 15 |
| Anotasi kardus train aktif | 93 → 108 |

Seluruh 3.893 record v4 mempertahankan gambar, label dan split. Pilihan 943 foto train aktif v4 juga dipertahankan. Foto RealWaste baru memakai satu kelompok proksi konservatif pada train karena metadata identitas objek fisik/sesi tidak tersedia. Tidak ada foto dari sumber baru ini yang ditambahkan ke validation/test. Checksum, geometri bbox, daftar split, kelompok dan pHash lintas split lulus verifier. [Audit batch](../../evaluations/realwaste-v5-audit.json), [verifikasi file nyata](../../evaluations/realwaste-v5-verification.json), [keputusan dan bbox manual](../../annotations/realwaste-cardboard-v1.json).

Ada perbedaan lisensi konkret. UCI menyatakan CC BY 4.0, sementara README penulis dan halaman IEEE DataPort menyatakan CC BY-NC-SA 4.0. Batch lokal memakai ketentuan penulis yang lebih ketat dengan status `EXPERIMENTAL_NONCOMMERCIAL_RESEARCH`. Perbedaan ini perlu diselesaikan sebelum penggunaan yang lebih luas. [UCI](https://archive.ics.uci.edu/dataset/908/realwaste), [repositori penulis](https://github.com/sam-single/realwaste), [IEEE DataPort](https://doi.org/10.21227/65m5-rr72).

Bbox dibuat dan ditinjau Codex. Review manusia independen masih diperlukan. Hasil ini menambah data yang siap dipakai pada eksperimen riset. Peningkatan akurasi perlu diukur setelah fine-tuning dan benchmark ulang.

## FPI-Det

Snapshot kode dan anotasi penerbit dipatok ke commit `83c452d1a1e5b1bac88afe1a6834cb9c25e036eb`. `phone_train_yolo.rar` berukuran sekitar 3,25 MB berisi kode training, termasuk aset contoh, sehingga arsip tersebut tidak menyediakan korpus foto mentah. Repositori juga menyediakan JSON anotasi, YAML kelas dan gambar contoh berbentuk kolase. Penerbit menyatakan lisensi MIT untuk dataset. [Sumber penerbit](https://github.com/KvCgRv/FPI-Det).

Arsip foto asli pada tautan Drive penerbit berukuran **1.162.345.390 byte**. Konektor berhasil membaca metadata, tetapi menolak unduhan mentah dengan HTTP 413 karena batas 256 MiB. Unduhan melalui browser berhasil tersimpan di Downloads meskipun alat browser tidak mengembalikan path penyelesaiannya. File ditemukan di filesystem, disalin ke cache proyek dan diverifikasi melalui SHA-256 serta CRC seluruh anggota ZIP. Arsip lokal berada di `data/local/downloads/external-focus-v5/fpi-images.zip`, dengan SHA-256 `5eea5d8ce0317c3bfabecb29c54eea2845a022bcfd190b4c8ca781b52bf7cb48`.

| Split penerbit dalam arsip asli | Foto | Bbox ponsel | Bbox kepala/wajah |
|---|---:|---:|---:|
| Train | 18.800 | 8.567 | 24.993 |
| Validation | 1.730 | 809 | 1.318 |
| Test | 2.349 | 1.132 | 2.968 |
| Total | 22.879 | 10.508 | 29.279 |

Sebanyak **242 baris label memuat beberapa record bbox lima kolom yang tersambung dalam satu baris**. Parser audit memisahkan record lengkap tersebut dengan mempertahankan nilai koordinat. Hasilnya 10.508 bbox ponsel, dibanding 10.255 pada README. JSON konversi yang tersedia di GitHub juga memiliki jumlah yang berbeda dari label YOLO asli. Anotasi yang jumlah kolom atau angkanya tidak valid ditolak, dan bbox keluar batas tidak dipotong otomatis. [Audit metadata dan arsip](../../evaluations/fpi-det-audit.json).

Pada label YOLO asli, **82 bbox ponsel train**, dua validation dan sembilan test terdeteksi keluar batas atau mempunyai ukuran tidak positif pada pemeriksaan geometri ketat. Label kepala/wajah juga mempunyai flag geometri. Flag harus diperiksa terhadap gambar karena sebagian merupakan penyimpangan tepi kecil. Native class 0 adalah `phone` dan class 1 adalah `face` pada YAML penerbit, sementara README menyebut anotasi kepala. Bbox kepala/wajah tidak dapat dipetakan sebagai bbox seluruh tubuh `person` di COCO.

## Duplikasi dan sampel visual FPI

Seluruh 22.879 foto berhasil didekode. Pemeriksaan menggunakan SHA-256 dan pHash berjarak paling banyak lima bit. Indeks enam bagian digunakan untuk menemukan semua pasangan yang memenuhi batas tersebut tanpa membandingkan seluruh pasangan secara naif.

| Pemeriksaan | Hasil |
|---|---:|
| Salinan foto identik yang berlebih | 62 |
| Kelompok hash identik melintasi split | 1 |
| Pasangan pHash dekat melintasi split penerbit | 78 |
| Pasangan pHash dekat dalam split penerbit | 495 |
| Pasangan dengan dataset/benchmark proyek yang dilindungi | 0 |
| Kelompok gambar identik dengan anotasi berbeda | 60 |

Perbedaan anotasi diukur setelah urutan bbox dinormalisasi dan koordinat dibulatkan ke enam desimal. Perbedaan itu memerlukan rekonsiliasi label sebelum memilih salah satu salinan gambar. Jumlah pasangan tidak sama dengan jumlah foto independen. pHash juga belum membuktikan independensi sesi fisik, sehingga frame berbeda dari adegan yang sama masih bisa tersebar ke split berbeda. [Audit pixel dan contoh pasangan](../../evaluations/fpi-det-pixel-audit.json).

Sampel visual terdiri dari 40 train, 10 validation dan 10 test, dengan bbox ponsel seluas minimal 3% gambar serta geometri valid. Dari 60 foto, **45 lolos screening untuk objek ponsel dan 15 ditunda**. Tiga belas kandidat yang lolos ditandai tampak belakang. Contoh dengan identitas/visibilitas target yang meragukan atau domain video ber-overlay ditunda untuk review lanjutan. Review ini berfokus pada ponsel. Kelengkapan anotasi kepala dan seluruh kelas COCO memerlukan pemeriksaan terpisah.

Foto sampel serta label native satu bbox per baris tersedia di `data/local/fpi-det-audit-sample-v1/`. Folder berstatus `AUDIT_ONLY_NATIVE_PHONE_FACE`. Native label ponsel/wajah memiliki cakupan yang berbeda dari anotasi lengkap COCO 80 kelas. Dataset model umum memerlukan reanotasi kelas lain yang terlihat atau eksperimen specialist yang dievaluasi terpisah. [Keputusan screening](../../annotations/fpi-det-sample-v1.json), [ringkasan sampel](../../evaluations/fpi-det-sample-audit.json).

## Reproduksi

```powershell
.\.venv-ai\Scripts\python.exe scripts/prepare_external_focus.py acquire
.\.venv-ai\Scripts\python.exe scripts/prepare_external_focus.py audit --fpi-archive data/local/downloads/external-focus-v5/fpi-images.zip
.\.venv-ai\Scripts\python.exe scripts/prepare_external_focus.py pixels
.\.venv-ai\Scripts\python.exe scripts/prepare_external_focus.py sample --output data/local/fpi-det-audit-sample-rerun
# Pilih folder output baru jika batch riset sudah tersedia.
.\.venv-ai\Scripts\python.exe scripts/prepare_external_focus.py build --output data/local/broad-waste-v5-realwaste-research-rerun
.\.venv-ai\Scripts\python.exe scripts/verify_broad_data.py --data data/local/broad-waste-v5-realwaste-research --output evaluations/realwaste-v5-verification.json
.\.venv-ai\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Tahap acquire mematok snapshot sumber dan mengambil 60 foto RealWaste, metadata/anotasi FPI serta aset contoh penerbit. Arsip foto Drive diterima sebagai input lokal pada tahap audit. File gambar, cache, ZIP dan paket dataset tetap lokal. Keputusan review serta bbox manual disimpan sebagai JSON dalam repo. Seluruh 255 tes Python lulus, termasuk pemulihan bbox multi-record, pencarian pasangan pHash, validasi checksum penerbit, ekspor sampel native dan pelestarian daftar train aktif.
