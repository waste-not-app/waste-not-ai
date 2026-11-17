# Rencana tuning YOLO WasteNot

Tanggal penyusunan: 17 November 2026. Status: **rencana, belum diimplementasikan**.

Repo tujuan adalah `https://github.com/waste-not-app/waste-not-ai.git`, pada folder lokal `waste-not-ai-integration`. Dokumen ini merinci lima tahap yang akan dikerjakan satu per satu. Angka target, jumlah data, dan anggaran run di bawah adalah usulan awal untuk dikunci sebelum eksperimen; semuanya belum merupakan hasil pengukuran.

## Titik awal dan tujuan

Baseline yang tercatat adalah gabungan **Broad Waste v4 YOLO11s + COCO11s**, versi `broad-coco-b0f633193403`, dengan 88 label gabungan dan input square640.

| Komponen | SHA-256 |
|---|---|
| Broad Waste v4 | `3fc8b7a0baf192b6a88554ce83b24c3ac4ffbd4551e5ee2109a81b68a608462f` |
| COCO11s | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |

Laporan 7 November mencatat hasil berikut pada replay lama berisi 117 foto dan 222 objek. Angka ini menjadi konteks diagnosis; baseline dan kandidat harus diukur kembali pada dataset identik untuk setiap perbandingan baru.

| Ukuran | Hasil historis |
|---|---:|
| Precision / recall / F1 gabungan | 67,8% / 43,7% / 53,2% |
| Kardus benar | 5 dari 15 objek |
| Ponsel benar | 12 dari 23 objek |
| Remote benar | 9 dari 37 objek |
| Recall plastic_wrapper / paper / drink_carton | 5,9% / 0% / 0% |
| Latency CPU median / p95 | 211,0 / 278,1 ms |
| Pasangan kotak dengan label bertentangan | 18 |

Dukungan paper hanya empat objek dan drink_carton lima objek. Sebagian replay juga berasal dari test Broad Waste. Test lama dan replay tidak boleh dianggap dua sampel independen, dan keduanya telah diamati selama pengembangan. Karena itu diperlukan holdout baru untuk menilai hasil akhir. Latency historis hanya mencakup jalur detektor, tanpa upload dan overhead HTTP.

Tujuan utama adalah meningkatkan deteksi enam kelas fokus `cardboard`, `plastic_wrapper`, `paper`, `drink_carton`, `cell phone`, dan `remote`. Kelas lain tetap diperiksa untuk mencegah regresi. YOLO tetap bertugas mendeteksi objek; pilihan penanganan Reduce, Reuse, Recycle, dan Trash tetap mengikuti sistem rekomendasi dan konteks pengguna.

## Persiapan sebelum tahap 1

- Sinkronkan basis implementasi dengan main terbaru sambil mempertahankan pekerjaan lokal. Saat rencana dibuat, checkout lokal berada di `9eb0e64`, branch `codex/direct-handling-recommendations`, ahead satu commit terhadap tracking branch; referensi `origin/main` berada di `feb61aa`. Buat branch tuning dari basis yang sudah diperiksa, misalnya `feat/yolo-quality-v5`.
- Inventarisasi dataset, checkpoint, ruang penyimpanan, GPU/VRAM, serta environment yang benar-benar tersedia. Folder `data/local` pada checkout ini belum menyediakan paket data training yang dapat dipakai. Angka pada laporan lama tidak membuktikan asetnya tersedia sekarang.
- Verifikasi checksum checkpoint dan bundle yang akan menjadi baseline. Bekukan versi Ultralytics yang dipakai repo (`8.4.183`), Python, Torch, CUDA, konfigurasi preprocessing, dan label map. Catat bila environment training berbeda dari runtime CPU.
- Pisahkan `train`, `validation`, replay regresi lama, dan holdout akhir baru. Identitas benda fisik, sesi pengambilan, atau rangkaian video menjadi unit pembagian. Seluruh turunan crop, augmentasi, dan kompresi mengikuti split sumbernya.
- Siapkan manifest berisi sumber, lisensi, checksum, `object_id`/`session_id` bila tersedia, split, cakupan anotasi, dan status review. Gunakan SHA-256 dan pHash untuk screening duplikasi, dilanjutkan review pasangan yang ditandai.
- Tambahkan mode pemilihan kandidat menggunakan validation pada trainer. Kedua trainer sekarang otomatis mengevaluasi test; perilaku itu perlu dipisahkan agar pencarian parameter tidak berulang kali membuka holdout.

**Selesai ketika** baseline dapat direproduksi, identitas data/model tercatat, dan split serta aturan pemilihan telah dibekukan. Pengukuran inference/training dilakukan saat implementasi, bukan saat menyusun rencana ini.

## 1. Perbaikan data dan anotasi kelas sampah

**Masalah yang dituju.** Kardus masih sering terlewat atau salah terdeteksi. Bungkus plastik, kertas, dan karton minuman memiliki recall rendah serta bentuk yang saling membingungkan.

**Langkah kerja**

1. Audit kesalahan per kelas, ukuran bbox, pencahayaan, dan sumber data. Pisahkan salah label, bbox kurang lengkap, objek terlewat, serta kesalahan taksonomi.
2. Tulis panduan anotasi singkat untuk membedakan kardus, kertas, karton minuman, bungkus plastik, dan wadah makanan. Contoh dengan bahan yang tidak dapat dipastikan mendapat status review.
3. Tambahkan foto berizin yang mendekati pemakaian aplikasi: kemasan penyok, kardus terbuka/lipat, bungkus mengilap, objek kecil, latar meja/lantai, genggaman tangan, serta cahaya cukup/redup.
4. Sertakan contoh pembanding seperti buku, dompet, kain, dan kemasan dari kelas lain. Semua objek dalam cakupan label model yang terlihat tetap dianotasi. Foto hanya menjadi negatif kosong jika memang tidak memuat objek dalam cakupan tersebut.
5. Lakukan review anotasi, terutama bbox ambigu dan seluruh data validation/holdout. Identitas reviewer serta keputusan koreksi dicatat.
6. Gabungkan dengan data lama dan periksa komposisi sampling agar penambahan satu kelas tidak menyingkirkan bottle, can, cup, atau kelas kontrol lain.

**Sasaran akuisisi awal**

| Kelas | Tambahan foto train yang diusulkan | Variasi penting |
|---|---:|---|
| cardboard | 150–250 | Kardus tipis/tebal, warna, lipatan, terbuka, objek kecil |
| plastic_wrapper | 100–200 | Transparan/berwarna, refleksi, remasan, cetakan |
| paper | 100–150 | Lembaran, sobekan, tumpukan, kusut |
| drink_carton | 100–150 | Kotak kecil/besar, sedotan, tutup, posisi miring |

Jumlah foto bukan jumlah objek independen. Sediakan validation dan holdout tambahan dengan sasaran masing-masing sedikitnya 50 instance beranotasi dari sedikitnya 20 kelompok benda/sesi per kelas fokus. Ini batas awal untuk evaluasi eksploratif; ketidakpastian tetap dilaporkan. Jika jumlahnya belum tercapai, perluas data sebelum membuat klaim peningkatan kelas tersebut.

**Aset yang sudah diaudit.** Ada 15 foto RealWaste beranotasi kardus untuk eksperimen riset. Catatan lisensinya masih berbeda antara sumber, berstatus `EXPERIMENTAL_NONCOMMERCIAL_RESEARCH`, dan bbox memerlukan review manusia independen. Pakai jalur riset terpisah sampai hak penggunaan untuk rilis diselesaikan, atau gunakan sumber pengganti yang jelas izinnya. Sebanyak 45 foto yang ditunda tidak otomatis dimasukkan.

**Berkas yang direncanakan.** Perluasan `scripts/prepare_focus_datasets.py`, `scripts/prepare_external_focus.py`, dan `scripts/verify_broad_data.py`; manifest/notices dataset; keputusan anotasi di `annotations/`; laporan audit di `evaluations/`. Nama dataset final dipilih setelah sumber dan status lisensinya pasti.

**Kriteria selesai.** Seluruh label dan geometri valid, objek dalam cakupan telah dianotasi, duplikasi/kelompok lintas split direkonsiliasi, sumber serta hak penggunaan tercatat, dan dataset lolos verifier. Tahap ini menghasilkan data; peningkatan akurasi baru dinilai setelah training.

## 2. Data khusus dan evaluasi ponsel versus remote

**Masalah yang dituju.** Ponsel tampak belakang masih diprediksi remote. Dua fine-tune YOLO11n terdahulu belum memperbaiki validation, sehingga eksperimen berikutnya menggunakan COCO11s sebagai kontrol awal dengan data yang diperbaiki.

**Langkah kerja**

1. Kumpulkan ponsel tampak belakang, depan, sisi, kamera satu/tiga, casing polos/bermotif, dipegang tangan, dan sebagian tertutup. Pasangkan dengan remote pendek/panjang, tombol sedikit/banyak, serta benda serupa seperti kalkulator dan power bank.
2. Targetkan tambahan 200–400 foto train ponsel dan 150–250 foto remote. Alokasikan validation/holdout terpisah dengan aturan dukungan yang sama seperti tahap 1. Foto dari satu burst, benda, atau sesi tetap satu split.
3. Manfaatkan FPI-Det secara selektif. Audit lama menemukan 242 baris bbox multi-record, geometri bermasalah, salinan dengan label berbeda, serta duplikasi lintas split. Rekonsiliasi setiap kandidat sebelum ekspor training.
4. Pertahankan 80 label COCO pada model umum. Dataset FPI hanya memberi label phone/face; anotasi wajah tidak dipetakan menjadi seluruh tubuh `person`. Lengkapi objek COCO lain yang terlihat sebelum foto dipakai untuk fine-tuning model umum.
5. Sediakan subset diagnosis khusus ponsel belakang dan pasangan phone/remote. Foto diagnosis yang sudah dilihat tetap menjadi regresi terpisah; jangan mengakuinya sebagai holdout baru.
6. Evaluasi arah kesalahan `phone → remote` dan `remote → phone` secara terpisah, selain precision/recall/F1 masing-masing kelas. Jaga contoh replay kelas COCO lain agar kemampuan umum ikut terukur.

**Berkas yang direncanakan.** Manifest general-focus baru, keputusan review FPI, pembaruan preparer/verifier, daftar kelompok sumber, serta laporan confusion per kondisi. Seluruh 80 class ID harus cocok dengan checkpoint COCO11s.

**Kriteria selesai.** Dua kelas memiliki data pembanding yang memadai, anotasi lengkap sesuai taksonomi, split bebas kebocoran yang diketahui, dan baseline subset diagnosis tercatat. Pengumpulan data tahap 1 dan 2 boleh memakai sesi yang sama dengan anotasi lengkap; penyelesaian implementasinya tetap dilaporkan per tahap.

## 3. Fine-tuning parameter training

**Tujuan.** Mengukur dampak data baru, lalu memilih parameter dengan perubahan terkontrol. Model sampah dan model umum dilatih serta dievaluasi sendiri-sendiri sebelum digabungkan.

**Perubahan tooling lebih dahulu**

- Ekspos `imgsz`, batch, workers, patience, seed, dan augmentasi yang masih hardcoded pada `scripts/train_broad_waste.py` dan `scripts/train_general_focus.py` melalui konfigurasi tervalidasi.
- Pisahkan evaluasi validation untuk seleksi dari evaluasi holdout akhir. Simpan run config, hash data/model, library, perangkat, waktu, dan alasan pemilihan checkpoint.
- Verifikasi bahwa `best.pt` dan epoch yang dilaporkan mengikuti kriteria checkpoint trainer. Jangan menyimpulkan epoch terbaik hanya dari maksimum satu kolom mAP jika fitness trainer berbeda.
- Pertahankan checkpoint awal yang sama untuk seluruh perbandingan dalam satu cabang; gunakan nama run baru, tanpa menimpa eksperimen lama.

**Kontrol awal.** Broad mulai dari checkpoint v4; general mulai dari COCO11s. Gunakan `imgsz=640`, AdamW, batch awal 16 sesuai kapasitas, `lrf=0.01`, cosine LR, maksimal 30 epoch, patience 10, seed 42. Catat effective batch dan konfigurasi warmup; jika batch perlu turun, ulang kontrol pembanding dengan setting yang sebanding. Freeze diperiksa terhadap struktur model yang dipatok.

| Run per cabang | Dataset | Freeze | lr0 | Pertanyaan |
|---|---|---:|---:|---|
| B0, evaluasi saja | Validation baru | — | — | Bagaimana checkpoint lama pada data baru? |
| B1 | Train diperbaiki | 10 | 0.0001 | Apakah data baru membantu dengan resep v4? |
| B2 | Sama dengan B1 | 0 | 0.0001 | Apakah membuka backbone membantu? |
| B3 | Sama dengan B1 | Pilihan B1/B2 dari validation | 0.00003 | Apakah LR lebih kecil mengurangi regresi? |

Anggaran awal enam training run, yaitu tiga per cabang. B3 baru dimulai setelah B1/B2 dinilai. Jika hasil tidak menjanjikan, diagnosis data dan loss terlebih dahulu sebelum menambah pencarian. LR 0.0003 hanya menjadi eksperimen lanjutan bila kurva menunjukkan adaptasi terlalu kecil; tidak otomatis memperluas grid.

**Augmentasi.** Pertahankan resep awal repo (`mosaic=0.5`, `close_mosaic=6`, rotasi 15 derajat, `scale=0.35`, tanpa flip vertikal), lalu bandingkan satu profil lebih ringan pada kandidat terbaik bila diperlukan. Profil ringan menguji mosaic 0.2, rotasi 5 derajat, dan scale 0.2. Variasi cahaya, blur, dan kompresi menggunakan implementasi yang tersedia pada versi terpatok dengan intensitas dari contoh pemakaian. Catat parameter warna aktif; hindari distorsi yang menghilangkan petunjuk kemasan. Uji foto redup nyata secara terpisah dari simulasi.

**Seleksi.** Pakai validation dengan operating threshold 0.50 terlebih dahulu, macro-F1 kelas fokus, confusion phone/remote, mAP50:95 per komponen, serta regresi kelas kontrol. Jangan rata-ratakan mAP dua model untuk mengklaim mAP gabungan. Kandidat terpilih perlu diperiksa kestabilannya pada seed tambahan 17 dan 73; hasil seed 42 tetap disertakan. Run augmentasi dan pengulangan seed dihitung di luar enam run awal dan dicatat dalam anggaran eksperimen.

**Kriteria selesai.** Ada paling banyak satu kandidat utama per cabang beserta laporan pembanding dan keputusan diterima/ditolak. Jika fine-tuning general belum mengalahkan COCO11s, komponen tersebut tetap memakai checkpoint baseline. Akurasi yang belum membaik dicatat apa adanya.

## 4. Ukuran input dan biaya inference

**Tujuan.** Menilai apakah resolusi lebih tinggi membantu objek kecil dengan biaya CPU yang layak.

**Urutan eksperimen**

1. Audit aliran gambar dari upload sampai detektor. Runtime saat ini memakai `imgsz=640`; periksa juga batas ukuran/resize sebelum gambar mencapai worker. Upscale gambar yang sebelumnya telah dikecilkan tidak mengembalikan detail yang hilang.
2. Bekukan checkpoint kandidat dari tahap 3. Bandingkan inference 640 dan 768 pada validation dengan preprocessing, padding, threshold, IoU matching, dan jumlah maksimum deteksi yang sama. Uji 960 hanya jika 768 memberikan manfaat yang layak terhadap biaya.
3. Bandingkan F1/recall per ukuran objek dan kelas, serta mAP komponen. Bila perlu adaptasi training pada 768, lakukan satu fine-tune lanjutan dari kandidat yang sama; laporkan terpisah dari eksperimen inference-only.
4. Ukur baseline dan kandidat pada CPU/perangkat yang sama, batch inference 1, thread tetap, minimal lima warmup yang dikeluarkan dari timer dan tiga pengulangan. Selang-seling urutan model untuk mengurangi pengaruh suhu/beban.
5. Laporkan p50/p95 waktu detektor, penggunaan memori, dan pengukuran HTTP terpisah. Profil JPEG90 lama tetap menjadi replay; tambahkan variasi gambar nyata dari sumber yang tersedia.
6. Jika resolusi aktif berubah, perbarui metadata input, identitas bundle, cache prediksi, dan kalibrasi. Verifikasi transformasi bbox kembali ke koordinat gambar asli serta kesesuaian kontrak API.

**Sasaran biaya awal.** p95 gabungan maksimal 1.25 kali baseline yang diukur ulang pada mesin dan input identik. Resolusi 640 tetap menjadi pilihan jika tambahan kualitas tidak memenuhi tujuan. Angka 278,1 ms historis tidak dijadikan SLA server.

**Berkas yang direncanakan.** Parameterisasi `src/wastenot_ai/detector.py`, metadata model, dan `scripts/benchmark_camera_models.py`; laporan per resolusi; pengujian transformasi bbox dan identitas konfigurasi.

**Kriteria selesai.** Satu ukuran input dipilih berdasarkan kualitas dan latency yang terukur. Tidak ada profil baru yang aktif hanya karena resolusinya lebih besar.

## 5. Threshold per kelas dan penggabungan hasil

**Tujuan.** Mengurangi salah deteksi serta kotak berlabel bertentangan tanpa menyembunyikan objek benar.

**Kalibrasi threshold**

1. Bekukan checkpoint dan ukuran input yang terpilih. Buat cache validation baru dengan prediksi minimal confidence 0.05, lebih rendah dari rentang threshold operasional yang akan diuji. Cache lama yang hanya menyimpan confidence ≥0.50 tidak dapat dipakai untuk menilai threshold yang lebih rendah.
2. Eksperimen threshold operasional 0.10–0.90 dengan langkah 0.05, lalu pemeriksaan lokal berjarak 0.01 di sekitar kandidat. Selaraskan batas prefilter detektor dengan profil; runtime saat ini memfilter mulai 0.25.
3. Untuk kelas dengan sedikitnya 50 GT validation dari sedikitnya 20 kelompok, pilih F1 tertinggi yang memenuhi batas precision dan recall yang telah dikunci. Usulan awal membatasi penurunan precision dan recall masing-masing dua poin persentase terhadap kontrol. Jika dukungan tidak cukup atau syarat tidak tercapai, pertahankan threshold kontrol dan laporkan keterbatasannya.
4. Ikat profil pada hash dua checkpoint, label map, ukuran/preprocessing, dan aturan penggabungan. Cache menyertakan split, manifest, confidence floor, IoU NMS, max-det, serta versi kode. Perubahan identitas memerlukan cache/profil baru.
5. Pisahkan prediksi untuk menghitung AP, misalnya confidence 0.001 sesuai evaluator terpatok, dari cache threshold operasional. Confidence tetap skor model, bukan persentase kebenaran yang terkalibrasi.

**Penggabungan dua model**

1. Jadikan aturan saat ini, same-class IoU 0.65 dengan maksimal 50 kandidat, sebagai kontrol. Uji 0.55 dan 0.75 secara terpisah pada validation, dengan tie-break deterministik dan class mapping yang konsisten.
2. Pertahankan asal model dan skor mentah. Skor dua model tidak diasumsikan langsung sebanding; ukur kasus kelas bersama sebelum mengubah cara memilih kotak pemenang.
3. Analisis konflik phone/remote, cardboard/paper/drink_carton, serta can/bottle. Gunakan overlap tinggi untuk menandai hipotesis yang bertentangan; penghapusan label lintas kelas hanya ditambahkan bila validation membuktikan manfaatnya tanpa merusak kasus dua benda nyata yang bertumpuk.
4. Pertahankan alur konfirmasi pengguna pada hasil ambigu. Sertakan negatif kosong, multiobjek, bbox hampir sama, objek berdampingan, nilai tepat di batas threshold/IoU, dan urutan output sebagai kasus uji.

`scripts/calibrate_yolo.py` saat ini khusus empat label material model lama. Implementasi tahap ini perlu evaluator/profil sesuai taksonomi Broad+COCO, dengan memakai pola validasi identitas yang sudah ada. Profil threshold kardus 0.65 yang pernah ditolak tetap tercatat sebagai kandidat ditolak; tidak diaktifkan ulang berdasarkan validation saja.

**Kriteria selesai.** Profil terpilih memperbaiki tujuan pada validation, hasil runtime sama dengan evaluator, identitas cache/model terlindungi, dan kasus konflik serta multiobjek lulus pengujian. Tahap ini menghasilkan konfigurasi kandidat yang siap menjalani evaluasi akhir.

## Aturan evaluasi dan keputusan rilis

Aturan berikut dikunci sebelum seleksi. “Poin persentase” berarti selisih absolut, misalnya F1 50% menjadi 55% adalah kenaikan lima poin.

| Aspek | Usulan syarat kelulusan |
|---|---|
| Tujuan utama | Macro-F1 enam kelas fokus naik sedikitnya 5 poin terhadap baseline pada dataset identik |
| Perlindungan kelas fokus | Precision dan recall tiap kelas fokus tidak turun lebih dari 2 poin; dukungan memadai sesuai aturan data |
| Perlindungan kelas lain | Micro-F1 gabungan tidak menurun; penurunan F1 >2 poin pada kelas kontrol dengan dukungan memadai menolak kandidat |
| Phone/remote | Proporsi kesalahan dua arah tidak naik dan recall kedua kelas tidak turun lebih dari batas di atas |
| Kondisi sulit | Tidak ada penurunan F1 >2 poin pada subset redup nyata/objek kecil yang memiliki dukungan memadai |
| Biaya CPU | p95 ≤1.25 kali baseline pada pengukuran yang sebanding |
| Integrasi | Skema respons, 88 label, bbox, perilaku konfirmasi, dan identitas artifact konsisten |
| Hak penggunaan | Semua data untuk kandidat rilis memiliki status penggunaan yang sesuai dan atribusi lengkap |

Baseline, kandidat, ukuran sampel, dan TP/FP/FN dilaporkan bersama. Sertakan interval ketidakpastian 95% dengan resampling kelompok benda/sesi. Jika selisih kualitas masih tidak meyakinkan atau dukungan kelas kurang, tandai **belum cukup bukti**, bukan peningkatan yang sudah terbukti. Tidak ada janji mencapai angka akurasi tertentu.

**Evaluasi akhir**

1. Setelah seluruh seleksi validation selesai, bekukan satu kandidat lengkap beserta hash/config dan aturan lulus.
2. Evaluasi baseline dan kandidat tersebut satu kali pada holdout baru yang belum dibuka. Jalankan replay lama sebagai pemeriksaan regresi terpisah.
3. Jika gagal, catat penolakan dan pertahankan baseline. Holdout yang sudah dibuka berubah menjadi data evaluasi teramati untuk siklus berikutnya; seleksi ulang memerlukan holdout baru.
4. Jalankan pengujian kode yang relevan untuk parser/anotasi, split, parameter trainer, identitas cache, threshold, penggabungan, serta kompatibilitas upload/response worker. Verifikasi pipeline GitHub Actions terbaru pada basis main saat implementasi; checkout lokal masih memuat workflow lama yang menggabungkan CI/CD.
5. CI memeriksa kebenaran kode, kontrak, artifact, dan packaging sesuai workflow. Kenaikan kualitas model memerlukan laporan evaluasi dataset; status CI hijau sendiri tidak membuktikan peningkatan akurasi.

## Hasil kerja per tahap dan urutan penerapan

| Urutan | Hasil yang harus tersedia sebelum lanjut |
|---|---|
| Persiapan | Baseline terverifikasi, inventaris aset/environment, manifest dan split beku |
| 1 | Dataset sampah yang direview, pedoman label, notices, laporan verifier |
| 2 | Dataset phone/remote lengkap, rekonsiliasi FPI, subset diagnosis dan baseline |
| 3 | Config serta hasil setiap run, checkpoint kandidat, keputusan seleksi validation |
| 4 | Laporan kualitas/latency per resolusi dan profil input terpilih |
| 5 | Profil threshold, aturan merge, pengujian runtime/evaluator, kandidat lengkap beku |
| Final | Evaluasi holdout, hasil CI, model card, manifest rilis dan rencana rollback |

Setiap tahap dikerjakan sebagai perubahan terpisah agar hasilnya mudah diperiksa. Run yang gagal dan alasannya tetap dicatat. Lama akuisisi/review data serta durasi training ditaksir setelah inventaris aset dan satu run kontrol; jumlah foto atau epoch tidak digunakan untuk menjanjikan waktu selesai.

Untuk GitHub, simpan kode, config, keputusan anotasi yang sesuai izin, notices, model card, dan laporan evaluasi. Dataset/foto mentah serta checkpoint tetap mengikuti pengelolaan artifact; bobot kandidat rilis diterbitkan sebagai Release asset dengan SHA-256 dan diikat ke manifest. Verifikasi kebersihan paket rilis agar data training tidak ikut terkemas. Pertahankan bundle baseline beserta profilnya untuk rollback.

Dokumen ini belum menjalankan training, benchmark inference, emulator, commit, push, maupun deployment. Implementasi pertama adalah persiapan dan tahap 1.

## Rujukan

- [Kalibrasi v4 dan COCO11s](V4_CALIBRATION_AND_COCO11S.md) — baseline, threshold kardus yang ditolak, batas benchmark.
- [Hasil fokus v4](FOCUS_V4_RESULTS.md) — parameter training dan dua fine-tune general yang belum berhasil.
- [Audit dataset eksternal](EXTERNAL_DATASET_AUDIT_V5.md) — RealWaste, FPI-Det, label, duplikasi, dan catatan lisensi.
- [Benchmark per kelas](../evaluations/camera-replay-v1-v4-coco11s-comparison.md) — dukungan kelas, konflik, dan latency historis.
- [Ultralytics Train](https://docs.ultralytics.com/modes/train/) — rujukan parameter training. API harus dicocokkan dengan versi repo yang dipatok.
- [Ultralytics Val](https://docs.ultralytics.com/modes/val/) — rujukan evaluasi dan metrik komponen.
- [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/) — rujukan parameter inference dan filtering.
