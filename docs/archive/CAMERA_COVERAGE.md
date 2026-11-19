# Ponsel menghasilkan nol kandidat pada kamera

Perbandingan terukur pada 117 foto kini tersedia dalam [benchmark replay kamera](CAMERA_BENCHMARK.md), termasuk precision/recall, waktu CPU, salah label dan kotak bertumpuk untuk v3, COCO serta gabungan.

7 November 2026. Root cause: demo v3 yang diberikan memakai checkpoint khusus sepuluh kelas sampah. Taxonomy-nya tidak memiliki ponsel, remote, laptop, buku, atau sebagian besar benda umum. Foto dalam screenshot pengguna direplay secara lokal: v3 menghasilkan nol bbox, termasuk pada probe confidence 0,05. Model umum pada crop foto yang sama menghasilkan remote, sehingga ini kehilangan cakupan model dan bukan kegagalan pengiriman bingkai kamera. Skor dan ambang tidak diturunkan untuk memaksakan deteksi.

## Perbaikan

`demo-camera-v3.bat` sekarang memilih v3 sebagai model sampah dan YOLO11n COCO sebagai pendamping melalui `--general-model`. Kedua checkpoint memeriksa foto yang sama secara lokal. Tidak ada pengiriman foto ke Gemini, perubahan default COCO, training ulang, atau perubahan policy rekomendasi 08.

Hasil memakai satu taxonomy gabungan 88 label: 80 ID COCO dipertahankan, delapan label sampah yang belum ada ditambahkan. Bottle/cup berbagi label yang sama. Mapping mendukung 66 kelas visual ke 77 dari 85 pilihan katalog; pilihan lainnya tetap manual. Angka ini cakupan mapping, bukan akurasi.

Box dengan label sama dan IoU minimal 0,65 digabung dengan mempertahankan skor asli yang lebih tinggi. Label berbeda tetap terlihat sebagai hipotesis yang dapat dikonfirmasi; overlap ditandai UNCERTAIN. Semua label Broad Waste tetap review-only, termasuk bottle/cup pada model gabungan. Maksimum 50 bbox mengikuti kontrak. Bbox/list menyediakan jalur pilihan ketika bingkai bertumpuk; jangan menganggap label berskor tertinggi selalu benar.

Identitas gabungan mencakup checksum dan preprocessing kedua komponen, taxonomy dan versi merge. Mapping `broad-coco-objects-01` dan review profile `broad-coco-experimental-01` dapat direplay oleh service rekomendasi dari snapshot. ID COCO dipertahankan agar risiko elektronik/tajam tidak hilang ketika hasil dipilih. Taxonomy pendamping yang tidak cocok ditolak saat startup.

Perintah dari root repo:

```powershell
.\demo-camera-v3.bat
# Atau eksplisit, tanpa membuka kamera fisik otomatis:
.\.venv-ai\Scripts\python.exe -m wastenot_ai.cli demo --demo-guides --no-browser --port 8098 --device cpu --input-shape square --model artifacts/training/broad-waste-v3-batch2-yolo11s/weights/best.pt --general-model artifacts/models/yolo11n.pt
```

Worker mendukung konfigurasi yang sama melalui flag CLI atau `AI_GENERAL_MODEL_PATH`, bersama `AI_MODEL_PATH` Broad Waste. Model tunggal tetap tersedia tanpa pendamping. UI menjelaskan apakah mode aktif memakai model khusus sampah, umum, atau gabungan.

## Hasil dan batas pengujian

- 232 tes otomatis lulus. Regresi endpoint demo terlebih dahulu gagal karena dukungan ponsel hilang saat v3 dipilih, lalu lulus setelah pendamping ditambahkan.
- Uji gabungan mencakup ID/mapping kardus dan ponsel, perlindungan risiko elektronik, deduplikasi tanpa membuang hipotesis berbeda, replay selection melalui service rekomendasi, perubahan checksum, penolakan taxonomy salah, maksimum 50 bbox, dan hasil kosong yang tetap jujur.
- 25 pemeriksaan browser sebelumnya diulang pada kedua model nyata dan provider fixture: semuanya lulus tanpa error JavaScript. Termasuk rekomendasi botol kaca tajam, koreksi kardus, lifecycle kamera simulasi, kuota dan tampilan mobile. Export 34 artifact kontrak tetap sesuai.
- Foto screenshot pengguna direplay ke camera stream simulasi, kemudian diunggah lewat pemilih file. Keduanya menghasilkan kandidat remote (skor sekitar 0,88 pada capture), bukan nol kandidat. Setelah koreksi manual ke Ponsel dan penerima YES, formulir menghasilkan RECYCLE dengan penanganan khusus. Tidak ada error JavaScript, kamera fisik tidak diakses, salinan foto tidak disimpan dan Gemini tidak dipanggil.

**Pengenalan ponsel pada foto tersebut masih salah sebagai remote.** Perbaikan ini memperluas cakupan dan mengatasi hasil kosong; belum membuktikan semua benda akan terdeteksi atau benar kelasnya. Memilih Ponsel merupakan koreksi pengguna, bukan keberhasilan YOLO mengenali ponsel. Label berbeda dari dua checkpoint juga dapat menambah false positive; konfirmasi tetap diperlukan. Evaluasi mAP v3 tidak boleh digunakan sebagai angka model gabungan. Belum ada benchmark akurasi gabungan atau uji kamera fisik baru.

Artifact metadata: `artifacts/camera-phone-replay.json` dan `artifacts/combined-camera-browser/camera-demo-browser.json`. Foto pribadi tidak ditambahkan ke dataset atau regression fixture. Ulangi replay menggunakan gambar lokal milik penguji:

```powershell
.\.venv-ai\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv-ai\Scripts\python.exe scripts/verify_camera_coverage.py C:/path/to/screenshot.png --photo-region 44 114 700 483 --url http://127.0.0.1:8098
```

Koordinat crop harus sesuai gambar yang diuji. Replay menyimulasikan capture dan upload tanpa menyimpan foto. Jika memakai foto langsung, hapus `--photo-region`. Untuk suite browser penuh gunakan instance segar karena suite sengaja menguji batas generation.
