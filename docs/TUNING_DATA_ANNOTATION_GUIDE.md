# Panduan anotasi tuning YOLO — issue #8

Anotasi memakai sepuluh class ID Broad Waste yang sudah ada. Model tetap mendeteksi jenis objek. Bahan yang belum jelas dan pilihan penanganan memerlukan konteks/konfirmasi pengguna.

| ID | Label | Aturan dan contoh |
|---|---|---|
| 0 | bottle | Botol dengan badan dan leher; bahan kaca/plastik tidak dipaksakan dari penampilan |
| 1 | can | Kaleng minuman/makanan yang identitasnya terlihat, termasuk penyok |
| 2 | cup | Gelas/cangkir terbuka; bedakan dengan botol berleher dan wadah makanan |
| 3 | plastic_bag | Kantong pembawa dengan rongga/pegangan; kemasan pembungkus masuk wrapper |
| 4 | paper | Lembaran, sobekan, atau kertas kusut; buku utuh tidak otomatis menjadi paper |
| 5 | cardboard | Kardus bergelombang atau karton struktural/lipat yang identitasnya jelas |
| 6 | drink_carton | Kemasan karton minuman, misalnya kotak susu/jus; bukan kotak kardus pengiriman |
| 7 | plastic_wrapper | Bungkus/film fleksibel, sachet atau bungkus makanan; foil/bahan ambigu ditunda untuk pemeriksaan |
| 8 | food_container | Wadah makanan berbentuk baki, kotak atau mangkuk; bedakan dari cup dan drink_carton berdasarkan fungsi yang terlihat |
| 9 | bottle_cap | Tutup botol terpisah dan terlihat jelas; tutup yang menempel tidak diberi bbox terpisah jika batasnya tidak dapat dipastikan |

Prioritaskan identitas objek dan batas anotasi yang konsisten. Bungkus bercetak tidak otomatis kertas; warna coklat tidak otomatis kardus. Kemasan yang identitasnya tidak dapat ditentukan diberi `DEFERRED` dengan alasan, tanpa memaksakan class ID. Pertahankan keputusan penolakan lama untuk ditinjau, bukan diubah menjadi persetujuan otomatis.

## Bbox dan kelengkapan

- Format YOLO `[class_id, center_x, center_y, width, height]`, ternormalisasi terhadap piksel gambar yang disimpan, semua angka finite, lebar/tinggi positif, seluruh tepi di [0,1].
- Bbox ketat mengelilingi bagian objek yang terlihat; jangan menebak bagian di luar gambar atau tertutup. Tandai occlusion/truncation pada catatan review. Normalisasi EXIF sebelum menggambar bbox.
- Anotasi seluruh objek yang terlihat dalam sepuluh kelas tersebut, termasuk kelas kontrol. Beberapa benda tetap memiliki bbox masing-masing meskipun bertumpuk.
- Foto negatif kosong hanya disetujui setelah reviewer memastikan tidak ada objek dalam cakupan. Kalkulator, kain, buku atau dompet menjadi pembanding sesuai isi foto; objek sampah lain yang ikut terlihat tetap dianotasi.
- Semua validation/holdout dan bbox ambigu memerlukan pemeriksaan manusia independen. Simpan nama/identitas reviewer, tanggal, keputusan, cakupan label, dan koreksi. Pemeriksaan ini merupakan bukti review, bukan pengukuran akurasi model.

## Kelompok, sumber, dan split

Simpan sumber asli, URL lisensi, atribusi, SHA-256, pHash, ID benda/kelompok, serta sesi pengambilan. Satu benda fisik/sesi/burst/video dan turunannya tetap satu split. Kelompok yang terhubung oleh sesi atau benda dihitung sebagai satu kelompok independen. SHA-256/pHash melengkapi metadata sesi, tidak menggantikannya.

Train tambahan ditargetkan minimal 150 foto cardboard serta masing-masing 100 foto paper, drink_carton, dan plastic_wrapper. Validation dan holdout **baru** masing-masing memerlukan minimal 50 instance dari 20 kelompok per kelas fokus. Jumlah ini batas awal eksploratif; ketidakpastian tetap dilaporkan. Foto legacy, review lama, screenshot diagnosis dan replay bukan holdout baru.

RealWaste tetap berstatus riset karena catatan lisensinya belum selesai dan review manusia independen masih diperlukan. Jangan menulis `RELEASE_ELIGIBLE` untuk data tersebut tanpa bukti penyelesaian hak penggunaan. Data dengan sumber/izin belum jelas tetap berada di antrean review.

## Bentuk record intake

Gunakan label lengkap seperti `examples/tuning-intake-empty.json`. Setiap record baru berisi:

```json
{
  "id": "capture-unique-id",
  "image": "images/capture-unique-id.png",
  "split": "train",
  "group": "physical-object-id",
  "session_id": "capture-session-id",
  "prepared_sha256": "SHA256_DARI_FILE",
  "phash": 123,
  "boxes": [[5, 0.5, 0.5, 0.4, 0.3]],
  "rights": {
    "accepted": true,
    "status": "RELEASE_ELIGIBLE",
    "license_url": "URL_IZIN",
    "source_page": "URL_SUMBER",
    "attribution": "PEMILIK_FOTO_DAN_ATRIBUSI"
  },
  "visual_review": {
    "decision": "APPROVED",
    "reviewer": "IDENTITAS_REVIEWER",
    "reviewer_type": "human",
    "reviewed_at": "TANGGAL_REVIEW",
    "all_objects_annotated": true,
    "labels": ["bottle", "can", "cup", "plastic_bag", "paper", "cardboard", "drink_carton", "plastic_wrapper", "food_container", "bottle_cap"]
  }
}
```

Nilai di atas merupakan contoh format, bukan foto atau bukti review nyata. Hitung checksum dan pHash dari gambar sebenarnya. Untuk negatif kosong, `boxes=[]` memerlukan `visual_review.verified_negative=true`. Simpan JSON intake di direktori yang menjadi root path gambar relatifnya. pHash menggunakan fungsi repo `prepare_drinking_waste.perceptual_hash` dengan OpenCV/NumPy environment riset.

## Perintah kerja

```powershell
# Pakai output baru. Sumber backup tidak dimutasi.
python scripts/prepare_tuning_data.py restore --source PATH_BASELINE_LAMA --output data/local/broad-waste-v4
python scripts/verify_broad_data.py --data data/local/broad-waste-v4 --output evaluations/issue8-baseline-data-verification.json

# Siapkan foto belum terpakai beserta contact sheet dan keputusan review historis.
python scripts/prepare_tuning_data.py stage-review --cache PATH_CACHE_TACO --base data/local/broad-waste-v4 --output data/local/issue8-review

# Wajib melindungi replay dan manifest model lain yang telah dipakai.
python scripts/prepare_tuning_data.py audit --base data/local/broad-waste-v4 --intake PATH_INTAKE_JSON --protected benchmarks/camera-replay-v1.json --protected PATH_GENERAL_MANIFEST --output evaluations/issue8-new-data-readiness.json

# Hanya setelah audit lengkap; output baru, lalu verifikasi file hasilnya.
python scripts/prepare_tuning_data.py build --base data/local/broad-waste-v4 --intake PATH_INTAKE_JSON --protected benchmarks/camera-replay-v1.json --protected PATH_GENERAL_MANIFEST --output data/local/broad-waste-issue8-v5
python scripts/verify_broad_data.py --data data/local/broad-waste-issue8-v5 --output evaluations/issue8-v5-data-verification.json
```

Audit mengembalikan exit code 2 ketika data/review belum memenuhi sasaran. Ini menjaga kekurangan data terlihat. Hasil `READY_FOR_TRAINING_EXPERIMENT` hanya menyatakan pemeriksaan intake dan dukungan data lulus; bukan bukti model membaik. Verifikasi manusia terhadap sumber dan anotasi tetap diperlukan. Builder menolak intake yang belum lulus; mempertahankan train aktif legacy, menambahkan train/validation baru, dan memakai test baru sebagai holdout. Test legacy tetap berada di baseline untuk regresi dan tidak dicampur ke holdout baru. Pipeline produksi tidak diubah. Setelah intake lengkap, penggabungan serta pemeriksaan dataset final masih harus dijalankan sebelum issue #8 ditutup.
