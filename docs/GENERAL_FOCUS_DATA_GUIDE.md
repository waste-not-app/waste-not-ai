# Data ponsel dan remote — issue #9

Implementasi ini bertumpu pada PR #13 (issue #8). Review manusia ditunda sampai akhir sesuai arahan pengguna; data belum direview tetap kandidat. Tidak ada training, penggantian model produksi, atau pembukaan holdout akhir.

## Taksonomi dan cakupan anotasi

Model general memakai seluruh 80 class ID COCO. `remote` adalah ID 65 dan `cell phone` ID 67; `person` tetap ID 0. Semua kelas lain yang terlihat harus tetap dianotasi.

FPI-Det memakai taksonomi native **0=phone, 1=face**. Angka tersebut tidak boleh langsung dimasukkan ke label COCO. Antrean FPI menyimpan native bbox satu record per baris, termasuk bbox wajah, tanpa membuat `data.yaml`. Foto FPI memerlukan anotasi COCO lengkap oleh reviewer; mengganti ID phone saja tidak menyelesaikan anotasi seluruh objek. Bbox wajah tidak diubah otomatis menjadi bbox person.

Untuk COCO, semua anotasi publisher yang valid dipertahankan, termasuk orang dan kelas selain target. Paket review baru menyertakan gambar resolusi asli, metadata sumber/izin, bbox lengkap, dan contact sheet `coco-*.jpg`. Contact sheet adalah alat screening; kelengkapan bbox tetap diperiksa pada gambar resolusi penuh. Dua foto yang semula tertolak hanya karena pembulatan biner dipulihkan setelah toleransi numerik diperbaiki (1e-12, tanpa clipping objek yang benar-benar di luar gambar).

## Pemeriksaan sebelum review

- Cocokkan checksum arsip FPI dengan snapshot yang sudah diaudit serta hash cache piksel dengan laporan audit yang tersimpan di repo.
- Parse ulang semua label native, termasuk beberapa record lima kolom yang berada pada satu baris. Pertahankan nilai koordinat asli.
- Karantina geometri tidak valid, foto tanpa anotasi ponsel, salinan piksel dengan anotasi berbeda, pasangan pHash dekat, benturan split publisher, dan kecocokan dengan dataset/replay terlindungi. Karantina tidak berarti semua foto tersebut salah; perlu rekonsiliasi sebelum dipertimbangkan ulang.
- Verifikasi kembali checksum, ukuran, orientasi, dan pHash setiap foto FPI yang diekspor. Kelompok pHash bukan bukti independensi benda/sesi.
- Prioritaskan kandidat train dengan tag lama `rear_visible`, tetapi pertahankan review lama sebagai metadata historis. Tag/review lama tidak menjadi persetujuan COCO80 baru.
- Kandidat COCO baru berasal dari train2017 dengan metadata lisensi per foto yang diterima preparer. Metadata snapshot dipatok dengan checksum arsip. Semua kandidat COCO dianggap **training candidates**, tidak menjadi holdout baru karena paparan pretraining/sumber sudah diketahui.

## Review yang dilakukan di akhir

Gunakan `examples/general-focus-intake-empty.json` sebagai kerangka, dengan seluruh nama COCO dalam urutan aslinya. Setiap record mengikuti format intake issue #8: ID unik, file gambar relatif, split, group benda, session ID, SHA-256, pHash, bbox, izin/atribusi, dan review manusia lengkap.

Tambahkan atribut berikut pada setiap record general:

```json
{
  "source": "FPI-Det",
  "annotation_scope": "COCO80_COMPLETE",
  "reannotated_from_native": true,
  "view_tags": ["phone_rear", "handheld"]
}
```

`reannotated_from_native=true` hanya ditulis untuk FPI setelah reanotasi selesai. Status tersebut merupakan bukti keputusan reviewer, tidak dibuat otomatis oleh preparer. Semua record tetap memerlukan `visual_review.labels` berisi 80 label, `reviewer_type=human`, identitas/tanggal reviewer, dan `all_objects_annotated=true`. Izin harus `RELEASE_ELIGIBLE` beserta atribusi. Negatif kosong memerlukan verifikasi negatif eksplisit.

Tag yang diterima:

| Kelompok | Tag |
|---|---|
| Tampilan ponsel | `phone_front`, `phone_rear`, `phone_side` |
| Kondisi | `handheld`, `occluded`, `low_light`, `small_object` |
| Remote | `remote_short`, `remote_long` |
| Pembanding | `counterexample_calculator`, `counterexample_powerbank`, `control_other` |

Catat variasi casing, jumlah kamera, bentuk/tombol remote dan alasan anotasi ambigu dalam catatan reviewer. Jangan menyimpulkan tampak depan hanya karena tag tampak belakang belum tersedia. Benda/sesi/burst yang sama serta kelompok yang terhubung tetap satu split.

Sasaran data tambahan: sedikitnya **200 foto train ponsel dan 150 foto train remote** yang disetujui. Validation serta holdout baru masing-masing memerlukan **50 instance dari 20 kelompok independen per kelas**. Satu foto bisa memuat kedua kelas, sehingga jumlah kelas tidak dijumlahkan menjadi jumlah foto independen.

## Perintah

Atur environment sebelum memakai CLI. Akuisisi/pemeriksaan pHash memakai environment riset dengan OpenCV dan NumPy; CI service tidak perlu ditambahi dependency tersebut.

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONIOENCODING = 'utf-8'

python scripts/prepare_general_focus.py restore --source PATH_GENERAL_LAMA --output data/local/general-confusion-v1
python scripts/verify_broad_data.py --data data/local/general-confusion-v1 --output evaluations/issue9-legacy-data-verification.json

python scripts/prepare_general_focus.py fpi-queue --archive PATH_FPI_ZIP --pixels PATH_FPI_PIXELS_JSON --protected data/local/broad-waste-v4/manifest.json --protected data/local/general-confusion-v1/manifest.json --protected benchmarks/camera-replay-v1.json --output data/local/issue9-fpi-review --limit 400

python scripts/prepare_general_focus.py coco-plan --annotations PATH_COCO_ANNOTATIONS_ZIP --protected data/local/broad-waste-v4/manifest.json --protected data/local/general-confusion-v1/manifest.json --protected benchmarks/camera-replay-v1.json --output data/local/issue9-coco-plan.json
python scripts/prepare_general_focus.py coco-acquire --plan data/local/issue9-coco-plan.json --protected data/local/broad-waste-v4/manifest.json --protected data/local/general-confusion-v1/manifest.json --protected benchmarks/camera-replay-v1.json --output data/local/issue9-coco-review
# Untuk melanjutkan antrean yang sama, tambahkan --resume. Hash/label cache diperiksa ulang.

python scripts/prepare_general_focus.py audit --base data/local/general-confusion-v1 --intake PATH_INTAKE_JSON --protected data/local/broad-waste-v4/manifest.json --protected benchmarks/camera-replay-v1.json --output evaluations/issue9-data-readiness.json
python scripts/prepare_general_focus.py build --base data/local/general-confusion-v1 --intake PATH_INTAKE_JSON --protected data/local/broad-waste-v4/manifest.json --protected benchmarks/camera-replay-v1.json --output data/local/general-focus-issue9-v2
python scripts/verify_broad_data.py --data data/local/general-focus-issue9-v2 --output evaluations/issue9-built-data-verification.json
```

Tambahkan manifest issue #8 yang sudah diperbarui ke `--protected` bila tersedia. Audit mengembalikan exit code 2 bila data/review belum lengkap. Builder menolak kondisi tersebut. Train lama tetap dipertahankan; legacy test tidak dicampurkan ke holdout baru. Source provenance dipertahankan pada `original_source` ketika builder mengemas data baru.

## Diagnosis tanpa inference ulang

```powershell
python scripts/evaluate_phone_remote.py --manifest benchmarks/camera-replay-v1.json --predictions evaluations/issue8-baseline-replay.json --output evaluations/issue9-phone-remote-baseline.json
```

Evaluator memeriksa identitas manifest/gambar, keunikan dan kelengkapan cache, serta keberadaan checksum model general baseline yang terdaftar. Ia menghitung ulang precision/recall/F1 dan kesalahan `phone → remote` / `remote → phone` dari bbox prediksi. Matching salah label dan matching precision/recall dihitung terpisah.

Subset diagnosis berisi seluruh foto replay yang anotasinya mencakup phone atau remote, termasuk contoh negatif. Ini **diagnosis legacy**, bukan holdout baru atau bukti perbaikan model. Kasus ponsel biru pengguna yang pernah salah menjadi remote berada di luar benchmark ini; nol kesalahan phone→remote pada replay tidak membuktikan kasus tersebut selesai. Subset rear-view baru dan pengujian kandidat tetap menunggu anotasi/tag yang direview.
