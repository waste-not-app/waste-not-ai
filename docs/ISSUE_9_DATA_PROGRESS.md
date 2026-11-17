# Issue #9 — implementasi dan antrean review

Status: **kode dan persiapan data siap; review manusia serta penerimaan dataset final ditunda ke akhir sesuai arahan pengguna**. Issue tetap terbuka dan PR ditumpuk di atas PR #13 untuk issue #8.

## Hasil yang tersedia

- Dataset general legacy dipulihkan tanpa mengubah manifest/anotasi/split: **620 foto**, terdiri atas **535 train, 43 validation, 42 test**. Verifikasi checksum, bbox, daftar split dan pHash lintas split lulus.
- Arsip FPI-Det dipatok ke SHA-256 yang telah diaudit dan cache piksel dicocokkan dengan hash laporan repo. Semua label 22.879 foto dibaca ulang, termasuk 242 baris multi-record.
- **400 kandidat FPI** diekspor untuk review. Label native phone/face dipertahankan, dengan `data.yaml` tidak dibuat. Tag lama tampak belakang menjadi prioritas pemilihan, tanpa mengubahnya menjadi persetujuan anotasi COCO baru.
- **260 kandidat COCO baru** diunduh dari sumber publisher, di luar ID baseline general/waste dan replay. Seluruh anotasi COCO dipertahankan. Dua kandidat yang hanya mengalami pembulatan biner di batas gambar dipulihkan melalui resume terverifikasi.
- Semua **660 foto kandidat** masih berstatus review, bukan data training disetujui atau holdout baru.
- Evaluator diagnosis phone/remote menghitung ulang hasil dari cache baseline issue #8. Tidak ada training, emulator, atau inference baru dijalankan untuk issue #9.

Jumlah kandidat tidak membuktikan variasi benda/sesi yang independen. Foto FPI memerlukan reanotasi COCO80 lengkap dan penetapan kelompok. Foto COCO yang terkait pretraining tidak dijadikan holdout baru.

## Baseline diagnosis legacy

Subset mencakup 70 foto replay dengan cakupan anotasi phone atau remote. Pada profil JPEG90 dan model COCO11s:

| Kelas | TP / GT | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| cell phone | 12 / 23 | 80,0% | 52,2% | 63,2% |
| remote | 9 / 37 | 75,0% | 24,3% | 36,7% |

Matching geometrik salah label menghitung 0 phone→remote dan 3 remote→phone pada sampel ini. Ini belum mencakup kasus ponsel biru pengguna atau membuktikan kemampuan rear-view yang baru. Hasil model gabungan dan profil redup juga tersedia di laporan JSON.

## Review akhir yang masih diperlukan

1. Review gambar penuh dan kelengkapan semua objek COCO, terutama reanotasi FPI dan bbox person yang benar. Face native tidak dipetakan otomatis menjadi person.
2. Isi tag phone front/rear/side, genggaman/occlusion, jenis remote dan pembanding; pastikan keberagaman casing/kamera/tombol.
3. Konfirmasi sumber, hak penggunaan dan atribusi, serta kelompok benda/sesi yang sebenarnya.
4. Sediakan validation dan holdout baru sesuai dukungan minimum. Saat ini data baru yang disetujui masih nol: target train 200 phone/150 remote; validation dan holdout masing-masing 50 instance dari 20 kelompok per kelas.
5. Jalankan audit, build dan verifier dataset final setelah review lengkap. Pertahankan contoh replay kelas COCO lain sebagai kontrol regresi pada tahap training.

Alur audit/build mempertahankan 80 class ID dan menolak native-label intake, label di luar taksonomi, view tag tidak valid, hak/review belum lengkap, serta kebocoran gambar/kelompok/sesi. Batas generik di helper issue #8 diperluas dengan default Broad Waste yang tetap sama, agar aturan keamanan tidak diduplikasi untuk dua model.

Lihat [panduan data general](GENERAL_FOCUS_DATA_GUIDE.md), [template intake](../examples/general-focus-intake-empty.json), [verifikasi legacy](../evaluations/issue9-legacy-data-verification.json), [baseline diagnosis](../evaluations/issue9-phone-remote-baseline.json), dan [kesiapan data baru](../evaluations/issue9-data-readiness.json).

Ringkasan antrean lokal dan hash manifest tersimpan di folder `evaluations/`; foto, arsip, cache anotasi lengkap dan bobot tetap lokal. Tes unit memakai stub pHash/HTTP agar CI tidak mengunduh dataset atau memerlukan OpenCV riset. Akuisisi nyata tetap memeriksa hash dan pHash pada environment riset. CI Linux memverifikasi suite deployment lengkap.

355 tes yang kompatibel Windows dan Ruff lulus. Empat [catatan screening awal](../annotations/issue9-review-questions.json) ditambahkan untuk contoh gambar cetak, navigasi, dan meme yang memerlukan penilaian reviewer. Catatan ini tidak mengubah foto menjadi disetujui atau ditolak otomatis.
