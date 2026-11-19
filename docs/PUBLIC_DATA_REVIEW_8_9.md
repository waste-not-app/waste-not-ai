# Review kandidat data publik untuk issue #8 dan #9

Pada 19 November 2026, Codex memeriksa checksum, orientasi, geometri bbox, metadata sumber dan keputusan sebelumnya pada 708 foto kandidat. Pemeriksaan duplikasi membandingkan SHA-256 serta `pHash` dengan 4.630 catatan baseline dan replay. Screening visual mencakup 48 foto TACO, 60 contoh COCO, dan 20 contoh FPI. Pemeriksaan visual memakai contact sheet dan belum menetapkan kelengkapan anotasi setiap gambar resolusi penuh atau persetujuan reviewer manusia independen.

| Antrean | Hasil screening | Keputusan |
|---|---|---|
| TACO (#8), 48 foto | 45 berbagi kelompok sumber dengan data lama. Hanya 5 foto memuat kardus, dan 17 dari 18 bbox karton minuman menutupi kurang dari 1% luas foto. Foto 1257 memiliki karton kecil terpotong. Metadata lisensi per foto tersedia pada 48 catatan, dengan nama pembuat eksplisit pada 8 catatan. | Pertahankan status tertunda. Kumpulan ini belum menyediakan cukup foto baru beranotasi lengkap dan kelompok independen. |
| COCO train2017 (#9), 260 foto | Label merekam 99 foto ponsel dan 184 foto remote. Screening menemukan contoh iklan cetak, meme, dan perangkat navigasi dengan label ponsel. Metadata lokal menyimpan tautan sumber dan lisensi per foto. | Kandidat train setelah koreksi anotasi dan review sumber. Paparan COCO train2017 pada pretraining membuatnya tidak memenuhi syarat holdout independen. |
| FPI-Det (#9), 400 foto | Label native mencakup ponsel dan wajah. Foto sampel memuat orang dan objek lain yang perlu diaudit dalam cakupan COCO80. Penerbit menyatakan MIT untuk dataset, sementara antrean lokal belum mencatat hak dan atribusi foto dari video sumber. | Periksa hak sumber dan anotasi COCO80 sebelum memakai foto. Simpan label native agar wajah dapat dibedakan dari bbox orang. |

Checksum, orientasi, dan geometri bbox lulus untuk ketiga antrean. Audit mencatat **0 foto baru yang disetujui untuk train, validation, atau holdout**. [Laporan audit](../evaluations/issue8-9-public-queue-review.json), [contoh COCO yang perlu review](../annotations/issue9-review-questions.json), [TACO](https://github.com/pedropro/TACO), [COCO](https://cocodataset.org/#termsofuse), [FPI-Det](https://github.com/KvCgRv/FPI-Det), [YOLO11](https://docs.ultralytics.com/models/yolo11/).

## Uji sumber publik tambahan

Untuk #9, screening metadata validation Open Images menemukan maksimal 15 foto ponsel dan 5 foto remote setelah mengecualikan ID sumber yang pernah dicatat serta menyaring bbox, lisensi dan atribusi. Sebanyak 17 foto terunduh dan lolos pemeriksaan geometri gambar serta pHash. Screening visual menemukan pemutar musik berlabel ponsel dan telepon rumah nirkabel berlabel remote. Lima foto remote sudah berada di bawah target 20 kelompok per split sebelum koreksi. Foto pilot telah dilihat dan tetap berstatus review tanpa persetujuan holdout. [Laporan pilot](../evaluations/issue9-openimages-pilot.json), [metadata Open Images](https://storage.googleapis.com/openimages/web/download_v7.html).

Untuk #8, SortWaste mencakup kemasan cair di fasilitas pemilahan. Ketentuan penggunaan foto belum terverifikasi dan kondisi kamera industrinya berbeda dari kamera aplikasi. Sumber ini belum dimasukkan ke dataset. [SortWaste](https://sortwaste.di.ubi.pt/).

## Status penerimaan

Issue #8 dan #9 tetap terbuka. Untuk masing-masing validation dan holdout, targetnya 50 instance dari 20 kelompok independen per kelas fokus. Sampel tambahan masih perlu mencakup kondisi kamera aplikasi, kelengkapan anotasi, dan catatan sumber. Setelah keduanya siap, jalankan audit intake, build, dan verifier sebelum training #10. [Target #8](ISSUE_8_DATA_PROGRESS.md), [target #9](GENERAL_FOCUS_DATA_GUIDE.md), [status #8–#12](YOLO_ISSUES_8_12_REVIEW.md).
