# Issue #11 — implementasi resolusi dan benchmark

Implementasi masuk ke `main` lewat PR #25. Pilot CPU memakai bobot rilis dan validation legacy. Eksperimen penerimaan masih menunggu dataset hasil review #8/#9 dan receipt keputusan #10 yang cocok dengan dataset tersebut.

Yang sudah diimplementasikan:

- Input640/768/960 tervalidasi pada factory detektor; default dan identitas bundle640 lama tetap dipertahankan. Profil resolusi berbeda memiliki identitas/cache terpisah dan input size tidak diganti melalui setter.
- Audit resize/rotasi/kompresi demo, Android, gateway Go dan worker Python berdasarkan source lokal. Profil Android ditambahkan sebagai pendekatan codec terpisah, dengan batas interpretasinya.
- Runner memerlukan receipt kandidat beku dan snapshot dataset yang cocok. Manifest benchmark hanya memuat validation; foto bersama direkonsiliasi tanpa double count dan kebocoran lintas dataset/sesi diperiksa.
- Pengukuran CPU batch1 dengan warmup≥5, pengulangan≥3 serta urutan profil diselang-seling. 960 hanya dijalankan setelah768 layak; kontrol640/768 diukur ulang pada stage960.
- Laporan kualitas per kelas/ukuran objek, AP tiap komponen, p50/p95 detektor, native memori proses dan HTTP opsional yang terpisah. Cache mengikat input/model/shape/environment; snapshot diperiksa lagi pada akhir run.
- Gate kualitas, dukungan, regresi dan budget p95 1.25× kontrol640; tidak ada aktivasi atau promosi otomatis.

Tes mencakup bbox pada gambar landscape/portrait serta rotasi EXIF, penolakan orig_shape yang salah, skema wire88 label, identitas bundle640 yang tetap sama, invalid input size, pemisahan warmup, interleaving pada byte identik, pencatatan ukuran sumber/terkirim, bucket GT yang melintasi batas ukuran prediksi, receipt/input yang berubah, shared-photo annotation conflicts, kebocoran sesi lintas model, 960 bersyarat, sparse/NaN/regresi codec, pengukuran native memori dan konteks/auth HTTP.

## Pilot CPU 20 November 2026

Pilot membandingkan Broad Waste v4 dan COCO11s yang dirilis pada square 640 dan 768. Sampel mencakup 43 foto validation General, 45 foto validation Broad dengan kelas fokus, dan empat kontrol Broad per sumber yang dipilih menurut urutan checksum gambar. Totalnya 100 foto dan 403 anotasi. Setiap resolusi menerima byte yang sama pada browser JPEG90, simulasi redup, dan pendekatan kompresi Android. F1 dan recall dihitung pada kandidat yang bisa ditinjau di aplikasi dengan IoU 0,50 dan cakupan kelas sesuai anotasi sumber.

Pengukuran menggunakan Intel Core i7-14700HX, delapan thread PyTorch, batch 1, lima warmup di luar timer, tiga pengulangan per input, dan urutan resolusi diselang-seling. AP komponen memakai validation legacy masing-masing komponen (975 foto Broad dan 43 foto General), JPEG90, dan confidence floor 0,001. HTTP memakai endpoint loopback terpisah per resolusi pada 12 foto browser JPEG90 yang sama dan mengukur waktu respons termasuk inference.

| Profil | p50 640 / 768 (ms) | p95 640 / 768 (ms) | Rasio p95 | F1 640 / 768 |
|---|---:|---:|---:|---:|
| Browser JPEG90 | 333,2 / 462,2 | 380,5 / 511,7 | 1,345× | 44,5% / 42,7% |
| Simulasi redup 0,55 | 332,2 / 456,5 | 377,9 / 516,6 | 1,367× | 43,9% / 41,4% |
| Simulasi Android JPEG85 → gateway JPEG90 | 336,0 / 460,4 | 383,7 / 517,9 | 1,350× | 43,7% / 42,6% |

Recall objek kecil pada browser JPEG90 turun dari 30/144 ke 29/144. Pada simulasi redup berubah dari 27/144 ke 28/144, dan pada simulasi Android kedua profil menemukan 30/144. F1 ponsel pada browser turun dari 53,8% ke 40,0%. Lima dari enam kelas fokus punya kurang dari 50 instance. RSS puncak selama tahap detektor mencapai 1.234,8 MiB untuk satu proses yang memuat kedua profil.

HTTP lokal mencatat p50/p95 sebesar 324,1/355,6 ms pada 640 dan 447,4/480,0 ms pada 768. Angka ini berdasarkan 36 request terukur per resolusi. Jalur staging dan gateway berada di luar pengukuran ini.

| Komponen | Foto | mAP50 640 / 768 | mAP50–95 640 / 768 |
|---|---:|---:|---:|
| Broad Waste v4 | 975 | 35,82% / 33,99% | 25,83% / 24,08% |
| COCO11s | 43 | 68,99% / 65,43% | 49,83% / 48,20% |

**Pilot memilih 640.** P95 detektor 768 melampaui batas 1,25× pada ketiga profil, dan F1 keseluruhan turun pada ketiganya. Pengukuran 960 dilewati sesuai urutan eksperimen. Pilot memakai validasi yang telah dipakai saat pengembangan, dengan cakupan yang belum memenuhi syarat penerimaan.

[Laporan pilot lengkap](../evaluations/issue11-resolution-pilot.json) berisi hasil per kelas dan ukuran objek, konfigurasi pengukuran, serta hash model dan input.

## Penerimaan akhir

1. Review/build/verifier data #8/#9 dan keputusan kandidat/seed nyata #10.
2. Ulangi pengukuran AP, recall/F1, latency, memori, dan HTTP untuk 640/768 pada dataset yang disetujui dan bobot yang tercatat di receipt. Uji 960 bila 768 memenuhi syarat kualitas dan biaya.
3. Ukur kondisi kamera yang direkam dengan jalur capture nyata. Konfirmasi biaya HTTP pada lingkungan target untuk resolusi yang dipertimbangkan.
4. Catat pilihan resolusi final. Kalibrasi dan evaluasi akhir sebelum aktivasi berada di issue #12.

Konfigurasi produksi memakai input 640. Issue #11 tetap terbuka sampai penerimaan akhir selesai.

Lihat [panduan benchmark](RESOLUTION_EXPERIMENT_GUIDE.md) dan [rencana preflight](../evaluations/issue11-resolution-plan.json).
