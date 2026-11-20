# Issue #10 — tooling fine-tuning dan run yang ditunda

Status **tiga pilot training selesai pada 20 November 2026**. Pilot dijalankan setelah pengguna meminta mulai training dengan data yang tersedia. Rangkaian eksperimen terkontrol dan uji kestabilan seed masih terbuka.

## Perubahan

- Trainer Broad dan general memakai runner bersama dengan config tervalidasi untuk resolusi, batch/workers, seed, freeze, patience, learning rate, nominal batch/warmup dan augmentasi.
- Seleksi hanya memakai validation. File YAML yang diteruskan ke Ultralytics tidak memiliki entri test; test.txt, gambar serta label holdout tidak dibuka.
- Sebelum import model, runner memeriksa bukti readiness yang terikat checksum manifest/intake, file train/val, checkpoint awal, dan perubahan label/gambar. Metadata kelompok/pHash semua split diperiksa untuk mencegah kebocoran tanpa mengevaluasi holdout.
- Builder data menyertakan checksum manifest/intake pada bukti readiness. Output lama tanpa binding tersebut perlu dibangun ulang setelah review.
- B0 hanya mengevaluasi baseline. B1/B2 menerapkan freeze10/freeze0, LR0.0001 dan seed42. B3 memerlukan hasil seleksi B1/B2 yang cocok dan menggunakan LR0.00003 dengan config pemenang lainnya tetap sama.
- `best_epoch` berasal dari event penyimpanan checkpoint trainer, bukan maksimum satu kolom CSV. Hash sebelum stripping dan hash final dicatat terpisah.
- Laporan menyimpan config, input/model SHA-256, kurva, jejak effective batch per epoch, pipeline augmentasi, environment/perangkat, AP tiap komponen, precision/recall/F1 per kelas dan confusion dua arah. Tidak ada rata-rata mAP antar-model atau promosi otomatis.
- Selektor menolak perbandingan dengan dataset, baseline, seed atau batch/ukuran input yang berbeda, serta kandidat dengan regresi/dukungan yang tidak cukup. Pemeriksaan kestabilan membutuhkan seed17/42/73 dengan config dan input yang sama.
- Utility `evaluate_fixed` tetap dapat diimpor oleh skrip kalibrasi lama; report eksperimen baru menggunakan format validation-only dan tidak memakai `compare_broad_candidate.py` yang merupakan alat evaluasi historis.

## Rencana dan status aktual

[Rencana B0–B3](../evaluations/issue10-experiment-plan.json) memuat delapan entri untuk dua model, yaitu dua evaluasi B0 dan enam training run B1/B2/B3. Preflight rangkaian ini menunggu dataset final dengan bukti build. Pilot di bawah memakai jalur eksperimen lokal dan validation legacy.

| Cabang | B0 | B1 | B2 | B3 | Seed tambahan |
|---|---|---|---|---|---|
| Broad Waste | Ditunda | Ditunda | Ditunda | Menunggu data dan seleksi B1/B2 | Menunggu kandidat |
| General COCO11s | Ditunda | Ditunda | Ditunda | Menunggu data dan seleksi B1/B2 | Menunggu kandidat |

## Hasil pilot validation legacy

Setiap run memakai RTX 4070 Laptop, Ultralytics 8.4.183, square640, seed 42, batch 8, nominal batch 64, freeze 10, dan empat epoch. AdamW memakai laju awal 0,0001 dan warmup satu epoch. Bobot awal cocok dengan SHA-256 registry. Validation per cabang memakai gambar dan label yang sama pada kedua sisi perbandingan.

| Run | Gambar train / validation | mAP50–95 bobot awal | mAP50–95 kandidat | Epoch checkpoint |
|---|---:|---:|---:|---:|
| Broad Waste legacy | 943 / 975 | 26,10% | 23,21% | 3 |
| General legacy | 535 / 43 | 50,97% | 39,21% | 1 |
| General dengan tambahan COCO | 821 / 43 | 50,97% | 37,15% | 1 |

Run tambahan general memakai 286 foto COCO train2017 yang memiliki bbox penerbit dan atribusi terverifikasi. Foto tambahan hanya dimasukkan ke train. Semua checksum gambar dan label validation cocok dengan general legacy. F1 ponsel pada confidence 0,50 bergerak dari 59,26% pada bobot awal menjadi 19,05% pada pilot legacy dan 45,16% pada pilot tambahan COCO. F1 remote masing-masing 36,36%, 33,33%, dan 36,36%. Validation general memuat 19 objek ponsel dan 9 remote.

F1 kardus Broad pada confidence 0,50 naik dari 37,36% ke 41,38%. F1 wrapper turun dari 7,41% ke 6,67%. Validation hanya memiliki lima objek kertas dan dua karton minuman. mAP50–95 masing-masing kandidat memberi alasan untuk memilih baseline pada pilot ini.

Semua kurva, checksum dan bobot `best.pt` disimpan di `artifacts/training/issue10-pilot-*-20261120/`. Laporan mencatat kandidat sebagai eksperimen validation lama. [Hasil dan checksum ketiga run](../evaluations/issue10-training-pilots.json).

## Verifikasi

Tes menggunakan dataset sintetis, checkpoint dummy dan model tiruan; tidak ada GPU training/inference atau emulator dijalankan. Tes mencakup config salah, dataset/bobot berubah, proof stale, test.txt tidak dapat dibaca, data holdout tidak tersedia, konfigurasi Ultralytics tanpa test, serta kasus CSV memiliki puncak pada epoch berbeda dari checkpoint yang benar-benar dipilih trainer. Alur yang sama juga diuji dengan 80 class ID COCO.

Tes seleksi mencakup perbandingan yang tidak sebanding, metrik test/nonfinite, regresi kontrol/fokus, satu kandidat maksimum, serta kelengkapan/config seed stability. Suite Linux pada CI juga memeriksa deployment recovery, kontrak, dependency dan artifact seperti workflow repo.

## Penerimaan yang masih tertunda

Setelah review data selesai, jalankan builder dan verifier baru, B0/B1/B2, seleksi B1/B2, lalu B3 bila valid. Pilih paling banyak satu kandidat per komponen, lakukan uji seed dan augmentasi tambahan yang diperlukan, dan catat semua hasil/biaya. Evaluasi final holdout tetap menjadi tahap terpisah pada issue #12.

Issue #10 tetap terbuka sampai run serta laporan penerimaan nyata tersedia. Lihat [panduan eksperimen](CONTROLLED_TRAINING_GUIDE.md) untuk CLI dan urutan lengkap.
