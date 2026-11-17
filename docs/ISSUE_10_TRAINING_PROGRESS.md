# Issue #10 — tooling fine-tuning dan run yang ditunda

Status: **implementasi kode siap diuji/review; eksperimen model nyata belum dijalankan**. PR ditumpuk di atas implementasi issue #9. Review data issue #8/#9 tetap ditunda sampai akhir sesuai arahan pengguna.

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

[Rencana B0–B3](../evaluations/issue10-experiment-plan.json) memuat delapan entri untuk dua model: dua B0 evaluasi dan enam training run B1/B2/B3. Seluruh preflight tertunda karena dataset final dengan review/bukti build belum tersedia. Tidak ada dataset review-only yang dipakai training.

| Cabang | B0 | B1 | B2 | B3 | Seed tambahan |
|---|---|---|---|---|---|
| Broad Waste | Ditunda | Ditunda | Ditunda | Menunggu data dan seleksi B1/B2 | Menunggu kandidat |
| General COCO11s | Ditunda | Ditunda | Ditunda | Menunggu data dan seleksi B1/B2 | Menunggu kandidat |

Preflight pada dataset legacy yang sudah dipulihkan juga menolak memulai eksperimen karena belum memiliki readiness dataset baru. Ini sesuai penundaan review, bukan hasil training yang gagal. Baseline produksi tetap dipertahankan, dan belum ada kandidat nyata yang dinyatakan lebih baik/stabil.

## Verifikasi

Tes menggunakan dataset sintetis, checkpoint dummy dan model tiruan; tidak ada GPU training/inference atau emulator dijalankan. Tes mencakup config salah, dataset/bobot berubah, proof stale, test.txt tidak dapat dibaca, data holdout tidak tersedia, konfigurasi Ultralytics tanpa test, serta kasus CSV memiliki puncak pada epoch berbeda dari checkpoint yang benar-benar dipilih trainer. Alur yang sama juga diuji dengan 80 class ID COCO.

Tes seleksi mencakup perbandingan yang tidak sebanding, metrik test/nonfinite, regresi kontrol/fokus, satu kandidat maksimum, serta kelengkapan/config seed stability. Suite Linux pada CI juga memeriksa deployment recovery, kontrak, dependency dan artifact seperti workflow repo.

## Penerimaan yang masih tertunda

Setelah review data selesai, jalankan builder dan verifier baru, B0/B1/B2, seleksi B1/B2, lalu B3 bila valid. Pilih paling banyak satu kandidat per komponen, lakukan uji seed dan augmentasi tambahan yang diperlukan, dan catat semua hasil/biaya. Evaluasi final holdout tetap menjadi tahap terpisah pada issue #12.

Issue #10 tetap terbuka sampai run serta laporan penerimaan nyata tersedia. Lihat [panduan eksperimen](CONTROLLED_TRAINING_GUIDE.md) untuk CLI dan urutan lengkap.
