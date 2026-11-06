# Integrasi Android, Go dan AI

Diperiksa 7 November 2026. Android memakai `https://pred-autoscaler.bccdev.id/wastenot/api/v1/`. Handoff utama ada pada [docs/ANDROID.md backend](https://github.com/waste-not-app/waste-not-backend/blob/main/docs/ANDROID.md); source mobile yang ditinjau berada pada [PR #102](https://github.com/waste-not-app/waste-not-android/pull/102), bukan main/rilis yang sudah diterima.

## Pemilik tanggung jawab

| Lapisan | Tanggung jawab |
|---|---|
| Android | Kamera satu foto, orientasi/preview bbox, konfirmasi objek/material/kondisi, tampilan status/sumber, UUID operasi dan save eksplisit |
| Go | JWT/sesi pengguna, validasi media/schema, quota, ownership, TTL, dedup durable, provenance, transaksi log dan XP/statistik |
| YOLO | Inferensi CPU, model/checksum, raw class/bbox, band/threshold dan mapping kandidat |
| Keputusan | Policy/panduan yang tersedia, validasi konteks/evidence, panggilan Gemini dan satu metode atau hasil non-READY |

Port 8091/8093 privat pada loopback, memakai token layanan berbeda. Android tidak menerima token/key tersebut. Gemini tidak dipanggil oleh Go. Model mendeteksi bentuk/kelas, bukan memastikan material dari foto.

## Kontrak dan perubahan pasangan

`contracts/v1/public.openapi.json` adalah kontrak yang dikonsumsi mobile. Dua OpenAPI privat adalah kontrak Go–Python. Snapshot yang disepakati disimpan pada `contracts/ai/` di backend. [backend-pin.json](../contracts/backend-pin.json) dan `contracts/ai.lock.json` backend mengunci 34 artifact yang identik.

CI tidak meregenerasi kontrak untuk membuat tes lolos. Jalankan ekspor hanya ketika memang mengubah kontrak, tinjau diff, sinkronkan snapshot/pin kedua repo dan periksa keduanya sebelum merge:

```sh
PYTHONPATH=src python -m wastenot_ai.cli export-contracts --check
python scripts/check_contracts.py contracts/v1 contracts/backend-pin.json \
  --peer ../waste-not-backend/contracts/ai
```

`--check` memakai dependency dari `requirements-ci.lock`, yang dibatasi oleh lock runtime. CI setiap repo memeriksa pin lokal; akses langsung antar repo privat tidak diasumsikan. Perubahan pin harus melalui perubahan pasangan, bukan hanya menaikkan checksum di satu repo. Perubahan schema membutuhkan pembaruan DTO/fixtures/tes Android yang relevan. Contoh fixture tetap simulasi dan tidak boleh menjadi fallback sukses produksi.

## Alur dan hasil yang perlu dibedakan

1. GET `metadata` dan `ai/objects` melalui Go dengan bearer pengguna. Patuhi capability dan guide status.
2. POST `ai/detections`, multipart `image` JPEG/PNG + `client_request_id` UUID. Tidak ada base64, video streaming atau penyimpanan foto permanen. Batas 5 MiB/12 MP/sisi4096. Gunakan band dan threshold server.
3. Pengguna mengonfirmasi object key/category/material dan konteks independen. Kirim `input_source`, detection/object ID bila berasal dari scan, `confirmed_object=true` dan `context_revision`. Jangan menyuntikkan prompt/model/provider/raw class/confidence dari client.
4. POST `ai/recommendations`: READY dapat membawa satu metode HANDLING atau PREVENTION; NEEDS_CONTEXT/UNAVAILABLE tidak boleh dianggap rekomendasi siap. CONDITIONAL menyimpan kondisi yang belum dibuktikan. Sumber/caution tetap terlihat. Tidak ada aktivitas atau XP pada tahap ini.
5. Save dilakukan lewat POST `waste-logs` setelah pengguna mengonfirmasi metode dan jumlah. Provenance READY/HANDLING memakai recommendation ID dan dua flag konfirmasi; Reduce pencegahan/kategori baru memakai pencatatan manual. Hapus sesi sementara melalui DELETE `ai/detections/{id}`.

UUID yang sama selalu berarti operasi yang sama. Catatan dengan hasil tidak diketahui direplay memakai payload/UUID awal agar tidak menggandakan data. AI mereplay hasil/error tersimpan tanpa panggilan provider; generasi baru setelah kegagalan memerlukan tindakan eksplisit dan UUID baru. Mengganti UUID otomatis atau mereplay refresh yang tidak diketahui adalah salah.

Provider17s → worker18s → Android read20s → Go header23s/total25s. Budget ini tidak menjamin respons pasti sampai sebelum Android timeout. Jangan menambah timeout satu lapisan atau meluncurkan retry generasi tersembunyi.

## Batas pilot

Model aktif tetap baseline COCO, bukan checkpoint kandidat fine-tuning. Empat policy guide versi6 (PET, HDPE, kaleng aluminium non-aerosol, kardus) diperiksa sumber otomatis, berlaku sampai 4 Desember2026 dan memerlukan konfirmasi penerima lokal. Tujuh puluh empat draf lain bukan panduan produksi. Artikel panduan umum Go memakai schema terpisah dan masih kosong.

Unit/schema/HTTP tests tidak membuktikan akurasi material, independent holdout, keselamatan manusia, kapasitas/SLA provider, kamera perangkat atau penerimaan visual. Uji Android minimal API30 dan runtime terbaru; catatan TLS API24 tidak lagi menjadi baseline rilis.
