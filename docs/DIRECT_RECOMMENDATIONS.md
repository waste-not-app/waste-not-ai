# Saran langsung berbasis panduan

13 November 2026, `pilot-policy-09`.

Kasus gelas plastik bersih/utuh, layak tetapi tidak ingin dipakai ulang dan penerima daur ulang belum diperiksa sebelumnya berhenti pada panduan awal ketika Gemini gagal. Kini demo memilih **Recycle bersyarat** dengan syarat memeriksa penerima yang menerima jenis serta kondisi barang. Nilai UNKNOWN tidak diubah menjadi YES.

Demo memakai **Aturan panduan · langsung** sebagai pilihan awal. Pemilihan deterministik memakai evidence yang sama dengan Gemini, mengutamakan persyaratan yang sudah terkonfirmasi, lalu Reuse → Recycle → Trash. Reduce hanya tersedia untuk fokus PREVENTION yang dipilih pengguna. Tombol rekomendasi dan konfirmasi jenis barang tetap diperlukan; deteksi tidak langsung menyimpulkan material atau kondisi.

Jika pengguna memilih Gemini lalu layanan timeout, kehabisan kuota, belum dikonfigurasi atau mengeluarkan pilihan/evidence tidak valid, Python memilih ulang dari evidence yang sudah lolos. Respons READY menyebut `provider=rules` dan mempertahankan kode kegagalan pada `reason_code`. Ini bukan keberhasilan Gemini. Teks, langkah dan sumber tetap dirender dari evidence, tanpa mengambil prosa bebas provider. Batas 30 panggilan provider per jam pada demo tetap membatasi Gemini/simulasi; aturan langsung tidak menghabiskan batas tersebut dan tetap tersedia setelah batas tercapai.

`handling_routes.json` memperluas evidence **DRAFT bawaan** untuk 85 pilihan katalog: benda biasa memakai rute penggunaan ulang yang sesuai fungsi, penerima daur ulang terkonfirmasi, atau residu nonberbahaya setelah opsi penggunaan ulang/daur ulang tidak tersedia; organik memakai program pengomposan yang sesuai; elektronik memakai penerima khusus. Baterai dan elektronik yang belum diketahui jenisnya tidak memperoleh aturan penggunaan ulang perangkat generik. Evidence spesifik, misalnya kaca pecah dan sikat gigi, tetap didahulukan bila cocok.

Reuse memerlukan kelayakan, kebersihan, keutuhan dan niat penggunaan ulang. Recycle dengan penerima UNKNOWN adalah saran bersyarat, bukan klaim bahwa pengelola lokal menerima barang. Trash memerlukan syarat residu yang sesuai. Risiko kimia/biologis/tidak jelas, riwayat isi yang belum diketahui pada wadah tertentu, serta benda yang belum memiliki evidence cocok masih meminta klarifikasi atau panduan khusus; sistem tidak mengubahnya menjadi sampah biasa agar mendapat label.

Rute umum mengacu pada [hierarki limbah nonberbahaya EPA](https://www.epa.gov/smm/sustainable-materials-management-non-hazardous-materials-and-waste-management-hierarchy), [bahan daur ulang umum](https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables), [pengomposan](https://www.epa.gov/recycle/composting-home), dan [elektronik](https://www.epa.gov/recycle/electronics-donation-and-recycling). Sumber umum tidak membuktikan material dari foto atau penerimaan fasilitas Indonesia.

File panduan operator tidak ditambah evidence atau dipublikasikan otomatis. Worker produksi tetap memerlukan panduan PUBLISHED yang valid. Pada pilot VM hanya empat jenis yang diterbitkan: PET, HDPE, kaleng aluminium non-aerosol dan kardus. Fallback aturan produksi memakai evidence terbitan yang sama, tanpa melewati pemeriksaan masa berlaku, risiko, sumber atau prototype. Perluasan 85 pilihan berlaku pada demo draf dan belum menjadi publikasi produksi.

Verifikasi lokal: 317 tes Python portable, 44/44 skenario fixture, semua 85 pilihan dalam kondisi normal terkonfirmasi dengan penerima UNKNOWN menggunakan aturan langsung maupun Gemini offline, serta 25 pemeriksaan browser dengan detektor gabungan nyata dan kamera Chromium simulasi. Uji UI terpisah memeriksa Reuse/Recycle/Trash, kegagalan Gemini dan lebar layar 390 px tanpa error JavaScript. Pengujian integrasi Go + database terisolasi + detektor dan keputusan Python juga lolos; database/akun pengujian dibersihkan setelahnya. Ini pengujian perilaku aplikasi, bukan peningkatan bobot atau bukti akurasi deteksi lapangan. Kamera fisik dan foto pengguna tidak dipakai.

```powershell
python -m pytest tests/test_direct_handling.py tests/test_demo.py -q
python scripts/evaluate_recommendations.py
python scripts/verify_camera_demo.py --url http://127.0.0.1:8092 --image <foto-botol-publik> --artifact-dir artifacts/policy09-browser
```
