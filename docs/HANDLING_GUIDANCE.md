# Panduan awal untuk setiap pilihan barang

7 November 2026, `pilot-policy-08`, katalog `demo-catalog-05`.

Kasus pengguna: botol kaca CLEAN/INTACT/reusable YES, reuse_intention=true, recycling_access=YES, risk_flag=true dan risk_type=SHARP sebelumnya menghasilkan UNAVAILABLE/RISK_NOT_COVERED tanpa langkah. Kombinasi kondisi utuh dengan risiko pecahan tidak memenuhi evidence yang khusus memerlukan DAMAGED; UI kemudian menampilkan “Saran belum tersedia” dan menyebut kontaminasi meskipun risikonya tajam.

Kombinasi tersebut sekarang menghasilkan READY/RECYCLE dengan peringatan khusus melalui evidence `glass-sharp-intact-recycle-v1`. Keutuhan utuh dan risiko tajam dapat terjadi bersamaan; aturan policy-07 yang memaksa klarifikasi keutuhan sudah diganti. Penerima harus menerima jenis dan kondisi tajam yang dilaporkan; akses UNKNOWN hanya memberi saran bersyarat, akses NO tidak memberi RECYCLE. Sistem mempertahankan kondisi pengguna dan tidak menawarkan REUSE biasa pada risiko tajam. Mengubah ke DAMAGED tetap dapat menghasilkan READY/TRASH melalui evidence kaca pecah yang sesuai. Risiko listrik, kimia atau biologis tidak memakai evidence residu kaca biasa.

Semua hasil non-READY kini mempunyai `steps`, `reason`, `source_urls` dan metadata sumber dari modul panduan awal yang terpisah dari evidence klasifikasi. Jalur chemical/riwayat kimia, biologis, elektronik/baterai, tajam, risiko belum jelas dan barang biasa mendapatkan arahan berbeda. Panduan umum tetap tersedia bila panduan spesifik tidak ada/tidak ditinjau atau provider gagal. Teks/URL model yang gagal validasi tidak dipakai sebagai sumber.

Katalog bertambah dari 78 menjadi 85 pilihan: `other_organic`, `other_plastic`, `other_paper`, `other_glass`, `other_metal`, `other_electronic`, `other_other`. Pilihan ini mendukung barang yang tidak disebut namanya di katalog. Pengguna memilih bahan yang diketahui atau “Barang lain / bahan belum diketahui”; ini tidak memperluas kemampuan deteksi YOLO. Template general_handling tidak memiliki evidence empat metode dan tidak digunakan untuk membuat klasifikasi palsu.

Status API tetap READY/NEEDS_CONTEXT/UNAVAILABLE. READY tetap mensyaratkan satu metode, evidence dan sumber yang lolos validasi. Pada hasil lain, metode/scope/basis tetap null, tetapi UI menampilkan “Panduan penanganan awal/khusus” atau “Lengkapi kondisi · panduan awal” beserta langkah. Kegagalan Gemini tidak ditampilkan seolah hasil Gemini berhasil. Client Go/Android perlu merender langkah dan sumber pada semua status, bukan hanya READY. Schema tidak ditambah field baru; contoh kontrak dan versi policy telah diregenerasi.

Sumber yang diperiksa: [EPA common recyclables](https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables), [EPA HHW](https://www.epa.gov/hw/household-hazardous-waste-hhw), [baterai](https://www.epa.gov/recycle/used-household-batteries), [elektronik](https://www.epa.gov/recycle/electronics-donation-and-recycling), [Cornwall broken glass](https://www.cornwall.gov.uk/rubbish-recycling-and-waste/recycling/recycling-items/broken-glass/), dan [WHO health-care waste](https://www.who.int/news-room/fact-sheets/detail/health-care-waste). Prinsip sumber luar negeri bukan jaminan rute/penerimaan fasilitas Indonesia. Sumber WHO ditambahkan ke registri; panduan klasifikasi bawaan tetap DRAFT dan tidak dipublikasikan otomatis.

Verifikasi policy-08: 226 tes Python lulus, 21.420 kombinasi kondisi tanpa pelanggaran invariant, 44/44 skenario fixture dan 34 artifact kontrak sesuai. Browser dengan YOLO nyata, fixture provider serta kamera Chromium simulasi lulus 25 pemeriksaan tanpa error JavaScript. Skenario screenshot, perubahan ke kaca pecah, barang tak bernama, koreksi manual kardus, HTTP 429 dan panduan setelah kuota diuji melalui formulir/endpoint nyata. Satu panggilan Gemini live pada konteks screenshot juga menghasilkan READY/RECYCLE dengan peringatan. Kamera fisik tidak diakses; foto tidak dikirim ke Gemini. [Rincian dan batas audit](DEBUG_20261107.md).

Reproduksi tes fokus:

```powershell
.\.venv-ai\Scripts\python.exe -m pytest tests/test_recommendation_guidance.py -q
.\.venv-ai\Scripts\python.exe scripts/verify_camera_demo.py --url http://127.0.0.1:8098 --artifact-dir artifacts/policy08-browser
```

Screenshot dan hasil browser berada di `artifacts/policy08-browser/`; kredensial dan foto pengguna tidak dicatat ke laporan. Lampiran screenshot pengguna tidak dijadikan data training. Batas 30 generation per jam demo tetap berlaku; permintaan yang diputuskan policy tanpa provider tidak menghabiskan batas tersebut. HTTP 429 generation menyertakan panduan dalam `data` dan UI merendernya. Jalankan pengujian penuh dengan instance demo segar. HTTP gagal karena sesi/media/request invalid atau koneksi server terputus tetap merupakan error layanan, bukan rekomendasi tervalidasi.
