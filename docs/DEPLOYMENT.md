# Deployment AI pada cp-bcc

Hanya worker AI yang dikelola repo ini. Go blue-green dan PostgreSQL dikelola backend; deploy AI tidak menyalin source ke rilis Go, memigrasikan DB atau mengubah routing Nginx.

## Prasyarat host yang sudah diprovisi

- SSH `proxy.bccdev.id:11049`, alias operator `cp-bcc`; deployment memerlukan hak administrator untuk helper/systemd.
- Python3.12 CPU venv `/opt/wastenot-ai/venv`, dipasang dengan `--require-hashes -r requirements-cpu.lock` dan pemeriksaan dependency. Fingerprint yang ditinjau tersimpan root-only di `/etc/wastenot/ai-requirements.sha256`.
- Direktori model `/opt/wastenot-ai/models` memuat setiap checkpoint yang dipatok `model.json`. Isi `AI_MODEL_DIRECTORY=/opt/wastenot-ai/models` pada konfigurasi worker untuk memilih bundle sesuai manifest release. Runtime dan helper deployment memeriksa checksum setiap komponen serta identitas model gabungan. Konfigurasi lama dengan `AI_MODEL_PATH` dan `AI_MODEL_SHA256` tetap didukung untuk release satu model.
- Akun OS `wastenot-yolo` dan `wastenot-decisions`, unit hardened dari [deploy/systemd](../deploy/systemd/), serta rilis awal valid pada `/opt/wastenot-ai/current`. Helper menolak first install tanpa rilis pemulihan; provisioning host baru adalah langkah operator terpisah.
- `/etc/wastenot/ai-detection.env` dan `ai-recommendation.env` root:root0600. Dua token layanan berbeda; key/model Gemini hanya di file rekomendasi. Jangan menaruh key di repo, log atau parameter command.
- File policy guide operator di luar direktori release tetap read-only dan tidak ditimpa CD. Source draf di repo tidak otomatis diterbitkan.

## CI

Pada PR/main: dependency runtime dan alat CI dipasang dari `requirements-ci.lock` dengan hash; Ruff; audit versi CPU lock; generator kontrak `--check`; pin 34 artifact; pytest (termasuk rollback/arsip rusak); dan unduh/verifikasi SHA-256 bobot sebenarnya. Tidak menggunakan Gemini berbayar atau kamera di CI.

Actions dipin ke commit. Permissions hanya contents:read. Job verify15menit/deploy10menit; main tidak dibatalkan di tengah deploy, PR lama boleh dibatalkan. Deploy host juga mempunyai file lock agar dua operator/workflow tidak beraktivasi bersamaan.

## Rilis dan pemulihan

1. `scripts/package_release.py` mengemas source `src/wastenot_ai`, entrypoint, manifest model, CPU lock dan pin kontrak. `release.json` mengikat full source SHA serta SHA-256 setiap file. Key, environment, model, training, foto dan dataset tidak masuk paket.
2. Artifact yang telah diuji dikirim melalui SSH dengan known-hosts wajib. Helper terpasang root:root pada `/usr/local/sbin/wastenot-ai-deploy`; arsip dikirim lewat stdin, tanpa path home pengguna yang dihardcode.
3. Helper memvalidasi batas compressed/expanded, regular files, traversal/symlink/duplikasi, kelengkapan dan checksum. Commit yang sudah mempunyai isi berbeda ditolak; rilis tidak diekstrak di atas rilis lama.
4. Sebelum aktivasi, fingerprint lock, **versi paket yang benar-benar terpasang**, Python3.12 serta checksum bobot harus cocok. Lock berubah → deploy berhenti dengan layanan lama tetap hidup. Tidak ada instalasi dependency otomatis di VM.
5. Rilis distage lalu dipindah ke `/opt/wastenot-ai/releases/<sha>`. Symlink `current` diganti atomik. Drop-in systemd mengarahkan ExecStart ke `current/serve.py` dan PYTHONPATH ke `current/src`, dengan hardening unit lama tetap berlaku.
6. Kedua worker direstart; health terautentikasi wajib mengirim `X-WasteNot-Release` sesuai commit baru. Jika gagal, source dan drop-in sebelumnya dikembalikan, worker direstart dan readiness pemulihan diperiksa. Worker restart tetap mempunyai jeda singkat; ini bukan blue-green AI atau jaminan tanpa downtime.
7. Sukses memperbarui symlink `previous`; rilis lama tidak dimutasi. Rilis lama dipertahankan untuk inspeksi/rollback dan perlu retensi operator sesuai kapasitas disk.

YOLO loopback8091: MemoryMax1280MiB, CPUQuota75%, satu thread, egress ditolak kecuali localhost. Keputusan loopback8093: MemoryMax256MiB, CPUQuota25%, HTTPS keluar untuk Gemini diperlukan. Batas tersebut membatasi worker, bukan menjamin seluruh VM bebas overload.

## Mengubah dependency/model atau memulihkan

Perubahan dependency adalah rollout operator terpisah: buat venv staging dari lock baru dengan hash, periksa package/CPU/model dan startup worker pada port staging, hentikan worker selama pengalihan environment yang disengaja, lalu perbarui venv serta marker sebagai satu perubahan yang dapat dipulihkan. Simpan venv/marker sebelumnya; kegagalan aktivasi source tidak otomatis mengembalikan upgrade venv yang dikerjakan di luar helper. Jangan hanya mengganti marker agar gate lolos.

Perubahan model harus menyediakan bobot/manifest/checksum dan evaluasi yang sesuai; jangan mengganti model aktif hanya berdasarkan hasil validation training. Restore source sebelumnya memakai artifact dari commit tersebut melalui helper yang sama, sehingga checksum/runtime/readiness tetap diperiksa. Bila dependency berbeda, pulihkan environment/model yang cocok dahulu. Helper menolak paket lama pra-manifest; untuk migrasi pertama, rollback otomatis tetap dapat kembali ke jalur unit sebelumnya.

```sh
ssh cp-bcc 'readlink -f /opt/wastenot-ai/current; systemctl show wastenot-yolo wastenot-decisions -p ExecStart -p ActiveState'
# Kirim artifact yang telah diunduh dari run sukses:
cat wastenot-ai-<full-sha>.tar.gz | ssh cp-bcc 'sudo -n /usr/local/sbin/wastenot-ai-deploy'
```

Secrets GitHub: SSH_PRIVATE_KEY/SSH_KNOWN_HOSTS. Variables: VM_HOST/VM_PORT/VM_USER. Tidak memakai OAuth pribadi untuk membaca peer repo privat. Health proses tidak membuktikan key/model Google sedang tersedia atau kualitas panduan; lakukan smoke melalui Go setelah perubahan operasional yang relevan.

## Bundle pilot v4 dan COCO11s

Manifest memilih `broad-coco-b0f633193403`, square640, dengan 88 label gabungan. Checkpoint waste dan atribusi dataset tersedia pada GitHub Release `models-v4-coco11s-20261107`. Pendamping COCO11s berasal dari rilis resmi Ultralytics. `scripts/fetch_model.py` memverifikasi kedua hash. Unduhan aset release privat memerlukan `GITHUB_TOKEN` dengan akses baca repo. Token hanya dikirim ke API GitHub melalui HTTPS dan dihapus dari request redirect lintas origin. CI memakai token job dengan permission contents read.

Service demo yang sudah terpasang juga menerima drop-in release untuk menjalankan `serve.py demo` pada port8092 dengan `AI_MODEL_DIRECTORY`. Entrypoint membaca manifest dan menjalankan pemanasan yang sama seperti worker deteksi. Deployment menyimpan serta mengembalikan override demo sebelumnya bila aktivasi gagal. Mode panduan draf mengikuti pengaturan demo yang sudah tersedia.

Sebelum merge manifest, operator memasang kedua file model sebagai root-owned/read-only di direktori model. `AI_MODEL_DIRECTORY` memungkinkan pemilihan model mengikuti source release sehingga rollback ke manifest sebelumnya juga memilih checkpoint sebelumnya. Simpan checkpoint yang diperlukan release pemulihan.

Uji staging pada VM dua CPU menggunakan satu thread, kuota CPU75% dan batas memori1280MiB. Tiga foto publik menghasilkan waktu hangat 6,4–6,6 detik per foto dan peak RSS sekitar570MiB. Panggilan pertama sekitar15detik. Entrypoint menjalankan inferensi pemanasan kedua komponen sebelum server HTTP menerima request. Pengukuran ini mencakup sembilan inferensi eksploratif. Kapasitas paralel dan beban pengguna masih memerlukan pengujian terpisah.
