"""Useful first steps when a specific handling method cannot yet be validated."""
COMMON = "https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables"
HAZARDOUS = "https://www.epa.gov/hw/household-hazardous-waste-hhw"
SHARP = "https://www.cornwall.gov.uk/rubbish-recycling-and-waste/recycling/recycling-items/broken-glass/"
BIOLOGICAL = "https://www.who.int/news-room/fact-sheets/detail/health-care-waste"
BATTERIES = "https://www.epa.gov/recycle/used-household-batteries"
ELECTRONICS = "https://www.epa.gov/recycle/electronics-donation-and-recycling"


def handling_guidance(context, item, *, risk):
    chemical = (context.risk_type == "CHEMICAL" or context.condition.soiling == "HAZARDOUS"
                or context.condition.former_contents == "CHEMICAL")
    if chemical and context.risk_type != "BIOLOGICAL":
        return {
            "reason": "Ada indikasi bahan berbahaya. Ikuti langkah penanganan sementara ini sambil mengonfirmasi jalur pengelola khusus.",
            "steps": ["Ikuti petunjuk pada label produk; pertahankan wadah dan label aslinya jika kondisinya aman.",
                      "Jangan campurkan isinya, tuangkan ke saluran air, atau masukkan ke sampah/daur ulang biasa.",
                      "Hubungi DLH atau pengelola limbah berbahaya setempat dengan menyebut jenis produk dan kondisi wadah; minta petunjuk pengemasan dan penyerahan."],
            "source_urls": [HAZARDOUS],
        }
    if context.risk_type == "BIOLOGICAL":
        return {
            "reason": "Barang dengan dugaan kontaminasi biologis memerlukan jalur khusus. Berikut langkah awal sebelum penyerahan.",
            "steps": ["Hindari kontak langsung dengan cairan tubuh, bahan terkontaminasi, dan bagian tajam.",
                      "Pisahkan dari aliran sampah umum sesuai arahan petugas; jangan mencoba memakai ulang atau mengolahnya sendiri.",
                      "Hubungi fasilitas kesehatan atau pengelola limbah medis yang berwenang untuk petunjuk wadah, pengumpulan, dan penyerahan.",
                      *(["Jika ada bahan kimia juga, jangan mencampur isinya atau menuangkannya ke saluran air; jelaskan kedua jenis risiko kepada pengelola."] if chemical else [])],
            "source_urls": [BIOLOGICAL, *([HAZARDOUS] if chemical else [])],
        }
    if context.risk_type == "ELECTRICAL" or context.category == "ELECTRONIC":
        return {
            "reason": "Barang elektronik atau baterai memerlukan pemeriksaan jenis dan penerima khusus. Berikut langkah awal yang bisa diikuti.",
            "steps": ["Periksa label/jenis perangkat dan baterai tanpa membongkar atau menusuk komponennya.",
                      "Jangan masukkan baterai litium atau perangkat yang mengandungnya ke tong sampah atau daur ulang umum.",
                      "Cari pengumpulan elektronik/baterai resmi atau program produsen; konfirmasi penerimaan dan petunjuk pengemasan untuk kondisi barangmu."],
            "source_urls": [BATTERIES, ELECTRONICS],
        }
    if context.risk_type == "SHARP" or (context.category == "GLASS" and context.condition.integrity == "DAMAGED"):
        return {
            "reason": "Ada risiko pecahan atau tepi tajam. Periksa kondisinya dan ikuti langkah penanganan berikut sebelum menentukan jalur akhirnya.",
            "steps": ["Jangan memakai kembali barang yang retak/pecah atau menyentuh tepi tajam dengan tangan langsung.",
                      "Untuk pecahan kaca, ikuti pengemasan pelindung dalam wadah kokoh dan beri label agar petugas mengetahui isinya.",
                      "Konfirmasikan jenis barang dan kondisi tajamnya kepada pengelola setempat sebelum menyerahkan; penerimaan dan jalur pembuangan mengikuti aturan lokal."],
            "source_urls": [SHARP, COMMON],
        }
    if risk:
        return {
            "reason": "Risikonya perlu diperjelas. Sambil memeriksa, ikuti langkah awal ini dan minta arahan pengelola yang sesuai.",
            "steps": ["Periksa label dan jenis risiko tanpa membongkar, mencampur isi, atau mencoba mengolah barang.",
                      "Jangan gunakan ulang atau campurkan barang berisiko dengan aliran sampah umum sebelum jenis bahayanya jelas.",
                      "Hubungi pengelola sampah atau DLH setempat dengan menjelaskan jenis barang serta risiko yang diketahui."],
            "source_urls": [HAZARDOUS],
        }
    material = {"PLASTIC": "plastik", "PAPER": "kertas/kardus", "GLASS": "kaca",
                "METAL": "logam", "ORGANIC": "bahan organik"}.get(context.category, "bahan barang")
    return {
        "reason": f"Pilihan metode untuk {item['label']} belum dapat dipastikan. Berikut arahan awal berdasarkan informasi yang tersedia.",
        "steps": [f"Periksa jenis {material}, kondisi, dan riwayat isi; bila belum diketahui, jangan menebak dari foto saja.",
                  "Pertimbangkan penggunaan ulang hanya jika fungsi, kebersihan, dan keamanannya memang sudah dipastikan.",
                  "Tanyakan kepada pengelola setempat apakah jenis dan kondisi barang ini diterima, lalu ikuti petunjuk pemilahan dan penyerahannya.",
                  "Jika ada tanda baterai/listrik, tepi tajam, bahan kimia, atau kontaminasi biologis, tandai jenis risikonya untuk mendapat arahan khusus."],
        "source_urls": [COMMON],
    }
