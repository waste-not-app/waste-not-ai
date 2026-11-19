# Benchmark replay kamera v1

Tanggal UTC 2026-11-07T09:53:17.143129+00:00

117 foto dasar, 2 profil input, 3 pengulangan waktu per input/model.

Matching IoU 0,50 pada kandidat yang bisa dipilih di demo. Precision/recall hanya menghitung kelas yang dianotasi oleh sumber pada setiap foto.

| Profil | Model | Precision | Recall | F1 | TP | FP | FN | Median ms | P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| browser_jpeg90 | v4 | 67.6% | 32.0% | 43.4% | 71 | 34 | 151 | 106.9 | 137.9 |
| browser_jpeg90 | coco11s | 77.5% | 24.8% | 37.5% | 55 | 16 | 167 | 106.3 | 143.2 |
| browser_jpeg90 | combined | 67.8% | 43.7% | 53.2% | 97 | 46 | 125 | 211.0 | 278.1 |
| simulated_dim055 | v4 | 68.8% | 28.8% | 40.6% | 64 | 29 | 158 | 105.0 | 143.6 |
| simulated_dim055 | coco11s | 78.8% | 23.4% | 36.1% | 52 | 14 | 170 | 107.8 | 141.9 |
| simulated_dim055 | combined | 69.5% | 41.0% | 51.6% | 91 | 40 | 131 | 213.1 | 272.8 |

## Per kelas pada browser JPEG90

| Kelas | GT | v4 P / R | coco11s P / R | combined P / R |
|---|---:|---:|---:|---:|
| bottle | 45 | 78.1% / 55.6% | 66.7% / 40.0% | 69.8% / 66.7% |
| can | 19 | 70.6% / 63.2% | n/a / 0.0% | 70.6% / 63.2% |
| cup | 29 | 80.0% / 69.0% | 94.1% / 55.2% | 80.8% / 72.4% |
| plastic_bag | 15 | 60.0% / 20.0% | n/a / 0.0% | 60.0% / 20.0% |
| paper | 4 | n/a / 0.0% | n/a / 0.0% | n/a / 0.0% |
| cardboard | 15 | 54.5% / 40.0% | n/a / 0.0% | 50.0% / 33.3% |
| drink_carton | 5 | 0.0% / 0.0% | n/a / 0.0% | 0.0% / 0.0% |
| plastic_wrapper | 17 | 12.5% / 5.9% | n/a / 0.0% | 12.5% / 5.9% |
| food_container | 7 | 60.0% / 42.9% | n/a / 0.0% | 60.0% / 42.9% |
| bottle_cap | 6 | 100.0% / 16.7% | n/a / 0.0% | 100.0% / 16.7% |
| cell phone | 23 | n/a / 0.0% | 80.0% / 52.2% | 80.0% / 52.2% |
| remote | 37 | n/a / 0.0% | 75.0% / 24.3% | 75.0% / 24.3% |

## Diagnosis pada browser JPEG90

| Model | Salah label pada lokasi GT | GT tanpa kandidat bertumpuk | Pasangan kotak kelas sama | Pasangan label bertentangan |
|---|---:|---:|---:|---:|
| v4 | 18 | 134 | 1 | 4 |
| coco11s | 13 | 155 | 1 | 3 |
| combined | 25 | 106 | 0 | 18 |

## Pasangan salah label yang paling sering

| Model | Ground truth | Prediksi | Jumlah |
|---|---|---|---:|
| v4 | paper | plastic_wrapper | 3 |
| v4 | remote | can | 3 |
| v4 | bottle | can | 1 |
| v4 | bottle | cup | 1 |
| v4 | bottle_cap | cup | 1 |
| coco11s | remote | cell phone | 3 |
| coco11s | can | bottle | 2 |
| coco11s | food_container | toilet | 2 |
| coco11s | plastic_bag | handbag | 2 |
| coco11s | bottle_cap | frisbee | 1 |
| combined | can | bottle | 3 |
| combined | paper | plastic_wrapper | 3 |
| combined | remote | cell phone | 3 |
| combined | food_container | toilet | 2 |
| combined | plastic_bag | handbag | 2 |

## Batas interpretasi

- Public replay approximates browser preprocessing; local physical-camera capture holdout is pending.
- COCO val may have informed pretrained model development; it is not a new unseen deployment population.
- Per-image evaluated_classes restrict scoring to classes covered by that source's annotations.
- DW frames share physical objects and sessions; image counts exceed independent object counts.
- TACO rare classes have limited support; rankings describe this frozen sample.
- Publisher boxes were screened on contact sheets; an independent human annotation review is pending.
- Metrics use operational review thresholds and IoU 0.50; these scores are precision/recall, rather than mAP.
- Predictions outside each image's annotation scope are excluded from precision/recall and retained for overlap diagnostics.
- Wrong-label diagnostics use independent class-agnostic geometric matching and can coexist with class-aware TP/FP/FN.
- Latency reflects this machine and detector path. Camera acquisition, browser upload and HTTP overhead are outside the timer.

Input redup merupakan transformasi sintetis dari foto yang sama. Jumlah foto independen tetap sama dengan jumlah foto dasar.

Kesalahan per foto dan keluaran semua model tersedia di JSON serta CSV pendamping.
