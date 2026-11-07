# Benchmark replay kamera v1

Tanggal UTC 2026-11-07T08:14:38.603384+00:00

117 foto dasar, 2 profil input, 3 pengulangan waktu per input/model.

Matching IoU 0,50 pada kandidat yang bisa dipilih di demo. Precision/recall hanya menghitung kelas yang dianotasi oleh sumber pada setiap foto.

| Profil | Model | Precision | Recall | F1 | TP | FP | FN | Median ms | P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| browser_jpeg90 | v3 | 59.0% | 26.6% | 36.6% | 59 | 41 | 163 | 177.1 | 215.5 |
| browser_jpeg90 | coco | 75.0% | 16.2% | 26.7% | 36 | 12 | 186 | 86.8 | 103.8 |
| browser_jpeg90 | combined | 60.7% | 33.3% | 43.0% | 74 | 48 | 148 | 266.2 | 329.9 |
| simulated_dim055 | v3 | 67.1% | 24.8% | 36.2% | 55 | 27 | 167 | 175.2 | 223.1 |
| simulated_dim055 | coco | 80.0% | 16.2% | 27.0% | 36 | 9 | 186 | 86.9 | 103.6 |
| simulated_dim055 | combined | 67.3% | 32.4% | 43.8% | 72 | 35 | 150 | 265.1 | 317.9 |

## Per kelas pada browser JPEG90

| Kelas | GT | V3 P / R | COCO P / R | Gabungan P / R |
|---|---:|---:|---:|---:|
| bottle | 45 | 71.0% / 48.9% | 71.4% / 22.2% | 69.4% / 55.6% |
| can | 19 | 81.2% / 68.4% | — / 0.0% | 81.2% / 68.4% |
| cup | 29 | 64.3% / 62.1% | 88.2% / 51.7% | 63.3% / 65.5% |
| plastic_bag | 15 | 0.0% / 0.0% | — / 0.0% | 0.0% / 0.0% |
| paper | 4 | 0.0% / 0.0% | — / 0.0% | 0.0% / 0.0% |
| cardboard | 15 | 66.7% / 13.3% | — / 0.0% | 66.7% / 13.3% |
| drink_carton | 5 | 0.0% / 0.0% | — / 0.0% | 0.0% / 0.0% |
| plastic_wrapper | 17 | 10.0% / 5.9% | — / 0.0% | 11.1% / 5.9% |
| food_container | 7 | 50.0% / 28.6% | — / 0.0% | 66.7% / 28.6% |
| bottle_cap | 6 | 50.0% / 16.7% | — / 0.0% | 50.0% / 16.7% |
| cell phone | 23 | — / 0.0% | 60.0% / 26.1% | 60.0% / 26.1% |
| remote | 37 | — / 0.0% | 71.4% / 13.5% | 71.4% / 13.5% |

## Diagnosis pada browser JPEG90

| Model | Salah label pada lokasi GT | GT tanpa kandidat bertumpuk | Pasangan kotak kelas sama | Pasangan label bertentangan |
|---|---:|---:|---:|---:|
| v3 | 17 | 146 | 2 | 2 |
| coco | 12 | 175 | 1 | 2 |
| combined | 22 | 130 | 0 | 14 |

## Pasangan salah label yang paling sering

| Model | Ground truth | Prediksi | Jumlah |
|---|---|---|---:|
| v3 | remote | can | 3 |
| v3 | bottle | cup | 2 |
| v3 | drink_carton | plastic_wrapper | 2 |
| v3 | plastic_bag | plastic_wrapper | 2 |
| v3 | cardboard | plastic_wrapper | 1 |
| coco | can | kite | 3 |
| coco | remote | cell phone | 2 |
| coco | bottle | book | 1 |
| coco | bottle | surfboard | 1 |
| coco | bottle_cap | frisbee | 1 |
| combined | bottle | cup | 2 |
| combined | drink_carton | plastic_wrapper | 2 |
| combined | plastic_bag | plastic_wrapper | 2 |
| combined | remote | cell phone | 2 |
| combined | bottle | book | 1 |

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
