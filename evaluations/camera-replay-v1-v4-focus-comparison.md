# Benchmark replay kamera v1

Tanggal UTC 2026-11-07T09:33:07.898618+00:00

117 foto dasar, 2 profil input, 3 pengulangan waktu per input/model.

Matching IoU 0,50 pada kandidat yang bisa dipilih di demo. Precision/recall hanya menghitung kelas yang dianotasi oleh sumber pada setiap foto.

| Profil | Model | Precision | Recall | F1 | TP | FP | FN | Median ms | P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| browser_jpeg90 | v4 | 67.6% | 32.0% | 43.4% | 71 | 34 | 151 | 104.2 | 143.3 |
| browser_jpeg90 | coco | 75.0% | 16.2% | 26.7% | 36 | 12 | 186 | 48.6 | 72.0 |
| browser_jpeg90 | combined | 65.6% | 37.8% | 48.0% | 84 | 44 | 138 | 153.9 | 203.4 |
| simulated_dim055 | v4 | 68.8% | 28.8% | 40.6% | 64 | 29 | 158 | 105.2 | 149.7 |
| simulated_dim055 | coco | 80.0% | 16.2% | 27.0% | 36 | 9 | 186 | 50.4 | 71.7 |
| simulated_dim055 | combined | 70.1% | 36.9% | 48.4% | 82 | 35 | 140 | 156.3 | 208.7 |

## Per kelas pada browser JPEG90

| Kelas | GT | v4 P / R | coco P / R | combined P / R |
|---|---:|---:|---:|---:|
| bottle | 45 | 78.1% / 55.6% | 71.4% / 22.2% | 73.0% / 60.0% |
| can | 19 | 70.6% / 63.2% | — / 0.0% | 70.6% / 63.2% |
| cup | 29 | 80.0% / 69.0% | 88.2% / 51.7% | 77.8% / 72.4% |
| plastic_bag | 15 | 60.0% / 20.0% | — / 0.0% | 60.0% / 20.0% |
| paper | 4 | — / 0.0% | — / 0.0% | — / 0.0% |
| cardboard | 15 | 54.5% / 40.0% | — / 0.0% | 50.0% / 33.3% |
| drink_carton | 5 | 0.0% / 0.0% | — / 0.0% | 0.0% / 0.0% |
| plastic_wrapper | 17 | 12.5% / 5.9% | — / 0.0% | 12.5% / 5.9% |
| food_container | 7 | 60.0% / 42.9% | — / 0.0% | 60.0% / 42.9% |
| bottle_cap | 6 | 100.0% / 16.7% | — / 0.0% | 100.0% / 16.7% |
| cell phone | 23 | — / 0.0% | 60.0% / 26.1% | 60.0% / 26.1% |
| remote | 37 | — / 0.0% | 71.4% / 13.5% | 71.4% / 13.5% |

## Diagnosis pada browser JPEG90

| Model | Salah label pada lokasi GT | GT tanpa kandidat bertumpuk | Pasangan kotak kelas sama | Pasangan label bertentangan |
|---|---:|---:|---:|---:|
| v4 | 18 | 134 | 1 | 4 |
| coco | 12 | 175 | 1 | 2 |
| combined | 22 | 120 | 0 | 18 |

## Pasangan salah label yang paling sering

| Model | Ground truth | Prediksi | Jumlah |
|---|---|---|---:|
| v4 | paper | plastic_wrapper | 3 |
| v4 | remote | can | 3 |
| v4 | bottle | can | 1 |
| v4 | bottle | cup | 1 |
| v4 | bottle_cap | cup | 1 |
| coco | can | kite | 3 |
| coco | remote | cell phone | 2 |
| coco | bottle | book | 1 |
| coco | bottle | surfboard | 1 |
| coco | bottle_cap | frisbee | 1 |
| combined | paper | plastic_wrapper | 3 |
| combined | remote | cell phone | 2 |
| combined | bottle | book | 1 |
| combined | bottle | can | 1 |
| combined | bottle | cup | 1 |

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
