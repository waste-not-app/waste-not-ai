# Archived Reports

Historical research notes, experiment reports, and review logs from the pilot phase. They are kept as an audit trail and are not maintained. Most are written in Indonesian. Figures, paths, and statuses reflect the date of each report, not the current service. Paths prefixed with `ai/` refer to the layout before the AI code moved out of the Android repository.

For current behavior, see the [README](../../README.md), the [HTTP contract](../AI_HTTP_CONTRACT.md), [integration](../INTEGRATION.md), and [deployment](../DEPLOYMENT.md).

The trainer commands in these historical reports use the interface from revision [feb61aa](https://github.com/waste-not-app/waste-not-ai/tree/feb61aa). Use the [controlled training guide](../CONTROLLED_TRAINING_GUIDE.md) for the current experiment CLI.

## Datasets

| Report | Topic |
|---|---|
| [DATASET.md](DATASET.md) | Pilot dataset and evaluation plan |
| [DATASET_RESEARCH.md](DATASET_RESEARCH.md) | Survey of public object detection datasets |
| [BROADER_DATASET_RESEARCH.md](BROADER_DATASET_RESEARCH.md) | Options for broadening object coverage |
| [DATASET_V3.md](DATASET_V3.md) | Expanding and rebalancing the v3 dataset |
| [DATASET_V4_FOCUS.md](DATASET_V4_FOCUS.md) | Cardboard- and phone-focused v4 dataset |
| [EXTERNAL_DATASET_AUDIT_V5.md](EXTERNAL_DATASET_AUDIT_V5.md) | FPI-Det and RealWaste licensing and quality audit |

## Detection models

| Report | Topic |
|---|---|
| [BASELINE.md](BASELINE.md) | Prototype baseline measurements |
| [FINE_TUNING.md](FINE_TUNING.md) | YOLO fine-tuning procedure |
| [FINETUNING_RESULTS.md](FINETUNING_RESULTS.md) | First fine-tuning run results |
| [BROAD_WASTE_TRAINING.md](BROAD_WASTE_TRAINING.md) | Training on a broader set of waste objects |
| [BROAD_WASTE_V3_RESULTS.md](BROAD_WASTE_V3_RESULTS.md) | Broad v3 candidate results |
| [FOCUS_V4_RESULTS.md](FOCUS_V4_RESULTS.md) | Cardboard and phone focus training results |
| [V4_CALIBRATION_AND_COCO11S.md](V4_CALIBRATION_AND_COCO11S.md) | Cardboard threshold calibration and the COCO11s companion model |
| [DETECTION_QUALITY.md](DETECTION_QUALITY.md) | Per-class thresholds and review bands |
| [CAMERA_BENCHMARK.md](CAMERA_BENCHMARK.md) | Camera replay benchmark v1 |
| [CAMERA_COVERAGE.md](CAMERA_COVERAGE.md) | Diagnosis of missing phone detections |
| [CONDITIONAL_AND_BACKGROUND.md](CONDITIONAL_AND_BACKGROUND.md) | Conditional advice for unknown conditions and background false positives |

## Recommendations and guides

| Report | Topic |
|---|---|
| [RECOMMENDATION_CONTRACT.md](RECOMMENDATION_CONTRACT.md) | Four-method recommendation contract |
| [RECOMMENDATION_EVALUATION.md](RECOMMENDATION_EVALUATION.md) | Pilot guides and recommendation evaluation |
| [DIRECT_RECOMMENDATIONS.md](DIRECT_RECOMMENDATIONS.md) | Rule-based recommendations without the LLM |
| [GUIDE_QUALITY.md](GUIDE_QUALITY.md) | Guide quality and context evaluation |
| [HANDLING_GUIDANCE.md](HANDLING_GUIDANCE.md) | Baseline handling guidance per catalog item |
| [PILOT_GUIDANCE_RESEARCH.md](PILOT_GUIDANCE_RESEARCH.md) | Primary source research for pilot guides |

## Reviews and handoffs

| Report | Topic |
|---|---|
| [HANDOFF.md](HANDOFF.md) | Android, Go, and AI integration handoff |
| [REVIEW_20261021.md](REVIEW_20261021.md) | Pre-merge review, 21 October 2026 |
| [INTEGRATION_20261107.md](INTEGRATION_20261107.md) | Integration review, 7 November 2026 |
| [DEBUG_20261107.md](DEBUG_20261107.md) | Bug audit, 7 November 2026 |
