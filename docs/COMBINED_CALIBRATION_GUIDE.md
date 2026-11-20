# Combined detector calibration and final evaluation

Issue #12 adds an experimental calibration path for the Broad Waste and COCO detector pair. The current release configuration still uses its registered weights and 640 profile. A final result requires reviewed public data, the #10 component receipts and the #11 resolution report. Public images can qualify when their rights, annotation scope and independence are documented.

## Calibration on validation

1. Complete the reviewed data builds for #8 and #9. Keep public training candidates that were flagged as ineligible away from the final holdout. Preserve publisher attribution and all scoped annotations.
2. Complete #10 and #11. Use the validated component receipts, frozen weights and `validation-manifest.json` written by the resolution experiment.
3. Run the following from the repository root with the pinned research environment.

```powershell
$env:PYTHONPATH = 'src'
$components = @('--broad-data', 'PATH', '--broad-weights', 'PATH', '--broad-receipt', 'PATH',
                '--general-data', 'PATH', '--general-weights', 'PATH', '--general-receipt', 'PATH')
python scripts/calibrate_combined.py @components --resolution-report PATH --validation PATH --cache artifacts/calibration/combined-val.json --output artifacts/calibration/frozen.json
```

The cache stores native predictions at confidence 0.05, up to 50 per component, with native NMS at IoU 0.70. It binds the bytes of each input, validation manifest, checkpoint pair, label map, preprocessing, dependencies and code. AP remains a separate low-confidence validation measurement in #11. Reported recall and precision use IoU 0.50 and the operational review policy.

Threshold search tests 0.10–0.90 in steps of 0.05, then 0.01 around the best eligible threshold. Classes need at least 50 instances and 20 independent groups. A class threshold must improve F1 within the precision and recall tolerances. The runner compares same-class merge IoU 0.55, 0.65 and 0.75. The baseline uses 0.65 and the existing review thresholds. Both use identical label mapping and the maximum of 50 combined entries. Equal raw scores retain Broad priority and native order. Raw scores are kept as model scores and the diagnostic cache records their source. Overlapping different labels stay available for user confirmation.

If calibration fails to improve validation, the runner can still freeze an upstream checkpoint or resolution change with its default fusion. It keeps the released baseline when neither upstream configuration nor calibration provides a candidate.

## Legacy validation pilot

The November 20 pilot uses the released Broad Waste v4 and COCO11s checkpoints at square 640, the size selected by the [resolution pilot](ISSUE_11_RESOLUTION_PROGRESS.md). It covers 975 Broad and 43 General legacy validation photos. Both checkpoints, the validation photos and labels, and the transmitted browser JPEG90 bytes were checked against the corresponding inputs or their checksums. A fresh cache uses confidence 0.05 with native NMS IoU 0.70 and a 50-entry cap per component. Component AP comes from the separate confidence 0.001 measurement in the resolution report.

The pilot uncovered a normalized box with zero height after Ultralytics clipped it against the image edge. The detector now skips collapsed boxes and continues returning valid detections from the same frame. API bounding-box validation still requires positive area.

The support check counts distinct groups after joining near-duplicate images. None of the six focus classes clears both 50 instances and 20 groups. Cardboard has 51 instances from 10 groups. Only bottle, cup, and person meet both thresholds for the exploratory sweep.

| Same-class merge IoU | Overall F1 with control thresholds | Overall F1 with pilot thresholds | Focus macro-F1 |
|---|---:|---:|---:|
| 0.55 | 69.65% | 69.95% | 22.86% |
| 0.65 | 69.54% | 69.84% | 22.86% |
| 0.75 | 69.16% | 69.44% | 22.86% |

The exploratory sweep selected `bottle=0.46` and `cup=0.51` under the precision and recall constraints. The 0.55 combination gained 0.41 percentage points overall compared with the released control. Focus macro-F1 stayed at 22.86% across all trials. With the pilot thresholds, overlapping bottle/can hypotheses fell from 21 to 13, and phone/remote overlaps stayed at 10. Overlap counts describe candidate pairs and are separate from misclassifications matched to ground truth. The detector and cached evaluator agreed on 12 live control images, and the worker responses preserved the 88-label contract.

The pilot retains the released policy. A final calibration and one-time holdout evaluation still require the reviewed datasets and matched #10/#11 reports. The [pilot report](../evaluations/issue12-calibration-pilot.json) records the cache checksum, class support, threshold trials, and metrics for the three merge IoUs.

## Reserved holdout

Prepare a separate manifest under the workspace with `usage` set to `FINAL_HOLDOUT` and the complete `COMBINED_LABELS` in `target_classes`. Each record follows the camera benchmark shape with unique `id`, workspace-relative `image`, `sha256`, normalized `truth` entries and `evaluated_classes`. Supply `split=test`, `holdout_eligible=true`, `group`, `session_id`, and the image `phash`. The `rights` object provides `status=RELEASE_ELIGIBLE`, `source_page`, `license_url`, and `attribution`. The `review` object provides `reviewer_type=human`, `reviewer`, `reviewed_at`, and `complete_in_scope=true`. Annotate genuine low light separately with `condition=real_low_light`. Do not count transformed copies or photos from the same object/session as independent evidence.

The holdout command checks both rebuilt datasets, the original Broad v4 and general-v1 manifests, and the old benchmark for protected image hashes, group/session IDs and perceptual matches. Retain the original manifests under `data/local` so the command can check previously observed tests. Verify that the source was not used in the original checkpoint experiments, and document any possible pretraining exposure. Reserve the manifest before opening it for evaluation. Pass the exact SHA-256 of the reserved manifest.

```powershell
python scripts/evaluate_tuning_holdout.py @components --freeze artifacts/calibration/frozen.json --holdout PATH --holdout-sha256 HASH --output artifacts/calibration/holdout-result.json
```

The local ledger under `artifacts/holdout-ledger` records exposure before inference. Keep it for future evaluation cycles. It rejects overlaps by SHA-256, object/session and perceptual hash, even when manifest filenames change. An interrupted run after the claim consumes that holdout. For work on multiple machines keep an authoritative ledger with the evaluation records.

The report compares the released baseline and frozen candidate on the same inputs using paired resampling of connected groups. It records class and condition metrics, an interval for the six-class macro-F1 change and CPU p95. The gates require at least +5 percentage points for the focus macro-F1, a positive lower interval bound, sufficient support, bounded class/condition regressions, no increase in either direction of phone/remote confusion, and CPU p95 at most 1.25 times the matched baseline. A score can establish improvement on the reviewed public population. Camera parity requires a separate measurement.

## Release evidence and rollback

A successful holdout produces `MODEL_GATES_PASSED_PENDING_RELEASE_INTEGRATION`. `CombinedDetector` accepts an explicit policy and the recommendation app accepts the matching trusted `calibrated_model` artifact. The detection worker reports that policy's capabilities, and the recommendation worker matches both model version and SHA-256 before applying its thresholds. The in-process API test passes an ambiguous detection at confidence 0.40 through both workers and rejects a mismatched SHA. Before deploying actual weights, repeat this check across the real Go gateway and both workers with the verified candidate configuration. Production still needs release configuration and a rollback exercise.

After the model and integration gates pass, record weights and policy SHA-256, data notices and review receipts, experiment/holdout reports, known coverage limitations and the exact baseline bundle in the model card. Verify the runtime package excludes dataset photos and weights. Publish approved checkpoints as GitHub Release assets and bind their byte digests in the release manifest. Record the successful CI run and previous deployment version. Rollback restores the previous pinned weight pair and policy together, then verifies readiness, catalog/label count and an authenticated detection-to-recommendation request.

No eligible release candidate, final holdout result, model card measurement, or Release asset exists for this cycle yet. The baseline release remains available under `model.json`.
