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

## Reserved holdout

Prepare a separate manifest under the workspace with `usage` set to `FINAL_HOLDOUT` and the complete `COMBINED_LABELS` in `target_classes`. Each record follows the camera benchmark shape with unique `id`, workspace-relative `image`, `sha256`, normalized `truth` entries and `evaluated_classes`. Supply `split=test`, `holdout_eligible=true`, `group`, `session_id`, and the image `phash`. The `rights` object provides `status=RELEASE_ELIGIBLE`, `source_page`, `license_url`, and `attribution`. The `review` object provides `reviewer_type=human`, `reviewer`, `reviewed_at`, and `complete_in_scope=true`. Annotate genuine low light separately with `condition=real_low_light`. Do not count transformed copies or photos from the same object/session as independent evidence.

The holdout command checks both rebuilt datasets, the original Broad v4 and general-v1 manifests, and the old benchmark for protected image hashes, group/session IDs and perceptual matches. Retain the original manifests under `data/local` so the command can check previously observed tests. Verify that the source was not used in the original checkpoint experiments, and document any possible pretraining exposure. Reserve the manifest before opening it for evaluation. Pass the exact SHA-256 of the reserved manifest.

```powershell
python scripts/evaluate_tuning_holdout.py @components --freeze artifacts/calibration/frozen.json --holdout PATH --holdout-sha256 HASH --output artifacts/calibration/holdout-result.json
```

The local ledger under `artifacts/holdout-ledger` records exposure before inference. Keep it for future evaluation cycles. It rejects overlaps by SHA-256, object/session and perceptual hash, even when manifest filenames change. An interrupted run after the claim consumes that holdout. For work on multiple machines keep an authoritative ledger with the evaluation records.

The report compares the released baseline and frozen candidate on the same inputs using paired resampling of connected groups. It records class and condition metrics, an interval for the six-class macro-F1 change and CPU p95. The gates require at least +5 percentage points for the focus macro-F1, a positive lower interval bound, sufficient support, bounded class/condition regressions, no increase in either direction of phone/remote confusion, and CPU p95 at most 1.25 times the matched baseline. A score can establish improvement on the reviewed public population. Camera parity requires a separate measurement.

## Release evidence and rollback

A successful holdout produces `MODEL_GATES_PASSED_PENDING_RELEASE_INTEGRATION`. The optional policy in `CombinedDetector` is currently an experiment seam. Before activating it, configure both workers to resolve the same profile by bundle identity and test the actual detection-to-recommendation flow, including a candidate below the old confidence threshold and ambiguous overlapping detections. Contract parsing alone cannot establish that the recommendation worker accepts those selections.

After the model and integration gates pass, record weights and policy SHA-256, data notices and review receipts, experiment/holdout reports, known coverage limitations and the exact baseline bundle in the model card. Verify the runtime package excludes dataset photos and weights. Publish approved checkpoints as GitHub Release assets and bind their byte digests in the release manifest. Record the successful CI run and previous deployment version. Rollback restores the previous pinned weight pair and policy together, then verifies readiness, catalog/label count and an authenticated detection-to-recommendation request.

No eligible release candidate, final holdout result, model card measurement, or Release asset exists for this cycle yet. The baseline release remains available under `model.json`.
