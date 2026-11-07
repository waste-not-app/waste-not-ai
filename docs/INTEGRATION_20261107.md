# AI integration review â€” 7 November 2026

Base: `5405e150ceab04d9154d44fb1bbb52256cbfde41` in the standalone AI repository. The local monorepo AI subtree matched the extracted base `6d30abb` exactly. Integration preserves remote deployment, runtime locks, model manifest/checksum, provider timeout handling, hosting prefix, release readiness headers and origin middleware.

## Standards

Review of the camera, recommendation and dataset changes found no blocking documented privacy or runtime rule breach. One nonblocking maintenance observation remains: `eligible_evidence`, `missing_context` and `conditional_evidence` assemble similar context dictionaries in `recommendation.py` (possible Duplicated Code). This is a judgment call, not a safety failure.

Total: zero documented violations; one nonblocking maintenance observation.

## Spec

One integration finding was resolved: the Broad Waste trainer still published its final report beneath the former `ai/evaluations` directory. Training scripts and launchers now use the standalone root. Frozen benchmark paths retain their original bytes/checksum; a containment-checked resolver supports the legacy `ai/` prefix. Regression coverage verifies both checksum preservation and rejection of escaping paths.

Sourced initial guidance, sharp-glass handling, combined detector conflicts, rejected threshold isolation, noncommercial research dataset scope and production model preservation match the requested behavior. No pending blocking spec finding.

Total: one resolved integration finding; zero unresolved blockers.

## Verification before pull request

- Python 3.12 environment installed using pinned CI package hashes; Windows additionally needs pytest's platform dependency `colorama`. `pip check` passes.
- 292 portable Python tests pass. Linux deployment tests require `fcntl` and run in GitHub's Ubuntu job.
- 34 generated contracts match implementation and the refreshed backend integration pin.
- 44 fixture recommendation scenarios and 25 browser checks with real YOLO pass; no live Gemini requests are used. Browser coverage includes fake camera capture, risk handling, cardboard correction, late-response discard, guidance fallback and mobile layout.
- Critical Ruff checks, locked CPU dependency advisory audit, wheel resource/exclusion verification and actual production model SHA-256 verification pass.
- Raw datasets, source photos, trained weights, local environment files, caches and virtual environments are excluded from Git. Research annotation attribution and restrictions are documented in `annotations/README.md`.

Research checkpoints remain explicit experiments. Production retains the pinned COCO baseline. Physical-camera diversity, independent annotation review and live provider behavior are outside these automated checks.
