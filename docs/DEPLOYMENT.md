# Deployment

This repository deploys only the AI workers. The Go backend and PostgreSQL are deployed from the backend repository. An AI deployment never touches backend releases, migrates databases, or changes Nginx routing.

## Host prerequisites

The staging host needs the following before the first deployment.

| Component | Requirement |
|---|---|
| Runtime | A Python 3.12 CPU virtual environment at `/opt/wastenot-ai/venv`, installed with `--require-hashes -r requirements-cpu.lock`. The reviewed lock fingerprint is stored root-only in `/etc/wastenot/ai-requirements.sha256` |
| Models | `/opt/wastenot-ai/models` holds every checkpoint that `model.json` pins. Set `AI_MODEL_DIRECTORY=/opt/wastenot-ai/models` so each release loads the bundle its manifest names. The legacy single-model `AI_MODEL_PATH` and `AI_MODEL_SHA256` are still supported |
| Services | The `wastenot-yolo` and `wastenot-decisions` OS accounts, the hardened units from [`deploy/systemd`](../deploy/systemd), and a valid initial release at `/opt/wastenot-ai/current`. The deploy helper refuses a first install without a recovery release |
| Secrets | `/etc/wastenot/ai-detection.env` and `/etc/wastenot/ai-recommendation.env`, owned by `root:root` with mode `0600`. The two service tokens must differ, and Gemini keys belong only in the recommendation file |
| Guides | The operator's guide policy file lives outside the release directory, is read-only, and is never overwritten by CD. Draft guides in this repository are never published automatically |

## CI

On every pull request and push to `main`:

1. Install runtime and CI tools from `requirements-ci.lock` with hashes.
2. Run Ruff lint and format checks.
3. Audit the CPU lock with pip-audit.
4. Run the contract generator in `--check` mode and verify the 34-artifact pin.
5. Run pytest, including rollback and corrupt-archive cases.
6. Download the real model weights and verify their SHA-256.

CI never calls the paid Gemini API or a camera.

- **Actions.** All actions are pinned to commit SHAs, and permissions are limited to `contents: read`.
- **Concurrency.** Runs on `main` are never cancelled mid-deploy, while superseded pull request runs may be.
- **Host lock.** The host also holds a file lock so two deployments never activate at once.

## Release and recovery

CD runs on the self-hosted `alberd-staging-ai` runner only after CI succeeds on `main`.

1. **Package.** `scripts/package_release.py` packages:
   - `src/wastenot_ai` and the entrypoint;
   - the model manifest;
   - the CPU lock;
   - the contract pin.

   `release.json` binds the full commit SHA and the SHA-256 of every file. Keys, environment files, models, training output, photos, and datasets are never packaged.
2. **Install the helper.** The workflow installs `deploy/deploy.py` as `/usr/local/sbin/wastenot-ai-deploy`, owned by `root:root`, and streams the archive to it on stdin.
3. **Validate the archive.** The helper validates compressed and expanded size limits and accepts only regular files. It rejects path traversal, symlinks, duplicates, missing files, and checksum mismatches. A commit that already exists with different content is rejected, and a release is never extracted over an existing one.
4. **Check the environment.** Before activation, these must all match the release:
   - the lock fingerprint;
   - the installed package versions;
   - Python 3.12;
   - the model checksums.

   If the lock changed, the deployment stops and the old services keep running. Dependencies are never installed automatically on the host.
5. **Activate.** The release is staged, then moved to `/opt/wastenot-ai/releases/<sha>`, and the `current` symlink is swapped atomically. A systemd drop-in points `ExecStart` at `current/serve.py` and `PYTHONPATH` at `current/src`, keeping the existing unit hardening.
6. **Verify or roll back.** Both workers restart, and their authenticated health checks must report the new commit in `X-WasteNot-Release`. On failure, the previous source and drop-in are restored, the workers restart, and recovery readiness is verified. Restarts cause a short gap; this is not a zero-downtime deployment.
7. **Keep history.** On success, the `previous` symlink is updated. Old releases are never modified and are kept for inspection and rollback. Prune them according to the host's disk capacity.

## Resource limits

| Worker | Listen address | Memory | CPU | Network |
|---|---|---|---|---|
| Detection | `127.0.0.1:8091` | 1280 MiB | 75%, one thread | No egress except localhost |
| Recommendation | `127.0.0.1:8093` | 256 MiB | 25% | HTTPS egress to Gemini |

These limits constrain the workers; they do not protect the whole host from overload.

## Changing dependencies or models

**Dependency upgrades** are a separate operator rollout:

1. Build a staging virtual environment from the new hashed lock.
2. Check the packages, the CPU build, the model, and worker startup on a staging port.
3. Stop the workers for the planned switch.
4. Replace the virtual environment and the fingerprint marker together, as one reversible change.

Keep the previous environment and marker. A failed source activation does not roll back an environment upgrade made outside the helper. Never edit the marker just to pass the gate.

**Model changes** must ship weights, the manifest, checksums, and a matching evaluation. Never promote a model on validation metrics alone.

**Rollbacks** redeploy the artifact from the earlier commit through the same helper, so checksums, runtime, and readiness are checked again. If dependencies differ, restore the matching environment and model first.

## Pilot model bundle

`model.json` selects `broad-coco-b0f633193403`: a square 640 px input with 88 merged labels.

- **Downloading weights.** The waste checkpoint and its dataset attribution are attached to the GitHub release `models-v4-coco11s-20261107`. The COCO11s companion comes from the official Ultralytics release. `make model` (`scripts/fetch_model.py`) downloads both and verifies their hashes.
- **Access token.** Private release assets need a `GITHUB_TOKEN` with read access. The token is sent only to the GitHub API over HTTPS and is stripped on cross-origin redirects.
- **Installing on the host.** Before merging a manifest change, install both files on the host as root-owned and read-only. Keep every checkpoint that a recovery release still needs.

On the two-vCPU staging host (one thread, 75% CPU quota, 1280 MiB memory), the detection worker measured:

- **Warm inference.** 6.4–6.6 seconds per photo, with a peak RSS of about 570 MiB.
- **First call.** About 15 seconds. The entrypoint therefore warms up both models before accepting requests.

These were exploratory measurements; parallel capacity still needs load testing.

## Operations

```sh
readlink -f /opt/wastenot-ai/current
systemctl show wastenot-yolo wastenot-decisions -p ExecStart -p ActiveState
```

A healthy process does not prove that the Gemini key or model is available, or that the guides are correct. After any relevant operational change, run a smoke test through the Go API.
