# WasteNot AI

**Private object detection and waste handling recommendations for the WasteNot platform.**

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-uvicorn-009688?logo=fastapi&logoColor=white)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-111F68)
![Gemini](https://img.shields.io/badge/LLM-Gemini-8E75B2?logo=googlegemini&logoColor=white)
![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)

WasteNot AI runs two private HTTP workers behind the [WasteNot Backend](https://github.com/waste-not-app/waste-not-backend):

- **Detection worker.** Finds candidate objects in a photo with YOLO on the CPU.
- **Recommendation worker.** Turns a user-confirmed object and its condition into exactly one handling method: Reduce, Reuse, Recycle, or Trash. Each method is grounded in a reviewed guide and validated against that guide's evidence.

The Android app never reaches these workers, their tokens, the model, or the Gemini keys. Every request passes through the authenticated Go API.

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Development](#development)
- [Local tools](#local-tools)
- [Project structure](#project-structure)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Data and licensing](#data-and-licensing)
- [Related repositories](#related-repositories)

## Features

### Detection

- **Model.** A YOLO11 waste model paired with a COCO11s companion model: 88 labels, square 640 px input.
- **Integrity checks.** The SHA-256 of every checkpoint is verified against `model.json` at startup. Weights are never baked into the image.
- **Image handling.** Images up to 5 MiB, 12 megapixels, and 4096 px per side. Orientation is corrected, EXIF is stripped, and images never touch the disk.
- **Confidence bands.** Per-class thresholds produce a band for each candidate. Low scores and classes outside the catalog fall back to manual selection, and every candidate still needs the user's confirmation.

### Recommendations

- **One method per item.** Each confirmed item gets exactly one method, with a reason, steps, cautions, and linked sources. Reduce is scoped as prevention; the other methods are scoped as handling of the current item.
- **Guide-driven decisions.** Allowed methods come from published, unexpired guides.
  - Missing conditions return `NEEDS_CONTEXT` without calling the LLM.
  - Risky items, such as broken glass or batteries, keep their method and add a warning.
- **Gemini integration.**
  - Structured output with a pool of up to five keys and cooldowns after rate limiting.
  - Only object metadata and guide evidence are sent. Photos, names, and account data are never sent.
  - When no key is configured, the worker falls back to the reviewed rules.
- **Grounded text.** Reasons and cautions are rendered from the validated evidence, never from free-form LLM prose.

### Contracts and tooling

- **Contracts.** OpenAPI contracts, an error catalog, and examples are generated from the source and pinned in the backend.
- **Mock server.** Implements the public API for frontend and backend development.
- **Browser demo.** Runs the full pipeline locally against a camera.
- **CLI.** Covers detection, recommendations, benchmarks, and evaluation.

## Architecture

```
Android ──▶ Go API ──private HTTP──▶ Detection worker       (YOLO, :8091)
                    └─private HTTP──▶ Recommendation worker  (guides + Gemini, :8093)
```

| Layer | Responsibility |
|---|---|
| Go API | JWT, ownership, quotas, sessions and TTL, deduplication, provenance, waste log storage |
| Detection worker | Image decoding, model checksum, bounding boxes, thresholds, class mapping |
| Recommendation worker | Guides, risk rules, allowed methods, Gemini calls, evidence validation |

Each worker binds to loopback, uses its own service token, and handles one job at a time. A busy worker returns `503` with `Retry-After`. See the [HTTP contract](docs/AI_HTTP_CONTRACT.md) for the full division of responsibilities.

## Getting started

### With Docker (recommended)

```sh
cp .env.example .env            # set AI_RECOMMENDATION_TOKEN (24+ chars) and, optionally, GEMINI_API_KEY
make up                         # recommendation worker on 127.0.0.1:8093
GITHUB_TOKEN=... make model     # download YOLO weights into artifacts/models (once)
make up PROFILES=detection      # add the detection worker on 127.0.0.1:8091
make logs
make down
```

One `Dockerfile` builds two targets:

- `recommendation`, a light image without torch;
- `detection`, with CPU torch and Ultralytics.

Both run as a nonroot user with a read-only filesystem and a healthcheck.

To connect a local backend, set `AI_ENABLED=true` in the backend's `.env`, along with the same `AI_SERVICE_TOKEN` and `AI_RECOMMENDATION_TOKEN` that the workers use. The backend development stack already points at `host.docker.internal:8091` and `:8093`.

### From source

Requirements: Python 3.12 on Linux, macOS, or WSL. The lock files target `x86_64-manylinux_2_28`.

```sh
make install                # .venv with hashed CI dependencies
make install-cpu            # add CPU inference dependencies (torch, Ultralytics)
make run-recommendation     # 127.0.0.1:8093
make run-detection          # 127.0.0.1:8091
```

## Configuration

See [`.env.example`](.env.example) for every variable.

| Variable | Description |
|---|---|
| `AI_RECOMMENDATION_TOKEN` | Bearer token from Go to the recommendation worker |
| `AI_SERVICE_TOKEN` | Bearer token from Go to the detection worker. Must differ from the recommendation token |
| `GEMINI_API_KEY`, `GEMINI_API_KEY_1` … `_5` | Gemini keys. Numbered keys take priority and rotate. Keys in one Google project share its quota |
| `GEMINI_MODEL` | A model that supports `generateContent` with structured output |
| `GEMINI_TIMEOUT_SECONDS` | Provider timeout, at most 17 seconds |
| `AI_MODEL_DIRECTORY` | Directory that holds the weights named in `model.json` |
| `AI_GUIDES_PATH` | Published guide bundle for the recommendation worker |
| `AI_BIND_HOST`, `AI_PORT` | Listen address. Defaults to `127.0.0.1` and the worker's own port |

Never commit real keys. `doctor` reports how many keys are configured, never their values.

## Development

| Target | Purpose |
|---|---|
| `make install` / `make install-cpu` | Install CI dependencies, or CPU inference dependencies |
| `make fmt` / `make fmt-check` | Format with Ruff |
| `make lint` | Run Ruff: pyflakes, bugbear, pyupgrade, and Bandit security rules |
| `make audit` | Run pip-audit against the CPU lock |
| `make sec` | Run `lint` and `audit` |
| `make contracts` | Verify the generated contracts and the backend pin |
| `make export-contracts` | Regenerate contracts after an intentional change |
| `make test` | Run pytest |
| `make check` | Local gate: format, lint, contracts, and tests |
| `make ci` | CI gate: `check` plus `audit` |
| `make docker-build` | Build both images |

The tests use a fake Gemini transport and a fixture detector, so they need no keys, network access, or model downloads.

Continuous integration:

- **CI.** Runs lint, format checks, the dependency audit, contract checks, and tests, and verifies the real model weights.
- **CD.** Deploys to staging only after CI succeeds on `main`.
- **Dependabot.** Updates pip packages, GitHub Actions, and base images weekly.

## Local tools

### Command line

Run `python -m wastenot_ai.cli <command>` with `PYTHONPATH=src`.

| Command | Purpose |
|---|---|
| `doctor` | Check the environment, model, and configuration |
| `detect` | Run detection on an image |
| `recommend` | Get a recommendation from a context JSON, without detection |
| `pipeline` | Run detection and a recommendation for one image |
| `catalog` | Print the object catalog |
| `benchmark` | Measure cold and warm inference latency |
| `list-gemini-models` | List the models available to the configured key |
| `serve-worker` / `serve-recommendations` | Run a private worker |
| `serve-mock` | Run the public API mock |
| `demo` | Run the browser camera demo |
| `export-contracts` | Generate or check the contract artifacts |

`--provider fixture` simulates the LLM without a key and always marks its output `prototype_only=true`. `--demo-guides` enables draft guides for experiments; production uses `--guides` with published guides only. Context examples live in [`examples/`](examples).

### Browser demo

```sh
./demo-camera.sh
```

The demo opens <http://127.0.0.1:8092>.

1. Capture a photo.
2. Pick a detected object.
3. Confirm its material and condition.
4. Request a recommendation.

Photos stay in memory and are cleared after 15 minutes. The demo server is loopback-only, protected by CSRF tokens and Host/Origin checks, and limited to 30 LLM calls per hour. It never saves logs. Do not expose it as a public API.

### Mock API

```sh
PYTHONPATH=src python -m wastenot_ai.cli serve-mock --port 8094
```

Open <http://127.0.0.1:8094/docs> and authorize as `mock-user-a` or `mock-user-b`. See the [HTTP contract](docs/AI_HTTP_CONTRACT.md#mock-server) for the available scenarios.

## Project structure

```
waste-not-ai/
├── src/wastenot_ai/       Workers, decision rules, providers, CLI, demo, and mock
│   └── data/              Catalog, guides, sources, and templates
├── contracts/v1/          Generated OpenAPI contracts, errors, examples, and the backend pin
├── guides/                Published pilot guides
├── guide-proposals/       Draft guide proposals under review
├── examples/              Context fixtures for the CLI
├── tests/                 Pytest suite
├── scripts/               Dataset preparation, training, evaluation, and release tooling
├── benchmarks/            Camera replay benchmark definitions
├── evaluations/           Generated benchmark reports
├── annotations/           Research annotation provenance
├── deploy/                Deploy helper and hardened systemd units
├── docs/                  Contract, integration, and deployment guides; archive/ holds past reports
├── serve.py               Production entrypoint (detection, recommendation, or demo)
└── model.json             Pinned model manifest
```

## Deployment

The workers deploy to staging through a self-hosted runner as systemd services, with atomic release switching and automatic rollback. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Documentation

| Document | Description |
|---|---|
| [AI HTTP contract](docs/AI_HTTP_CONTRACT.md) | Contract artifacts, ownership, private endpoints, and adapter obligations |
| [Integration](docs/INTEGRATION.md) | End-to-end flow between Android, Go, and AI, plus the pilot scope |
| [Deployment](docs/DEPLOYMENT.md) | Host setup, CI, release, recovery, and resource limits |
| [Pilot guides](guides/README.md) | Published guide bundle and its review status |
| [Archived reports](docs/archive/README.md) | Historical dataset, training, and evaluation reports |

## Data and licensing

- **Model weights.** Ultralytics YOLO weights are licensed under AGPL-3.0 or an Ultralytics Enterprise license. Dataset attribution ships with the model release.
- **Training data.** Dataset notices for derived training data are in [docs/DATASET_NOTICES.md](docs/DATASET_NOTICES.md). Research annotation provenance is in [annotations/README.md](annotations/README.md).
- **What stays out of Git.** Raw photos, datasets, archives, and trained weights are never committed.

## Related repositories

| Repository | Description |
|---|---|
| [waste-not-backend](https://github.com/waste-not-app/waste-not-backend) | REST API and AI gateway |
| [waste-not-android](https://github.com/waste-not-app/waste-not-android) | Android client |
| [waste-not-landing](https://github.com/waste-not-app/waste-not-landing) | Marketing site and privacy policy |
