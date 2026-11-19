SHELL := /bin/sh
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV   ?= .venv
ifeq ($(OS),Windows_NT)
VENV_BIN := $(VENV)/Scripts
else
VENV_BIN := $(VENV)/bin
endif
PY      := $(VENV_BIN)/python
RUFF    := $(VENV_BIN)/ruff
MODELS  ?= artifacts/models
COMPOSE := docker compose
export PYTHONPATH := src

.PHONY: help
help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

.PHONY: install
install: $(PY) ## Install locked runtime and CI dependencies into .venv
	$(PY) -m pip install --require-hashes -r requirements-ci.lock

.PHONY: install-cpu
install-cpu: $(PY) ## Install locked CPU inference dependencies (torch, ultralytics)
	$(PY) -m pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.lock

.PHONY: fmt
fmt: ## Format code with ruff
	$(RUFF) format .
	$(RUFF) check --select I --fix .

.PHONY: fmt-check
fmt-check: ## Fail when code is not formatted
	$(RUFF) format --check .

.PHONY: lint
lint: ## Lint with ruff (pyflakes, bugbear, bandit security rules)
	$(RUFF) check .

.PHONY: audit
audit: ## Audit locked dependencies with pip-audit
	$(PY) audit_dependencies.py

.PHONY: sec
sec: lint audit ## Run all security checks

.PHONY: contracts
contracts: ## Verify generated contracts and the backend pin
	$(PY) -m wastenot_ai.cli export-contracts --check
	$(PY) scripts/check_contracts.py contracts/v1 contracts/backend-pin.json

.PHONY: export-contracts
export-contracts: ## Regenerate contracts after an intentional change
	$(PY) -m wastenot_ai.cli export-contracts

.PHONY: test
test: ## Run the test suite
	$(PY) -m pytest tests -q -p no:cacheprovider

.PHONY: check
check: fmt-check lint contracts test ## Run the local quality gate

.PHONY: ci
ci: check audit ## Run the CI quality gate

.PHONY: model
model: ## Download the released YOLO weights into artifacts/models (needs GITHUB_TOKEN)
	$(PY) scripts/fetch_model.py $(MODELS)

.PHONY: run-recommendation
run-recommendation: ## Run the recommendation worker on 127.0.0.1:8093
	@set -a; [ -f .env ] && . ./.env; set +a; $(PY) serve.py recommendation

.PHONY: run-detection
run-detection: ## Run the detection worker on 127.0.0.1:8091
	@set -a; [ -f .env ] && . ./.env; set +a; AI_MODEL_DIRECTORY=$(MODELS) $(PY) serve.py detection

.PHONY: docker-build
docker-build: ## Build both worker images
	docker build --target recommendation -t wastenot-ai-recommendation:local .
	docker build --target detection -t wastenot-ai-detection:local .

.PHONY: up
up: ## Start the recommendation worker (add PROFILES=detection for YOLO)
	$(COMPOSE) $(if $(PROFILES),--profile $(PROFILES)) up -d --build

.PHONY: down
down: ## Stop all workers
	$(COMPOSE) --profile detection down

.PHONY: logs
logs: ## Follow worker logs
	$(COMPOSE) --profile detection logs -f

.PHONY: clean
clean: ## Remove caches
	$(PY) -c "import shutil;[shutil.rmtree(p, ignore_errors=True) for p in ('.ruff_cache','.pytest_cache','build')]"
