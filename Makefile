# ──────────────────────────────────────────────────────────────────────────
# Makefile — SHARP Doppler Activity Recognition & Person Identification
# ──────────────────────────────────────────────────────────────────────────
#
# This file is organized into 5 sections:
#   0. Configuration & Helpers
#   1. Core Training & Testing  (bare-metal & defined jobs)
#   2. Docker Image & Local Container Runs
#   3. Remote Execution  (4nt0n / Hyperstack Parallel)
#   4. Utility / Cleanup
#
# Run `make help` to see all available targets.
# ──────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION & HELPERS
# ════════════════════════════════════════════════════════════════

# ── Paths ──
PYTHON       ?= uv run python
DATASET_PATH ?= dataset/data/doppler_traces
WEIGHTS_DIR  ?= weights
ENV_NAME     ?= local
MACHINE      ?= hyperstack
export DEPLOY_STATE_FILE := remote_state_$(MACHINE).json

ifneq ($(FLAVOR),)
export HYPERSTACK_FLAVOR := $(FLAVOR)
endif

ifneq ($(REGION),)
export HYPERSTACK_ENVIRONMENT := $(REGION)
endif
TRAIN_DIR    ?= $(WEIGHTS_DIR)/$(ENV_NAME)/train
TEST_DIR     ?= $(WEIGHTS_DIR)/$(ENV_NAME)/test

# ── Default Hyperparameters ──
ARCH         ?= resnet8
TASK         ?= activity
EPOCHS       ?= 100
LR           ?= 0.001
DROPOUT      ?= 0.5
BATCH_SIZE   ?= 128
NUM_WORKERS  ?= 4

# ── Docker Settings ──
DOCKERHUB_USERNAME ?= r15i
PROJECT_NAME       ?= nndl-project
DOCKER_IMAGE       ?= docker.io/$(DOCKERHUB_USERNAME)/$(PROJECT_NAME):latest


# ── Directory creation ──
$(TRAIN_DIR) $(TEST_DIR):
	mkdir -p $(TRAIN_DIR) $(TEST_DIR)

# ════════════════════════════════════════════════════════════════
# HELP
# ════════════════════════════════════════════════════════════════
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@printf "\n\033[1m  SHARP Doppler — Makefile Targets\033[0m\n"
	@awk ' \
		BEGIN {FS = ":.*## "} \
		/^[a-zA-Z0-9_-]+:.*## / { \
			printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2 \
		} \
		/^# [0-9]+\. / { \
			section = substr($$0, 3); \
			printf "\n\033[1m%s\033[0m\n", section \
		} \
	' $(MAKEFILE_LIST)
	@printf "\n  Override defaults:  make train ARCH=transformer TASK=person_id EPOCHS=10\n\n"


# ════════════════════════════════════════════════════════════════
# 1.  CORE TRAINING & TESTING  (bare metal & defined jobs)
# ════════════════════════════════════════════════════════════════

.PHONY: train test test-latest all \
        job-activity job-person-id job-all \
        job-inception job-transformer job-resnet8

train: | $(TRAIN_DIR) ## Train model locally (bare metal)
	$(PYTHON) train.py \
		--dataset-path $(DATASET_PATH) \
		--arch $(ARCH) \
		--task $(TASK) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--dropout $(DROPOUT) \
		--batch-size $(BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--output-dir $(TRAIN_DIR) \
		$(if $(filter 1,$(FULLVRAM)),--fullvram,)

test: | $(TEST_DIR) ## Evaluate ALL checkpoints locally
	$(PYTHON) test.py \
		--dataset-path $(DATASET_PATH) \
		--weights-dir $(TRAIN_DIR) \
		--arch $(ARCH) \
		--task $(TASK) \
		--num-workers $(NUM_WORKERS) \
		--output-dir $(TEST_DIR) \
		--create-graphs \
		$(if $(filter 1,$(FULLVRAM)),--fullvram,)

test-latest: | $(TEST_DIR) ## Evaluate checkpoint for given ARCH and TASK
	$(PYTHON) test.py \
		--dataset-path $(DATASET_PATH) \
		--weights-dir $(TRAIN_DIR) \
		--arch $(ARCH) \
		--task $(TASK) \
		--num-workers $(NUM_WORKERS) \
		--output-dir $(TEST_DIR) \
		--create-graphs \
		--latest-only \
		$(if $(filter 1,$(FULLVRAM)),--fullvram,)

all: train test-latest ## Train, then evaluate the latest checkpoint

# ── Defined Job targets for testing each architecture / task ──
job-resnet8: ## Train and test ResNet8 for the active TASK
	$(MAKE) train ARCH=resnet8 TASK=$(TASK)
	$(MAKE) test-latest ARCH=resnet8 TASK=$(TASK)

job-activity: ## Train and test ResNet8 for Activity Recognition
	$(MAKE) job-resnet8 TASK=activity

job-person-id: ## Train and test ResNet8 for Person Identification
	$(MAKE) job-resnet8 TASK=person_id

job-all: ## Run activity and person_id jobs using ResNet8 locally (with low VRAM usage)
	$(MAKE) job-activity BATCH_SIZE=64 FULLVRAM=0
	$(MAKE) job-person-id BATCH_SIZE=64 FULLVRAM=0

compile-paper: ## Compile the project report PDF
	cd paper && pdflatex template.tex && bibtex template || true && pdflatex template.tex && pdflatex template.tex



# ════════════════════════════════════════════════════════════════
# 2.  DOCKER IMAGE & LOCAL CONTAINER RUNS
# ════════════════════════════════════════════════════════════════

.PHONY: docker-build docker-push docker-push-nndl-local \
        train-local test-local nndl-reup-local

docker-build: ## Build the Docker image
	docker build -t $(DOCKER_IMAGE) .

docker-push: docker-build ## Build + push Docker image to Docker Hub
	@echo "Pushing $(DOCKER_IMAGE) to Docker Hub..."
	docker push $(DOCKER_IMAGE)

docker-push-nndl-local: ## Commit running local container & push
	@echo "Committing local nndl_worker container and pushing to Docker Hub..."
	docker commit nndl_worker $(DOCKER_IMAGE)
	docker push $(DOCKER_IMAGE)

train-local: ## Train via Docker locally (auto-detects GPU)
	@echo "Training locally via Docker (using pre-built image)..."
	DOCKER_IMAGE=$(DOCKER_IMAGE) ./scripts/docker_run.sh make train ARCH=$(ARCH) TASK=$(TASK) EPOCHS=$(EPOCHS) BATCH_SIZE=$(BATCH_SIZE) LR=$(LR) DROPOUT=$(DROPOUT) NUM_WORKERS=4 ENV_NAME=local

test-local: ## Evaluate ALL checkpoints locally via Docker
	@echo "Evaluating ALL checkpoints locally via Docker..."
	DOCKER_IMAGE=$(DOCKER_IMAGE) ./scripts/docker_run.sh make test NUM_WORKERS=4 ENV_NAME=local

nndl-reup-local: clean-local train-local ## Restart local nndl worker container


# ════════════════════════════════════════════════════════════════
# 3.  REMOTE EXECUTION  (Hyperstack Parallel)
# ════════════════════════════════════════════════════════════════

# ── Hyperstack base & parallel targets ──
.PHONY: train-hyperstack test-hyperstack pull-hyperstack pull-results-hyperstack \
        all-hyperstack hyperstack-activity hyperstack-person-id

train-hyperstack: docker-push ## Train single configuration on Hyperstack cloud
	@echo "Training on $(MACHINE)..."
	uv run python remote_exec/deploy.py execute --machine $(MACHINE) --target train ARCH=$(ARCH) TASK=$(TASK) EPOCHS=$(EPOCHS) BATCH_SIZE=$(BATCH_SIZE) LR=$(LR) DROPOUT=$(DROPOUT) NUM_WORKERS=8 FULLVRAM=0 ENV_NAME=$(MACHINE)

test-hyperstack: docker-push ## Evaluate checkpoints on Hyperstack
	@echo "Evaluating checkpoints on $(MACHINE)..."
	uv run python remote_exec/deploy.py execute --machine $(MACHINE) --target test BATCH_SIZE=$(BATCH_SIZE) NUM_WORKERS=8 FULLVRAM=0 ENV_NAME=$(MACHINE)

pull-hyperstack: ## Pull trained weights from Hyperstack to local machine
	@echo "Pulling weights from $(MACHINE)..."
	uv run python remote_exec/deploy.py download --machine $(MACHINE) --dest $(WEIGHTS_DIR)

pull-results-hyperstack: ## Pull test results from Hyperstack
	@echo "Pulling test results from $(MACHINE)..."
	uv run python remote_exec/deploy.py download --machine $(MACHINE) --dest $(WEIGHTS_DIR)/$(MACHINE)/ --subpath $(MACHINE)/test

all-hyperstack: docker-push ## Run job-all sequentially on single Hyperstack VM
	@echo "Training and evaluating job-all on $(MACHINE)..."
	uv run python remote_exec/deploy.py execute --machine $(MACHINE) --target job-all EPOCHS=$(EPOCHS) BATCH_SIZE=$(BATCH_SIZE) LR=$(LR) DROPOUT=$(DROPOUT) NUM_WORKERS=8 FULLVRAM=0 ENV_NAME=$(MACHINE)
	uv run python remote_exec/deploy.py download --machine $(MACHINE) --dest $(WEIGHTS_DIR)/$(MACHINE)/ --subpath $(MACHINE)/test
	uv run python remote_exec/deploy.py download --machine $(MACHINE) --dest $(WEIGHTS_DIR)

hyperstack-activity: ## Run job-activity (ResNet8) on Hyperstack
	@echo "Training and evaluating job-activity on hyperstack-activity..."
	DEPLOY_STATE_FILE=remote_state_hyperstack-activity.json HYPERSTACK_ENVIRONMENT=test HYPERSTACK_FLAVOR=n3-L40x1-spot uv run python remote_exec/deploy.py execute --machine hyperstack-activity --target job-activity EPOCHS=$(EPOCHS) BATCH_SIZE=$(BATCH_SIZE) LR=$(LR) DROPOUT=$(DROPOUT) NUM_WORKERS=8 FULLVRAM=0 ENV_NAME=hyperstack-activity
	DEPLOY_STATE_FILE=remote_state_hyperstack-activity.json uv run python remote_exec/deploy.py download --machine hyperstack-activity --dest $(WEIGHTS_DIR)/hyperstack-activity/ --subpath hyperstack-activity/test
	DEPLOY_STATE_FILE=remote_state_hyperstack-activity.json uv run python remote_exec/deploy.py download --machine hyperstack-activity --dest $(WEIGHTS_DIR)

hyperstack-person-id: ## Run job-person-id (ResNet8) on Hyperstack
	@echo "Training and evaluating job-person-id on hyperstack-person..."
	DEPLOY_STATE_FILE=remote_state_hyperstack-person.json HYPERSTACK_ENVIRONMENT=test HYPERSTACK_FLAVOR=n3-L40x1-spot uv run python remote_exec/deploy.py execute --machine hyperstack-person --target job-person-id EPOCHS=$(EPOCHS) BATCH_SIZE=$(BATCH_SIZE) LR=$(LR) DROPOUT=0.1 NUM_WORKERS=8 FULLVRAM=1 ENV_NAME=hyperstack-person
	DEPLOY_STATE_FILE=remote_state_hyperstack-person.json uv run python remote_exec/deploy.py download --machine hyperstack-person --dest $(WEIGHTS_DIR)/hyperstack-person/ --subpath hyperstack-person/test
	DEPLOY_STATE_FILE=remote_state_hyperstack-person.json uv run python remote_exec/deploy.py download --machine hyperstack-person --dest $(WEIGHTS_DIR)

# ════════════════════════════════════════════════════════════════
# 4.  UTILITY / CLEANUP
# ════════════════════════════════════════════════════════════════

.PHONY: clean-local clean-hyperstack clean-all clean-results-local clean-results-hyperstack clean-results-all

clean-local: ## Stop local servers & workers
	@echo "Cleaning local workers..."
	docker rm -f -v nndl_worker 2>/dev/null || true

clean-weights: ## Purge old checkpoints locally or in container
	@echo "Cleaning old weights..."
	uv run python scripts/cleanup_weights.py $(WEIGHTS_DIR)/$(ENV_NAME)

clean-hyperstack: ## Stop Hyperstack workers
	@echo "Cleaning hyperstack workers on $(MACHINE)..."
	uv run python remote_exec/deploy.py clean-container --machine $(MACHINE) || true
	uv run python remote_exec/deploy.py clean-machine --machine $(MACHINE) || true

clean-all: clean-local clean-hyperstack ## Stop everything everywhere
	@echo "All clean!"

clean-results-local: ## Clean local test results (CSVs and graphs)
	@echo "Cleaning local results..."
	rm -rf $(WEIGHTS_DIR)/local/test/*

clean-results-hyperstack: ## Clean local copy of Hyperstack test results
	@echo "Cleaning hyperstack local results..."
	rm -rf $(WEIGHTS_DIR)/hyperstack/test/*

clean-results-all: clean-results-local clean-results-hyperstack ## Clean all local test results
	@echo "All local results cleaned!"
