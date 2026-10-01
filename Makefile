.PHONY: help install pipeline test lint format serve mlflow-ui simulate-drift traffic traffic-drift drift docker-build up down clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:         ## Install everything (editable) for development
	pip install -e ".[dev]"

pipeline:        ## Run the DVC pipeline: generate -> prepare -> train -> evaluate
	dvc repro
	dvc metrics show

test:            ## Run unit, model and API tests
	pytest

lint:            ## Lint and check formatting
	ruff check .
	ruff format --check .

format:          ## Auto-fix lint issues and format
	ruff check --fix .
	ruff format .

serve:           ## Run the API locally with auto-reload
	uvicorn churn.serve.app:app --reload --port 8000

mlflow-ui:       ## Browse experiments and the model registry at :5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

simulate-drift:  ## Create a drifted "production" dataset
	python -m churn.data.generate --drift --n-rows 1000 --seed 99 --out data/simulated/drifted.csv

traffic:         ## Send normal customers to the running API
	python scripts/send_traffic.py

traffic-drift: simulate-drift  ## Send drifted customers to the running API
	python scripts/send_traffic.py --csv data/simulated/drifted.csv

drift:           ## Check the API's prediction log for data drift
	python -m churn.monitoring.drift

docker-build:    ## Build the serving image
	docker build -t mlops-churn-api .

up:              ## Start API + MLflow server with docker compose
	docker compose up -d --build

down:            ## Stop the compose stack
	docker compose down

clean:           ## Remove logs and generated reports
	rm -rf logs reports/drift.* data/simulated
