.PHONY: help build up down logs test lint clean

help:  ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker images
	docker-compose build

up: ## Start services in detached mode
	docker-compose up -d

down: ## Stop services
	docker-compose down

logs: ## View logs
	docker-compose logs -f

test: ## Run unit and integration tests
	docker-compose exec api pytest tests/ -v

test-unit: ## Run only unit tests
	docker-compose exec api pytest tests/unit -v

test-integration: ## Run only integration tests
	docker-compose exec api pytest tests/integration -v

lint: ## Check code quality with ruff
	docker-compose exec api pip install ruff
	docker-compose exec api ruff check .

format: ## Format code with ruff
	docker-compose exec api pip install ruff
	docker-compose exec api ruff format .

clean: ## Remove artifacts and cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf mlruns/ .coverage htmlcov/

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/secrets.yaml
	kubectl apply -f k8s/postgres-statefulset.yaml
	kubectl apply -f k8s/redis-deployment.yaml
	kubectl apply -f k8s/mlflow-deployment.yaml
	kubectl apply -f k8s/api-deployment.yaml
	kubectl apply -f k8s/worker-deployment.yaml

k8s-delete: ## Delete from Kubernetes
	kubectl delete -f k8s/worker-deployment.yaml
	kubectl delete -f k8s/api-deployment.yaml
	kubectl delete -f k8s/mlflow-deployment.yaml
	kubectl delete -f k8s/redis-deployment.yaml
	kubectl delete -f k8s/postgres-statefulset.yaml
	kubectl delete -f k8s/secrets.yaml
