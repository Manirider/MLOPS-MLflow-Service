# MLOps Platform

> A production-ready machine learning operations platform that makes experiment tracking, model versioning, and deployment actually work.

## What This Does

If you've ever trained a model, lost track of which hyperparameters worked best, or struggled to deploy a model to production, this platform solves those problems.

**In plain terms, this platform helps you:**

-  **Track every experiment** — Never lose track of what you tried and what worked
-  **Compare models side-by-side** — See which configuration actually performs better
-  **Version your models** — Know exactly which model is in production and why
-  **Deploy with confidence** — Serve predictions from registered models, not random files
-  **Reproduce results** — Every training run is reproducible with fixed seeds


## Quick Start

**Prerequisites:** Docker and Docker Compose (that's it!)

```bash

# Start everything
docker-compose up -d

# Verify it's working
curl http://localhost:8000/health
```

**That's it.** You now have a fully operational MLOps platform.

### What's Running?

| Service | URL | What It Does |
|---------|-----|--------------|
| **API** | [localhost:8000](http://localhost:8000/docs) | Your control plane — train models, make predictions |
| **MLflow** | [localhost:5000](http://localhost:5000) | Experiment dashboard — see all your runs |
| **Grafana** | [localhost:3000](http://localhost:3000) | Monitoring — watch your system health |
| **Prometheus** | [localhost:9090](http://localhost:9090) | Metrics — the raw numbers |


## How It Works

### The Architecture (Simplified)

You                           The Platform
 │                                 │
 │  POST /train {params}          │
 ├───────────────────────────────►│ ─── Celery queues the job
 │                                │
 │                                │ ─── Worker trains the model
 │                                │ ─── MLflow logs everything
 │                                │ ─── Model saved to registry
 │                                │
 │  GET /models                   │
 │◄───────────────────────────────┤ ─── See what's registered
 │                                │
 │  POST /predict {image}         │
 │◄───────────────────────────────┤ ─── Inference from Production model
 │                                │

### The Stack

- **PostgreSQL** — Stores all experiment metadata (not SQLite!)
- **Redis** — Message broker for distributed training + prediction caching
- **MLflow** — The experiment tracking and model registry backbone
- **FastAPI** — Clean, async REST API
- **Celery** — Distributed task queue for non-blocking training

## Using the API

### Train a Model

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mlops-secret-key-123" \
  -d '{
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 64,
    "hidden_size": 128
  }'
```

**Response:**
```json
{
  "message": "Training job accepted and queued",
  "job_id": "job_000042",
  "experiment_name": "MNIST_Experiments",
  "status": "running"
}
```

Training runs asynchronously. Your request returns immediately while the actual training happens in the background.

### Check Training Status

```bash
curl http://localhost:8000/train/status/job_000042 \
  -H "X-API-Key: mlops-secret-key-123"
```

### List Experiments

```bash
curl http://localhost:8000/experiments
```

**Response:**
```json
{
  "experiments": [
    {
      "experiment_id": "1",
      "name": "MNIST_Experiments",
      "total_runs": 15,
      "best_run": {
        "run_id": "a1b2c3d4",
        "metrics": {"accuracy": 0.9607}
      }
    }
  ],
  "total_count": 1
}
```

### See Registered Models

```bash
curl http://localhost:8000/models
```

**Response:**
```json
{
  "models": [
    {
      "name": "MNISTClassifier",
      "latest_version": "3",
      "latest_stage": "Production"
    }
  ],
  "total_count": 1
}
```

### Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mlops-secret-key-123" \
  -d '{"image": [0.0, 0.0, ..., 0.5, 0.9, ...]}'  # 784 pixel values
```

**Response:**
```json
{
  "prediction": 7,
  "confidence": 0.98,
  "probabilities": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.98, 0.0, 0.01],
  "model_name": "MNISTClassifier",
  "model_version": "3",
  "model_stage": "Production"
}
```

## Running Experiments

The platform includes an experiment orchestration script that:
1. Runs multiple training experiments with different hyperparameters
2. Compares results automatically
3. Registers the best model to the registry

```bash
docker-compose exec api python -m ml_core.experiments.run_experiments \
  --num-runs 10 \
  --register-best \
  --stage Production
```

This will:
- Run 10 experiments with random hyperparameter combinations
- Find the best one based on accuracy
- Register it as `MNISTClassifier` in the model registry
- Transition it to "Production" stage

Now your `/predict` endpoint will automatically use this model.

## Model Lifecycle

Models flow through stages:

Training → Registered → Staging → Production → Archived

### Promote a Model to Production

```bash
curl -X POST http://localhost:8000/models/transition \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "MNISTClassifier",
    "version": "3",
    "stage": "Production"
  }'
```

The platform ensures only one version is in Production at a time. Previous Production models are automatically archived.

## Testing

The platform includes a comprehensive test suite with 52 tests covering:
- Unit tests for all API endpoints
- Integration tests for MLflow tracking
- Integration tests for model registry operations
- End-to-end prediction flow tests

```bash
# Run all tests
docker-compose exec api pytest tests/ -v

# Run with coverage
docker-compose exec api pytest --cov=app --cov-report=term-missing

# Run just unit tests
docker-compose exec api pytest tests/unit -v

# Run just integration tests
docker-compose exec api pytest tests/integration -v
```

## Configuration

All configuration is managed through environment variables. Copy `.env.example` to `.env` and adjust as needed.

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_DB` | Database name | `mlflow_db` |
| `POSTGRES_USER` | Database user | `mlflow_user` |
| `POSTGRES_PASSWORD` | Database password | `mlflow_password` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://mlflow_server:5000` |
| `API_KEY` | API authentication key | `mlops-secret-key-123` |
| `EXPERIMENT_NAME` | Default experiment | `MNIST_Experiments` |
| `MODEL_NAME` | Default model name | `MNISTClassifier` |

**No credentials are hardcoded in the codebase.** Everything is environment-driven.


## Project Structure

mlops-mlflow-service/
├── api/                    # FastAPI application
│   ├── app/
│   │   ├── main.py        # Application entry point
│   │   ├── routes/        # API endpoints
│   │   ├── services/      # Business logic
│   │   ├── schemas/       # Pydantic models
│   │   └── middleware/    # Auth, logging, security
│   ├── Dockerfile         # Multi-stage production build
│   └── requirements.txt
│
├── ml_core/               # Machine learning code
│   ├── training/          # Training scripts
│   ├── experiments/       # Experiment orchestration
│   └── models/            # Model architectures
│
├── tests/                 # Test suite (52 tests)
│   ├── unit/             # Unit tests
│   └── integration/       # Integration tests
│
├── k8s/                   # Kubernetes manifests
├── monitoring/            # Prometheus & Grafana configs
├── docker-compose.yml     # Local orchestration
└── README.md

## Troubleshooting

### MLflow artifacts not saving?

On Windows with Docker Desktop, you might see permission errors. Fix:

```powershell
mkdir mlruns
icacls mlruns /grant "Everyone:(OI)(CI)F"
```

### Port already in use?

Update the ports in your `.env` file and rebuild:

```bash
docker-compose down
docker-compose up -d --build
```

### API returning 403?

Protected endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: mlops-secret-key-123" http://localhost:8000/train/jobs
```

### Tests failing?

Make sure all services are healthy first:

```bash
docker-compose ps
docker-compose logs api
```

## What's Next?

This platform is designed to be extended. Some ideas:

- **Add model drift detection** — Already included! Check `/drift/status`
- **Scale with Kubernetes** — Manifests included in `k8s/`
- **Add more model types** — The architecture supports any sklearn/PyTorch model
- **Integrate with CI/CD** — GitHub Actions workflows included


## Stopping the Platform

```bash
# Stop containers (keeps data)
docker-compose down

# Stop and remove everything including data
docker-compose down -v
```

## License

MIT License — Use it, modify it, ship it.

Built with ☕ and a healthy disregard for "it works on my machine."

## Author

MANIKANTA SURYASAI

AIML ENGINEER|DEVELOPER
