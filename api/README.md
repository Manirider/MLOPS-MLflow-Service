# FastAPI Service

> The RESTful control plane for all ML operations.

---

## What This Is

This is the API that sits between you and the ML platform. Instead of:
- Manually running training scripts
- Copying model files around
- Guessing which model version is in production

You just make HTTP requests.

---

## Quick Start

```bash
# Already running via docker-compose? Skip to the endpoints.

# Running standalone (for development):
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/train` | Start a training job |
| `GET` | `/train/status/{job_id}` | Check job status |
| `GET` | `/train/jobs` | List all training jobs |

### Experiments

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/experiments` | List all experiments |
| `GET` | `/experiments/{name}` | Get experiment details |
| `GET` | `/experiments/{name}/runs/{run_id}` | Get run details |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models` | List registered models |
| `GET` | `/models/{name}` | Get model details |
| `POST` | `/models/transition` | Promote model stage |

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single prediction (Production) |
| `POST` | `/predict/staging` | Single prediction (Staging) |
| `POST` | `/predict/batch` | Batch predictions |
| `DELETE` | `/predict/cache` | Clear prediction cache |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness check |
| `GET` | `/metrics` | Prometheus metrics |

---

## Authentication

Protected endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/train/jobs
```

The API key is set via the `API_KEY` environment variable.

---

## Directory Structure

```
api/
├── app/
│   ├── main.py            # FastAPI app, middleware, routers
│   ├── routes/            # Endpoint definitions
│   │   ├── train.py       # Training endpoints
│   │   ├── predict.py     # Prediction endpoints
│   │   ├── experiments.py # Experiment endpoints
│   │   ├── models.py      # Model registry endpoints
│   │   └── drift.py       # Drift detection endpoints
│   ├── services/          # Business logic
│   │   ├── training_service.py
│   │   ├── inference_service.py
│   │   ├── mlflow_service.py
│   │   └── drift_service.py
│   ├── schemas/           # Pydantic models
│   ├── middleware/        # Auth, security, logging
│   ├── models/            # Database models (SQLAlchemy)
│   ├── worker.py          # Celery app
│   └── tasks.py           # Celery tasks
├── Dockerfile             # Multi-stage production build
└── requirements.txt       # Dependencies
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=app --cov-report=html

# Format code
black app/
isort app/

# Type checking
mypy app/
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | Yes | — | MLflow server URL |
| `REDIS_HOST` | No | `redis` | Redis hostname |
| `REDIS_PORT` | No | `6379` | Redis port |
| `API_KEY` | No | `mlops-secret-key-123` | Authentication key |
| `DATABASE_URL` | No | — | PostgreSQL connection string |

---

*This service is designed to be stateless and horizontally scalable.*
