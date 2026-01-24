# How This Platform is Built

> The architectural decisions behind a production-ready MLOps system.

## The Big Picture

This platform follows a simple philosophy: **separate what changes from what doesn't**.

- Models change frequently → isolated training pipeline
- Experiments need comparison → centralized tracking
- Predictions need speed → cached model serving
- Operations need visibility → API over everything

Here's how the pieces fit together:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           What You Interact With                            │
│                                                                             │
│      ┌─────────────────┐              ┌──────────────────────────────┐     │
│      │   MLflow UI     │              │       API (FastAPI)          │     │
│      │   "See what     │              │       "Do stuff"             │     │
│      │    happened"    │              │                              │     │
│      │   :5000         │              │   POST /train                │     │
│      └────────┬────────┘              │   GET  /experiments          │     │
│               │                       │   GET  /models               │     │
│               │                       │   POST /predict              │     │
│               │                       │   :8000                      │     │
│               │                       └──────────────┬───────────────┘     │
└───────────────┼──────────────────────────────────────┼─────────────────────┘
                │                                      │
                ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          What Does The Work                                 │
│                                                                             │
│   ┌───────────────────────┐    ┌─────────────────────────────────────────┐ │
│   │   MLflow Server       │    │         Celery Workers                  │ │
│   │   (The Librarian)     │    │         (The Trainers)                  │ │
│   │                       │    │                                         │ │
│   │   Tracks experiments  │◄───┤   Run training jobs asynchronously     │ │
│   │   Manages registry    │    │   Report results back                  │ │
│   │   Stores artifacts    │    │   Scale horizontally                   │ │
│   └───────────┬───────────┘    └─────────────────────────────────────────┘ │
│               │                                                             │
└───────────────┼─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         What Persists Everything                            │
│                                                                             │
│   ┌───────────────────────────┐     ┌─────────────────────────────────┐   │
│   │       PostgreSQL          │     │     File System / Volumes       │   │
│   │                           │     │                                 │   │
│   │   Experiments            │     │   Model files (.pkl, .pt)        │   │
│   │   Run metadata           │     │   Training plots (.png)         │   │
│   │   Metrics & parameters   │     │   Classification reports        │   │
│   │   Model registry         │     │   Confusion matrices            │   │
│   └───────────────────────────┘     └─────────────────────────────────┘   │
│                                                                             │
│   ┌───────────────────────────┐                                            │
│   │        Redis              │                                            │
│   │                           │                                            │
│   │   Message queue (Celery)  │                                            │
│   │   Prediction cache        │                                            │
│   │   Rate limiting state     │                                            │
│   └───────────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────┘

## Why We Made These Choices

### MLflow as the Core

**The problem**: ML teams waste months building homegrown tracking systems that inevitably become unmaintainable.

**Our solution**: MLflow handles experiment tracking, model registry, and artifact storage. It's battle-tested by companies like Facebook, Microsoft, and Databricks.

**What this means for you**:
- Every experiment is automatically logged
- Model versions are tracked like code commits
- Any team member can see what's been tried

### PostgreSQL, Not SQLite

**The problem**: SQLite breaks under concurrent access. Fine for prototypes, disaster for production.

**Our solution**: PostgreSQL as MLflow's backend.

**What this means for you**:
- Multiple training jobs can run simultaneously
- Queries stay fast even with 10,000+ runs
- Data is properly backed up and recoverable

### Async Training with Celery

**The problem**: Training blocks the API. A 10-minute training job means 10 minutes of frozen endpoints.

**Our solution**: Celery + Redis for distributed task queues.

**What this means for you**:
- Training starts immediately, returns a job ID
- Multiple training jobs can run in parallel
- Add more workers to scale horizontally

### Redis for Caching

**The problem**: Loading a model from disk for every prediction is slow. Users hate slow.

**Our solution**: Redis caches predictions and loaded models.

**What this means for you**:
- Repeated predictions are instant
- Model loading happens once, not per-request
- Rate limiting protects against abuse

## How Requests Flow

### When You Train a Model
Your Request                        What Actually Happens
     │
     │  POST /train
     │  {epochs: 10, lr: 0.001}
     │
     ▼
┌─────────────────┐
│  FastAPI        │ ─── Validates your request
│  receives it    │ ─── Creates a job ID
└────────┬────────┘
         │
         │  Pushes to Redis queue
         ▼
┌─────────────────┐
│  Celery Worker  │ ─── Picks up the job
│  starts work    │ ─── Loads MNIST data
│                 │ ─── Trains the model
│                 │ ─── Logs to MLflow
└────────┬────────┘
         │
         │  Updates job status
         ▼
┌─────────────────┐
│  MLflow saves   │ ─── Parameters logged
│  everything     │ ─── Metrics recorded
│                 │ ─── Model artifact stored
└─────────────────┘

Meanwhile: You got your job_id back in 50ms.
Check /train/status/{job_id} to see progress.

### When You Make a Prediction


Your Request                        What Actually Happens
     │
     │  POST /predict
     │  {image: [784 pixels]}
     │
     ▼
┌─────────────────┐
│  Check Redis    │ ─── Seen this exact input before?
│  cache first    │     → Return cached result (5ms)
└────────┬────────┘
         │
         │  Cache miss
         ▼
┌─────────────────┐
│  Load model     │ ─── Is Production model in memory?
│  from registry  │     → Use it
│                 │     → Else load from MLflow
└────────┬────────┘
         │
         │  Model ready
         ▼
┌─────────────────┐
│  Run inference  │ ─── Preprocess image
│                 │ ─── Forward pass
│                 │ ─── Format probabilities
└────────┬────────┘
         │
         │  Cache result for future
         ▼
   Returns: {prediction: 7, confidence: 0.98}

## Scaling This Platform

### What Works Today (Local/Small Team)

- Docker Compose orchestration
- Single API instance
- File-based artifact storage
- Redis for caching and task queue

### What You'd Change at Scale

| Component | Today | At Scale |
|-----------|-------|----------|
| **Orchestration** | Docker Compose | Kubernetes |
| **API** | Single instance | Load-balanced replicas |
| **Artifacts** | Local filesystem | S3 / GCS / Azure Blob |
| **Training** | 1 Celery worker | Auto-scaling worker pool |
| **Database** | Single PostgreSQL | Managed RDS with replicas |
| **Caching** | Single Redis | Redis Cluster |

**The good news**: Kubernetes manifests are already in `k8s/`. The path to production is clear.

## Security Model

### What's Protected

| Endpoint | Protection | Why |
|----------|------------|-----|
| `/train` | API Key required | Prevents unauthorized training |
| `/predict` | API Key required | Protects model inference |
| `/models/transition` | API Key required | Controls model promotion |
| `/experiments` | Public | Read-only experiment data |
| `/health` | Public | Health checks need no auth |

### How Authentication Works

```bash
# Without API key → 403 Forbidden
curl -X POST http://localhost:8000/train

# With API key → Works
curl -X POST http://localhost:8000/train \
  -H "X-API-Key: your-secret-key"
```

### Production Security Checklist

- [ ] Replace default API key in `.env`
- [ ] Enable HTTPS (add nginx/traefik with TLS)
- [ ] Set up proper secrets management
- [ ] Enable audit logging for model transitions
- [ ] Restrict network access with firewall rules

## Monitoring & Observability

### Built-In Health Checks

```bash
# Is the service alive?
GET /health
→ {"status": "healthy", "timestamp": "..."}

# Is it ready to serve traffic?
GET /ready  
→ {"status": "ready", "mlflow": "connected", "redis": "connected"}

### Metrics Available

The platform exports Prometheus metrics:

- `http_requests_total` — API request counts
- `http_request_duration_seconds` — Latency histograms
- `training_jobs_active` — Currently running jobs
- `model_predictions_total` — Prediction counts by model version

### The Monitoring Stack

- **Prometheus** (`:9090`) — Scrapes metrics every 15s
- **Grafana** (`:3000`) — Dashboards for visualization
- **Alerts** — Configured for drift detection and high latency

## Directory Map

mlops-mlflow-service/
│
├── api/                      # The FastAPI application
│   ├── app/
│   │   ├── main.py          # Application entry point
│   │   ├── routes/          # Endpoint definitions
│   │   │   ├── train.py     #   POST /train
│   │   │   ├── predict.py   #   POST /predict  
│   │   │   ├── experiments.py#  GET /experiments
│   │   │   └── models.py    #   GET /models
│   │   ├── services/        # Business logic
│   │   │   ├── training_service.py
│   │   │   ├── inference_service.py
│   │   │   └── mlflow_service.py
│   │   └── middleware/      # Auth, logging, security
│   └── requirements.txt
│
├── ml_core/                  # Machine learning code
│   ├── training/            # Model training
│   ├── experiments/         # Hyperparameter search
│   └── models/              # Model definitions
│
├── tests/                   # 52 tests and counting
│   ├── unit/               # Fast, isolated tests
│   └── integration/        # End-to-end tests
│
├── k8s/                     # Kubernetes manifests
│   ├── api-deployment.yaml
│   ├── mlflow-deployment.yaml
│   └── worker-deployment.yaml
│
├── monitoring/              # Observability configs
│   ├── prometheus/
│   └── grafana/
│
└── docker-compose.yml       # Local development setup

## Common Questions

**Q: Why not use TensorFlow/PyTorch serving directly?**

A: MLflow provides model versioning and registry that serving solutions lack. We load models from MLflow, so you get the best of both: tracked experiments AND fast inference.

**Q: Can I use a different model architecture?**

A: Yes. Replace `ml_core/models/` with your model code. As long as it's scikit-learn compatible or logged with `mlflow.pyfunc`, inference will work.

**Q: How do I add a new endpoint?**

A: Add a route file in `api/app/routes/`, define schemas in `schemas/`, and register it in `main.py`. The pattern is consistent throughout.

**Q: What if MLflow goes down?**

A: The API has a readiness probe that fails when MLflow is unreachable. Kubernetes will stop sending traffic until it recovers. Training is queued in Redis and will resume.

*This architecture is designed to grow with you—from local experiments to production at scale.*
