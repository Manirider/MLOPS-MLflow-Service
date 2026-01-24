# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-01-24

### Initial Production Release

The platform is now production-ready with all core features implemented and verified.

### Added

**Infrastructure**
- Docker Compose orchestration with 7 services
- PostgreSQL backend for MLflow (production-grade)
- Redis for caching and message queuing
- Health and readiness endpoints
- Prometheus metrics and Grafana dashboards

**API Endpoints**
- `POST /train` — Async training with Celery
- `GET /train/status/{id}` — Job status tracking
- `GET /experiments` — List all experiments
- `GET /models` — List registered models
- `POST /models/transition` — Promote model stages
- `POST /predict` — Single prediction with caching
- `POST /predict/batch` — Batch predictions
- `POST /predict/staging` — Staging model predictions

**ML Lifecycle**
- MLflow experiment tracking
- Model registry with versioning
- Stage transitions (None → Staging → Production → Archived)
- Hyperparameter search orchestration
- Best model auto-registration

**Distributed Training**
- Celery worker for async training
- Redis as message broker
- Non-blocking API responses
- Job queue with status tracking

**Model Drift Detection**
- Prediction logging to PostgreSQL
- Population Stability Index (PSI) calculation
- Drift status API endpoints
- Prometheus alerts for drift thresholds

**Security**
- API key authentication
- Rate limiting with Redis
- Security headers middleware
- No hardcoded credentials

**Testing**
- 52 tests (unit + integration)
- Full API coverage
- Mock-based isolation

**Kubernetes**
- Deployment manifests in `k8s/`
- StatefulSet for PostgreSQL
- Service definitions
- Secrets management

### Technical Decisions

- **MLflow over custom tracking**: Industry-standard, extensible, UI included
- **PostgreSQL over SQLite**: Concurrent access, production reliability
- **Celery over threads**: Horizontal scaling, fault tolerance
- **Redis for caching**: Sub-millisecond latency, distributed state

## What's Next

Potential future additions:
- A/B testing framework
- Feature store integration
- Model explainability (SHAP/LIME)
- Blue/green deployments
- Custom web dashboard
