# Contributing

Thanks for considering contributing! Here's how to get involved.

---

## Development Setup

```bash
# Clone the repo
git clone <repo-url>
cd mlops-mlflow-service

# Start the platform
docker-compose up -d

# Run tests to verify everything works
docker-compose exec api pytest tests/ -v
```

---

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Keep commits small and focused
- Write tests for new functionality
- Update documentation if needed

### 3. Run the Tests

```bash
# All tests must pass
docker-compose exec api pytest tests/ -v

# Check coverage
docker-compose exec api pytest --cov=app
```

### 4. Submit a Pull Request

- Describe what you changed and why
- Reference any related issues
- Wait for review

---

## Code Style

- **Python**: Follow PEP 8. Use Black for formatting.
- **Commits**: Keep messages clear and concise.
- **Documentation**: Update docs when adding features.

```bash
# Format code
black api/app/
isort api/app/
```

---

## Adding New Endpoints

1. Create route file in `api/app/routes/`
2. Define schemas in `api/app/schemas/`
3. Add business logic in `api/app/services/`
4. Register router in `api/app/main.py`
5. Add tests in `tests/unit/api/`

---

## Reporting Issues

When reporting bugs, include:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Environment (OS, Docker version)
- Relevant logs

---

## Questions?

Open an issue with the `question` label. We'll get back to you.
