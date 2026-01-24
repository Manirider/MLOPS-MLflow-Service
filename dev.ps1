
param (
    [string]$Command
)

Switch ($Command) {
    "build" {
        docker-compose build
    }
    "up" {
        docker-compose up -d
    }
    "down" {
        docker-compose down
    }
    "logs" {
        docker-compose logs -f
    }
    "test" {
        docker-compose exec api pytest tests/ -v
    }
    "test-unit" {
        docker-compose exec api pytest tests/unit -v
    }
    "test-integration" {
        docker-compose exec api pytest tests/integration -v
    }
    "lint" {
        docker-compose exec api pip install ruff
        docker-compose exec api ruff check .
    }
    "format" {
        docker-compose exec api pip install ruff
        docker-compose exec api ruff format .
    }
    "clean" {
        Remove-Item -Path "mlruns", ".coverage", "htmlcov" -Recurse -ErrorAction SilentlyContinue
        Get-ChildItem -Path . -Include "__pycache__", ".pytest_cache" -Recurse | Remove-Item -Recurse -Force
    }
    Default {
        Write-Host "Usage: .\dev.ps1 [command]"
        Write-Host "Commands:"
        Write-Host "  build             Build Docker images"
        Write-Host "  up                Start services in detached mode"
        Write-Host "  down              Stop services"
        Write-Host "  logs              View logs"
        Write-Host "  test              Run all tests"
        Write-Host "  test-unit         Run unit tests"
        Write-Host "  test-integration  Run integration tests"
        Write-Host "  lint              Check code quality"
        Write-Host "  format            Format code"
        Write-Host "  clean             Remove artifacts"
    }
}
