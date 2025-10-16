# =============================================================================
# Makefile para EPP Detector
# Simplifica comandos comunes de desarrollo, testing y deployment
# =============================================================================

.PHONY: help install test lint format clean docker-build docker-up docker-down deploy

# Variables
PYTHON := python3
PIP := pip3
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := epp-detector
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Colores para output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# =============================================================================
# Help
# =============================================================================

help: ## Mostrar esta ayuda
	@echo "$(BLUE)EPP Detector - Makefile$(NC)"
	@echo ""
	@echo "$(GREEN)Comandos disponibles:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Setup & Installation
# =============================================================================

install: ## Instalar dependencias de desarrollo
	@echo "$(BLUE)Instalando dependencias...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)✓ Dependencias instaladas$(NC)"

install-prod: ## Instalar solo dependencias de producción
	@echo "$(BLUE)Instalando dependencias de producción...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencias instaladas$(NC)"

env: ## Crear archivo .env desde template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ .env creado desde .env.example$(NC)"; \
		echo "$(YELLOW)⚠ Edita .env con tus configuraciones$(NC)"; \
	else \
		echo "$(YELLOW)⚠ .env ya existe$(NC)"; \
	fi

setup: install env ## Setup completo del proyecto
	@echo "$(GREEN)✓ Setup completado$(NC)"

# =============================================================================
# Testing
# =============================================================================

test: ## Ejecutar todos los tests
	@echo "$(BLUE)Ejecutando tests...$(NC)"
	pytest tests/ -v

test-unit: ## Ejecutar solo tests unitarios
	@echo "$(BLUE)Ejecutando tests unitarios...$(NC)"
	pytest tests/ -v -m unit

test-api: ## Ejecutar tests de API
	@echo "$(BLUE)Ejecutando tests de API...$(NC)"
	pytest tests/test_api.py -v

test-coverage: ## Ejecutar tests con coverage
	@echo "$(BLUE)Ejecutando tests con coverage...$(NC)"
	pytest tests/ --cov=api --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Reporte en htmlcov/index.html$(NC)"

test-fast: ## Ejecutar tests rápidos (sin slow)
	@echo "$(BLUE)Ejecutando tests rápidos...$(NC)"
	pytest tests/ -v -m "not slow"

# =============================================================================
# Linting & Formatting
# =============================================================================

lint: ## Ejecutar linters (flake8)
	@echo "$(BLUE)Ejecutando flake8...$(NC)"
	flake8 api/ tests/ --max-line-length=100 --exclude=__pycache__

format: ## Formatear código con black e isort
	@echo "$(BLUE)Formateando código...$(NC)"
	black api/ tests/ --line-length=100
	isort api/ tests/
	@echo "$(GREEN)✓ Código formateado$(NC)"

format-check: ## Verificar formato sin modificar
	@echo "$(BLUE)Verificando formato...$(NC)"
	black api/ tests/ --check --line-length=100
	isort api/ tests/ --check-only

typecheck: ## Verificar type hints con mypy
	@echo "$(BLUE)Verificando tipos...$(NC)"
	mypy api/ --ignore-missing-imports

check: lint format-check ## Ejecutar todas las verificaciones

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build imagen Docker
	@echo "$(BLUE)Building Docker image...$(NC)"
	./scripts/docker-build.sh dev

docker-build-prod: ## Build imagen Docker para producción
	@echo "$(BLUE)Building production Docker image...$(NC)"
	./scripts/docker-build.sh prod

docker-up: ## Levantar servicios con docker-compose
	@echo "$(BLUE)Levantando servicios...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Servicios iniciados$(NC)"
	@echo "$(YELLOW)API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Docs: http://localhost:8000/docs$(NC)"

docker-down: ## Detener servicios
	@echo "$(BLUE)Deteniendo servicios...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Servicios detenidos$(NC)"

docker-logs: ## Ver logs de containers
	$(DOCKER_COMPOSE) logs -f api

docker-shell: ## Abrir shell en container de API
	$(DOCKER_COMPOSE) exec api /bin/bash

docker-restart: docker-down docker-up ## Reiniciar servicios

docker-clean: ## Limpiar containers, imágenes y volúmenes
	@echo "$(BLUE)Limpiando Docker...$(NC)"
	./scripts/docker-build.sh clean
	@echo "$(GREEN)✓ Cleanup completado$(NC)"

# =============================================================================
# Development
# =============================================================================

run: ## Ejecutar API en modo desarrollo
	@echo "$(BLUE)Iniciando API...$(NC)"
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

run-prod: ## Ejecutar API en modo producción
	@echo "$(BLUE)Iniciando API (producción)...$(NC)"
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# =============================================================================
# Cleaning
# =============================================================================

clean: ## Limpiar archivos temporales
	@echo "$(BLUE)Limpiando archivos temporales...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Limpieza completada$(NC)"

clean-all: clean docker-clean ## Limpiar todo (Python + Docker)

# =============================================================================
# CI/CD
# =============================================================================

ci: check test-coverage ## Ejecutar pipeline de CI (checks + tests)
	@echo "$(GREEN)✓ CI pipeline completado$(NC)"

pre-commit: format lint test-fast ## Ejecutar antes de commit
	@echo "$(GREEN)✓ Pre-commit checks completados$(NC)"

# =============================================================================
# Deployment
# =============================================================================

deploy-gcp: ## Deploy a Google Cloud Run
	@echo "$(BLUE)Deploying to GCP Cloud Run...$(NC)"
	gcloud builds submit --tag gcr.io/$(GCP_PROJECT_ID)/$(PROJECT_NAME)
	gcloud run deploy $(PROJECT_NAME) \
		--image gcr.io/$(GCP_PROJECT_ID)/$(PROJECT_NAME) \
		--platform managed \
		--region us-central1 \
		--allow-unauthenticated \
		--memory 4Gi \
		--cpu 2
	@echo "$(GREEN)✓ Deployed to GCP$(NC)"

# =============================================================================
# Utilities
# =============================================================================

version: ## Mostrar versión actual
	@echo "$(BLUE)Version: $(VERSION)$(NC)"

info: ## Mostrar información del proyecto
	@echo "$(BLUE)Proyecto:$(NC) $(PROJECT_NAME)"
	@echo "$(BLUE)Version:$(NC) $(VERSION)"
	@echo "$(BLUE)Python:$(NC) $(shell $(PYTHON) --version)"
	@echo "$(BLUE)Docker:$(NC) $(shell docker --version 2>/dev/null || echo 'No instalado')"

tree: ## Mostrar estructura del proyecto
	@tree -I '__pycache__|*.pyc|venv|env|.git|htmlcov|.pytest_cache' -L 2

# =============================================================================
# Default
# =============================================================================

.DEFAULT_GOAL := help
