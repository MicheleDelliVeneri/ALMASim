.PHONY: help start start-detached stop restart restart-console restart-frontend start-prod start-prod-detached stop-prod logs logs-console logs-frontend ps status build build-prod clean shell-console shell-frontend health test lint format typecheck check

# Default target
.DEFAULT_GOAL := help

# Compose v2 ("docker compose"); override with COMPOSE=docker-compose for v1.
COMPOSE ?= docker compose
# Development loads docker-compose.override.yml automatically; production
# layers docker-compose.prod.yml on the base file instead.
COMPOSE_PROD = $(COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml

help: ## Show this help message
	@echo "ALMASim Commands"
	@echo "================"
	@echo ""
	@echo "Development Checks (run locally with uv):"
	@echo "  make test               - Run the Python test suite"
	@echo "  make lint               - Check formatting and lint with ruff"
	@echo "  make format             - Format and apply ruff's automatic fixes"
	@echo "  make typecheck          - Type-check with mypy"
	@echo "  make check              - lint, typecheck and test"
	@echo ""
	@echo "Development Commands:"
	@echo "  make start              - Start all services (web app on :3000, console on :8765)"
	@echo "  make start-detached     - Start all services in detached mode"
	@echo "  make stop               - Stop all services"
	@echo "  make restart            - Restart all services"
	@echo "  make restart-console    - Restart only the console"
	@echo "  make restart-frontend   - Restart only frontend service"
	@echo ""
	@echo "Production Commands:"
	@echo "  make start-prod         - Start all services in production mode"
	@echo "  make start-prod-detached - Start production services in detached mode"
	@echo "  make stop-prod          - Stop production services"
	@echo ""
	@echo "Logs & Monitoring:"
	@echo "  make logs               - Follow logs for all services"
	@echo "  make logs-console       - Follow logs for the console only"
	@echo "  make logs-frontend      - Follow logs for frontend only"
	@echo "  make ps                 - Show running containers"
	@echo "  make status             - Show service status"
	@echo "  make health             - Query the console and web app health endpoints"
	@echo ""
	@echo "Build & Maintenance:"
	@echo "  make build              - Build all Docker images"
	@echo "  make build-prod         - Build production Docker images"
	@echo "  make clean              - Stop and remove all containers, networks, and volumes"
	@echo "  make shell-console      - Open shell in the console container"
	@echo "  make shell-frontend     - Open shell in frontend container"
	@echo ""

# Development checks (the same commands CI runs)
test: ## Run the Python test suite
	@uv run pytest tests

lint: ## Check formatting and lint with ruff
	@uv run ruff format --check .
	@uv run ruff check .

format: ## Format and apply ruff's automatic fixes
	@uv run ruff format .
	@uv run ruff check --fix .

typecheck: ## Type-check with mypy
	@uv run mypy

check: lint typecheck test ## lint, typecheck and test

# Local development targets
start: ## Start all services with Docker (development mode with hot reload)
	@echo "Starting ALMASim services with Docker..."
	@echo "Console (API v2) will run on port 8765"
	@echo "Frontend will run on port 3000 (with hot reload)"
	@echo "Press Ctrl+C to stop both services"
	@$(COMPOSE) up

start-detached: ## Start all services in detached mode
	@echo "Starting ALMASim services with Docker in detached mode..."
	@echo "Console (API v2) will run on port 8765"
	@echo "Frontend will run on port 3000 (with hot reload)"
	@$(COMPOSE) up -d
	@echo "Services started. Use 'make stop' to stop them."
	@echo "Use 'make logs' to view logs."

stop: ## Stop all services
	@echo "Stopping ALMASim services..."
	@$(COMPOSE) down
	@echo "Services stopped"

restart: ## Restart all services
	@echo "Restarting ALMASim services..."
	@$(COMPOSE) restart
	@echo "Services restarted"

restart-console: ## Restart the console only
	@echo "Restarting the console..."
	@$(COMPOSE) restart console
	@echo "Console restarted"

restart-frontend: ## Restart frontend service only
	@echo "Restarting frontend service..."
	@$(COMPOSE) restart frontend
	@echo "Frontend service restarted"

# Production targets
start-prod: ## Start all services in production mode
	@echo "Starting ALMASim services in PRODUCTION mode..."
	@echo "Console (API v2) will run on port 8765"
	@echo "Frontend (production build) will run on port 3000"
	@echo "Press Ctrl+C to stop both services"
	@$(COMPOSE_PROD) up

start-prod-detached: ## Start production services in detached mode
	@echo "Starting ALMASim services in PRODUCTION mode (detached)..."
	@echo "Console (API v2) will run on port 8765"
	@echo "Frontend (production build) will run on port 3000"
	@$(COMPOSE_PROD) up -d
	@echo "Production services started. Use 'make stop-prod' to stop them."

stop-prod: ## Stop production services
	@echo "Stopping production services..."
	@$(COMPOSE_PROD) down
	@echo "Production services stopped"

# Logs and monitoring
logs: ## Follow logs for all services
	@$(COMPOSE) logs -f

logs-console: ## Follow logs for the console only
	@$(COMPOSE) logs -f console

logs-frontend: ## Follow logs for frontend service only
	@$(COMPOSE) logs -f frontend

ps: ## Show running containers
	@$(COMPOSE) ps

status: ## Show detailed service status
	@echo "=== Service Status ==="
	@$(COMPOSE) ps
	@echo ""
	@echo "=== Health Status ==="
	@docker inspect --format='{{.Name}}: {{.State.Health.Status}}' $$($(COMPOSE) ps -q) 2>/dev/null || echo "No health checks available"

# Build targets
build: ## Build all Docker images (development)
	@echo "Building Docker images for development..."
	@$(COMPOSE) build --no-cache

build-prod: ## Build all Docker images (production)
	@echo "Building Docker images for production..."
	@$(COMPOSE_PROD) build --no-cache

# Maintenance
clean: ## Stop and remove all containers, networks, and volumes
	@echo "Cleaning up Docker resources..."
	@$(COMPOSE) down -v --remove-orphans
	@$(COMPOSE_PROD) down -v --remove-orphans 2>/dev/null || true
	@echo "Cleanup complete"

shell-console: ## Open shell in the console container
	@$(COMPOSE) exec console /bin/sh

shell-frontend: ## Open shell in frontend container
	@$(COMPOSE) exec frontend /bin/sh

# Health checks
health: ## Check health of all services
	@echo "=== Console Health ==="
	@curl -s -H "Authorization: Bearer $${ALMASIM_TOKEN:-almasim-laptop-console}" http://localhost:8765/api/v2/health || echo "Console not responding"
	@echo ""
	@echo "=== Frontend Health ==="
	@curl -s -o /dev/null -w "Status: %{http_code}\n" http://localhost:3000 || echo "Frontend not responding"
