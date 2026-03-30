.PHONY: help start start-detached stop restart restart-backend restart-frontend logs logs-backend logs-frontend build build-prod clean ps status shell-backend shell-frontend

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "ALMASim Docker Compose Commands"
	@echo "================================"
	@echo ""
	@echo "Development Commands:"
	@echo "  make start              - Start all services (frontend on :3000, backend on :8000)"
	@echo "  make start-detached     - Start all services in detached mode"
	@echo "  make stop               - Stop all services"
	@echo "  make restart            - Restart all services"
	@echo "  make restart-backend    - Restart only backend service"
	@echo "  make restart-frontend   - Restart only frontend service"
	@echo ""
	@echo "Production Commands:"
	@echo "  make start-prod         - Start all services in production mode"
	@echo "  make start-prod-detached - Start production services in detached mode"
	@echo "  make stop-prod          - Stop production services"
	@echo ""
	@echo "Logs & Monitoring:"
	@echo "  make logs               - Follow logs for all services"
	@echo "  make logs-backend       - Follow logs for backend only"
	@echo "  make logs-frontend      - Follow logs for frontend only"
	@echo "  make ps                 - Show running containers"
	@echo "  make status             - Show service status"
	@echo ""
	@echo "Build & Maintenance:"
	@echo "  make build              - Build all Docker images"
	@echo "  make build-prod         - Build production Docker images"
	@echo "  make clean              - Stop and remove all containers, networks, and volumes"
	@echo "  make shell-backend      - Open shell in backend container"
	@echo "  make shell-frontend     - Open shell in frontend container"
	@echo ""

# Local development targets
start: ## Start all services with Docker (development mode with hot reload)
	@echo "Starting ALMASim services with Docker..."
	@echo "Backend API service will run on port 8000"
	@echo "Frontend will run on port 3000 (with hot reload)"
	@echo "Press Ctrl+C to stop both services"
	@docker-compose up

start-detached: ## Start all services in detached mode
	@echo "Starting ALMASim services with Docker in detached mode..."
	@echo "Backend API service will run on port 8000"
	@echo "Frontend will run on port 3000 (with hot reload)"
	@docker-compose up -d
	@echo "Services started. Use 'make stop' to stop them."
	@echo "Use 'make logs' to view logs."

stop: ## Stop all services
	@echo "Stopping ALMASim services..."
	@docker-compose down
	@echo "Services stopped"

restart: ## Restart all services
	@echo "Restarting ALMASim services..."
	@docker-compose restart
	@echo "Services restarted"

restart-backend: ## Restart backend API service only
	@echo "Restarting backend API service..."
	@docker-compose restart backend
	@echo "Backend service restarted"

restart-frontend: ## Restart frontend service only
	@echo "Restarting frontend service..."
	@docker-compose restart frontend
	@echo "Frontend service restarted"

# Production targets
start-prod: ## Start all services in production mode
	@echo "Starting ALMASim services in PRODUCTION mode..."
	@echo "Backend API service will run on port 8000"
	@echo "Frontend (production build) will run on port 3000"
	@echo "Press Ctrl+C to stop both services"
	@docker-compose -f docker-compose.prod.yml up

start-prod-detached: ## Start production services in detached mode
	@echo "Starting ALMASim services in PRODUCTION mode (detached)..."
	@echo "Backend API service will run on port 8000"
	@echo "Frontend (production build) will run on port 3000"
	@docker-compose -f docker-compose.prod.yml up -d
	@echo "Production services started. Use 'make stop-prod' to stop them."

stop-prod: ## Stop production services
	@echo "Stopping production services..."
	@docker-compose -f docker-compose.prod.yml down
	@echo "Production services stopped"

# Logs and monitoring
logs: ## Follow logs for all services
	@docker-compose logs -f

logs-backend: ## Follow logs for backend service only
	@docker-compose logs -f backend

logs-frontend: ## Follow logs for frontend service only
	@docker-compose logs -f frontend

ps: ## Show running containers
	@docker-compose ps

status: ## Show detailed service status
	@echo "=== Service Status ==="
	@docker-compose ps
	@echo ""
	@echo "=== Health Status ==="
	@docker inspect --format='{{.Name}}: {{.State.Health.Status}}' $$(docker-compose ps -q) 2>/dev/null || echo "No health checks available"

# Build targets
build: ## Build all Docker images (development)
	@echo "Building Docker images for development..."
	@docker-compose build --no-cache

build-prod: ## Build all Docker images (production)
	@echo "Building Docker images for production..."
	@docker-compose -f docker-compose.prod.yml build --no-cache

# Maintenance
clean: ## Stop and remove all containers, networks, and volumes
	@echo "Cleaning up Docker resources..."
	@docker-compose down -v --remove-orphans
	@docker-compose -f docker-compose.prod.yml down -v --remove-orphans 2>/dev/null || true
	@echo "Cleanup complete"

shell-backend: ## Open shell in backend container
	@docker-compose exec backend /bin/sh

shell-frontend: ## Open shell in frontend container
	@docker-compose exec frontend /bin/sh

# Health checks
health: ## Check health of all services
	@echo "=== Backend Health ==="
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Backend not responding"
	@echo ""
	@echo "=== Frontend Health ==="
	@curl -s -o /dev/null -w "Status: %{http_code}\n" http://localhost:3000 || echo "Frontend not responding"
