# Docker Setup for ALMASim

This document describes how to run ALMASim using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10 or higher
- Docker Compose 2.0 or higher
- At least 8GB of RAM available for Docker
- At least 10GB of free disk space

## Quick Start

### Development Mode (with hot reload)

```bash
# Start all services
make start

# Or start in detached mode
make start-detached

# View logs
make logs

# Stop services
make stop
```

**Services will be available at:**
- Frontend: http://localhost:3000 (with hot reload)
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Production Mode

```bash
# Start production services
make start-prod

# Or start in detached mode
make start-prod-detached

# Stop production services
make stop-prod
```

## Available Commands

### Starting Services

```bash
make start                  # Start dev services (with hot reload)
make start-detached         # Start dev services in background
make start-prod             # Start production services
make start-prod-detached    # Start production services in background
```

### Stopping Services

```bash
make stop                   # Stop development services
make stop-prod              # Stop production services
```

### Restarting Services

```bash
make restart                # Restart all services
make restart-backend        # Restart only backend
make restart-frontend       # Restart only frontend
```

### Viewing Logs

```bash
make logs                   # All services
make logs-backend           # Backend only
make logs-frontend          # Frontend only
```

### Building Images

```bash
make build                  # Build development images
make build-prod             # Build production images
```

### Maintenance

```bash
make ps                     # Show running containers
make status                 # Show detailed service status
make health                 # Check health endpoints
make clean                  # Remove all containers and volumes
```

### Debugging

```bash
make shell-backend          # Open shell in backend container
make shell-frontend         # Open shell in frontend container
```

## Architecture

### Development Mode (`docker-compose.yml`)

- **Frontend**: 
  - Uses `Dockerfile.dev` with hot reload
  - Mounts source code as volumes for instant updates
  - Node.js development server on port 3000
  
- **Backend**:
  - Python FastAPI application on port 8000
  - Mounts data directories (read-only) and output directory
  
- **Network**: Both services on `almasim-network` bridge

### Production Mode (`docker-compose.prod.yml`)

- **Frontend**:
  - Multi-stage build with optimized production bundle
  - Serves pre-built static assets
  - Smaller image size, faster startup
  
- **Backend**:
  - Same as development but with resource limits
  - Python optimizations enabled
  
- **Resource Limits**:
  - Backend: 4 CPU cores max, 8GB RAM max
  - Frontend: 1 CPU core max, 1GB RAM max

## Directory Structure

```
ALMASim/
├── docker-compose.yml          # Development configuration
├── docker-compose.prod.yml     # Production configuration
├── Makefile                    # Command shortcuts
├── backend/
│   ├── Dockerfile              # Backend image
│   ├── app/                    # FastAPI application
│   └── outputs/                # Simulation outputs (mounted)
├── frontend/
│   ├── Dockerfile              # Production frontend image
│   ├── Dockerfile.dev          # Development frontend image
│   ├── src/                    # Source code (mounted in dev)
│   └── static/                 # Static assets (mounted in dev)
└── data/                       # Data directory (mounted read-only)
```

## Volume Mounts

### Development Mode
- `./frontend/src` → `/app/src` (hot reload)
- `./frontend/static` → `/app/static` (static files)
- `./data` → `/app/data` (read-only, data files)
- `./backend/outputs` → `/app/outputs` (simulation outputs)

### Production Mode
- `./data` → `/app/data` (read-only, data files)
- `./backend/outputs` → `/app/outputs` (simulation outputs)

## Environment Variables

### Backend
- `MAIN_DIR`: ALMASim source directory
- `DATA_DIR`: Data directory path
- `OUTPUT_DIR`: Simulation outputs directory
- `TNG_DIR`: TNG100-1 data directory
- `GALAXY_ZOO_DIR`: Galaxy Zoo data directory
- `HUBBLE_DIR`: Hubble data directory

### Frontend
- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)
- `NODE_ENV`: Environment (development/production)

## Health Checks

Both services have health checks configured:

- **Backend**: Checks `/health` endpoint every 30s
- **Frontend**: Checks HTTP response every 30s

View health status:
```bash
make status
# or
make health
```

## Troubleshooting

### Services won't start
```bash
# Check if ports are already in use
lsof -i :3000  # Frontend
lsof -i :8000  # Backend

# View logs for errors
make logs
```

### Frontend can't connect to backend
```bash
# Verify backend is running and healthy
curl http://localhost:8000/health

# Check network connectivity
docker network inspect almasim_almasim-network
```

### Hot reload not working (development mode)
```bash
# Ensure source code is mounted correctly
docker-compose exec frontend ls -la /app/src

# Restart frontend service
make restart-frontend
```

### Clean slate (remove everything)
```bash
# Stop all services and remove volumes
make clean

# Rebuild images from scratch
make build
```

## Performance Tips

### Development
- Use `make start-detached` to run in background
- Monitor resource usage: `docker stats`
- Frontend hot reload only watches mounted files

### Production
- Production images are optimized for size and performance
- Resource limits prevent runaway processes
- Consider using nginx as a reverse proxy for better performance

## Security Notes

- Data directory is mounted **read-only** by default
- Output directory is writable for simulation results
- No sensitive environment variables are included
- Consider using `.env` file for secrets (not committed to git)

## Updating

### Update dependencies
```bash
# Backend
cd backend && pip freeze > requirements.txt

# Frontend
cd frontend && npm update

# Rebuild images
make build
```

### Update Docker images
```bash
# Pull latest base images
docker-compose pull

# Rebuild with latest dependencies
make build --no-cache
```

## Support

For issues related to:
- Docker setup: Check this file and `docker-compose.yml`
- Frontend: See `frontend/MIGRATION.md`
- Backend: Check `backend/README.md` (if available)
- ALMASim core: See main project documentation
