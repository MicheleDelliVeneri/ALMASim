# ALMASim Docker Quick Start

This README provides quick instructions for running ALMASim using Docker.

## TL;DR

```bash
# Development mode (with hot reload)
make start

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Installation

1. **Install Docker Desktop** (includes Docker Compose)
   - macOS: https://docs.docker.com/desktop/install/mac-install/
   - Windows: https://docs.docker.com/desktop/install/windows-install/
   - Linux: https://docs.docker.com/desktop/install/linux-install/

2. **Clone the repository** (if not already done)
   ```bash
   git clone <repository-url>
   cd ALMASim
   ```

3. **Start the services**
   ```bash
   make start
   ```

That's it! The first time will take a few minutes to build the images.

## Common Commands

```bash
# Start services (foreground, see logs in real-time)
make start

# Start services (background)
make start-detached

# View logs
make logs                   # All services
make logs-backend           # Backend only
make logs-frontend          # Frontend only

# Stop services
make stop

# Restart a specific service
make restart-backend
make restart-frontend

# Check status
make status

# Clean up everything
make clean
```

## What Gets Started?

- **Frontend (Svelte 5 + SvelteKit)**: Port 3000
  - Modern UI with hot reload in development
  - Connects to backend API automatically
  
- **Backend (FastAPI)**: Port 8000
  - RESTful API for simulations, metadata, and visualizations
  - Interactive docs at http://localhost:8000/docs
  - WebSocket support for real-time updates

## Accessing the Application

Once services are running:

1. **Open your browser**: http://localhost:3000
2. **Navigate the app**:
   - Home: Overview and API status
   - Simulations: Create and monitor ALMA simulations
   - Metadata: Query and manage ALMA observation metadata
   - Visualizer: View and analyze datacubes

## Development vs Production

### Development Mode (default)
```bash
make start              # or make start-detached
```
- Hot reload for frontend (instant updates when you edit code)
- Source code mounted as volumes
- Detailed logging
- Better for active development

### Production Mode
```bash
make start-prod         # or make start-prod-detached
```
- Optimized production builds
- Smaller image sizes
- Resource limits enforced
- Better for testing production deployment

## Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
lsof -i :3000  # Frontend
lsof -i :8000  # Backend

# Kill the process or change ports in docker-compose.yml
```

### Services Won't Start
```bash
# Check logs for errors
make logs

# Try rebuilding images
make build

# Clean slate
make clean
make build
make start
```

### Frontend Can't Connect to Backend
- Make sure backend is healthy: `curl http://localhost:8000/health`
- Check the API URL in frontend logs: should be `http://localhost:8000`
- Verify both services are on the same network: `docker network ls`

### Need More Help?
- See detailed documentation: `DOCKER_SETUP.md`
- Check service logs: `make logs`
- View container status: `make status`

## File Structure

```
ALMASim/
├── Makefile                    # Commands for docker-compose
├── docker-compose.yml          # Development configuration
├── docker-compose.prod.yml     # Production configuration
├── DOCKER_SETUP.md             # Detailed Docker documentation
├── backend/
│   ├── Dockerfile              # Backend container definition
│   └── app/                    # FastAPI application
└── frontend/
    ├── Dockerfile              # Production frontend container
    ├── Dockerfile.dev          # Development frontend container
    └── src/                    # Svelte application source
```

## Next Steps

1. **Start the services**: `make start`
2. **Open the app**: http://localhost:3000
3. **Check API docs**: http://localhost:8000/docs
4. **Read detailed docs**: `DOCKER_SETUP.md`
5. **Start developing**: Edit files in `frontend/src/` and see changes instantly!

## Additional Resources

- Frontend migration docs: `frontend/MIGRATION.md`
- Detailed Docker setup: `DOCKER_SETUP.md`
- Makefile help: `make help`

## Quick Reference Card

| Command | Description |
|---------|-------------|
| `make start` | Start dev services (foreground) |
| `make start-detached` | Start dev services (background) |
| `make stop` | Stop all services |
| `make restart` | Restart all services |
| `make logs` | Follow all logs |
| `make status` | Show service status |
| `make health` | Check health endpoints |
| `make clean` | Remove everything |
| `make help` | Show all commands |

---

**Need help?** Run `make help` for a complete list of commands.
