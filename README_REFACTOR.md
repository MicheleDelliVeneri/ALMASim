# ALMASim Refactored Architecture

This document describes the new FastAPI + SolidStart architecture for ALMASim.

## Project Structure

```
ALMASim/
├── almasim/              # Core ALMASim library (unchanged)
│   ├── services/         # Business logic services
│   ├── skymodels/       # Refactored sky model classes
│   └── ...
├── backend/              # FastAPI backend
│   └── app/
│       ├── api/         # API routes
│       │   └── v1/
│       │       └── routers/
│       ├── core/        # Configuration and dependencies
│       ├── schemas/     # Pydantic models
│       └── services/    # Business logic services
└── frontend/            # SolidStart frontend
    └── app/
        ├── lib/         # Shared libraries
        │   └── api/     # API client
        └── routes/      # SolidStart routes
```

## Backend (FastAPI)

### Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Architecture

- **Routers** (`app/api/v1/routers/`): Define API endpoints
- **Schemas** (`app/schemas/`): Pydantic models for request/response validation
- **Services** (`app/services/`): Business logic that wraps core ALMASim functionality
- **Core** (`app/core/`): Configuration and shared dependencies

## Frontend (SvelteKit)

### Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```

4. Open browser: http://localhost:5173

### Architecture

- **Routes** (`src/routes/`): SvelteKit pages
- **API Client** (`src/lib/api/`): TypeScript client for backend API
- **Components** (`src/components/`): Reusable Svelte components
- **Styling**: Tailwind CSS for modern, responsive design

## Development Workflow

1. Start backend:
```bash
cd backend
uvicorn app.main:app --reload
```

2. Start frontend (in another terminal):
```bash
cd frontend
npm run dev
```

3. Access:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Key Features

### Backend
- ✅ RESTful API with FastAPI
- ✅ Pydantic schemas for validation
- ✅ Async support
- ✅ CORS configured for frontend
- ✅ Modular architecture
- ✅ Background task support for long-running simulations

### Frontend
- ✅ Modern SolidStart with TypeScript
- ✅ Tailwind CSS for styling
- ✅ Type-safe API client
- ✅ Responsive design
- ✅ Real-time status updates (to be implemented)

## Next Steps

1. **Add authentication** (if needed)
2. **Implement WebSocket** for real-time simulation progress
3. **Add database** for simulation history and status tracking
4. **Enhance frontend** with more features:
   - Simulation list/view
   - Progress tracking
   - Results visualization
   - Metadata browser
5. **Add tests** for both backend and frontend
6. **Dockerize** the application

## Migration Notes

The core ALMASim library (`almasim/`) remains unchanged and can still be used directly in Python scripts, notebooks, or other applications. The FastAPI backend provides a REST API wrapper around the existing services.

