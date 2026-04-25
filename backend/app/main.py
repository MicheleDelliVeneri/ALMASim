"""FastAPI application entry point."""

import logging

# Import database initialization
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.core.config import settings

# Add backend directory to path for database imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database.startup import initialize_database_on_startup  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting ALMASim API...")

    # Initialize database
    try:
        data_dir = settings.DATA_DIR
        if not data_dir.is_absolute():
            # Make relative to backend directory
            data_dir = backend_dir / data_dir

        initialize_database_on_startup(data_dir)
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        # Don't fail startup, but log the error

    yield

    # Shutdown
    logger.info("Shutting down ALMASim API...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="ALMASim API for generating realistic ALMA observations",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse({"message": "ALMASim API", "version": settings.VERSION})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({"status": "healthy"})
