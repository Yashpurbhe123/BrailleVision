"""
═══════════════════════════════════
📄 FILE 18/42: backend/main.py
═══════════════════════════════════

BrailleVision AI — FastAPI Application Entrypoint
Wires together the lifecycles, database, middleware, and API routers.
"""

from __future__ import annotations

import torch
import logging
import os
import sys

# Silence sklearn/joblib loky physical cores warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from dotenv import load_dotenv  # type: ignore

from database.db import init_db  # type: ignore
from routers import scan, tts, translate, history  # type: ignore

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT & LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Load env variables from .env
load_dotenv()

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("braillevision")

# ─────────────────────────────────────────────────────────────
# LIFECYCLE (STARTUP/SHUTDOWN)
# ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycles."""
    logger.info("Initializing BrailleVision AI database...")
    try:
        await init_db()
        logger.info("Database initialized successfully.")
    except Exception as exc:
        logger.critical("Failed to initialize database: %s", exc)
        raise SystemExit(1) from exc

    # Ensure local directory for audio exports exists
    os.makedirs("./data", exist_ok=True)
    logger.info("Data export directory verified at './data'.")

    yield

    logger.info("Shutting down BrailleVision AI server...")


# ─────────────────────────────────────────────────────────────
# APP CREATION & MIDDLEWARE
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="BrailleVision AI API",
    description=(
        "Backend service for Braille detection, Grade 1/2 decoding, "
        "AI text correction, neural translation, and TTS generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration to support mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# ROUTER REGISTRATION
# ─────────────────────────────────────────────────────────────

app.include_router(scan.router)
app.include_router(tts.router)
app.include_router(translate.router)
app.include_router(history.router)

# ─────────────────────────────────────────────────────────────
# APP HEALTH / GENERAL ENDPOINTS
# ─────────────────────────────────────────────────────────────


@app.get("/", summary="Root endpoint")
async def root() -> JSONResponse:
    """Return welcome message and service status."""
    return JSONResponse(
        content={
            "app": "BrailleVision AI Backend",
            "status": "online",
            "version": "1.0.0",
            "docs": "/docs",
        }
    )


@app.get("/health", summary="Health check endpoint")
async def health_check() -> JSONResponse:
    """Perform simple system health check."""
    return JSONResponse(
        content={
            "status": "healthy",
            "database": "connected",
        }
    )


# ─────────────────────────────────────────────────────────────
# EXECUTION ENTRYPOINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn  # type: ignore

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info("Starting uvicorn server on %s:%d", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=True)
