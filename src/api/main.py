"""Point d'entrée FastAPI — Orchestr'IA API (Phase 3 Sprint 3).

Lancement :
    uvicorn src.api.main:app --reload

Documentation Swagger :
    http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.project_state import ProjectStore
from src.utils.config import load_env
from src.utils.logger import setup_logging

logger = logging.getLogger("orchestria")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialisation et nettoyage de l'application."""
    # Startup
    load_env()
    setup_logging()
    logger.info("Orchestr'IA API — Démarrage")

    # Initialiser le store de projets
    application.state.project_store = ProjectStore()
    # Registre des providers IA actifs (peuplé via PUT /provider)
    application.state.providers = {}

    yield

    # Shutdown
    logger.info("Orchestr'IA API — Arrêt")


app = FastAPI(
    title="Orchestr'IA API",
    description=(
        "API REST et WebSocket pour le pipeline intelligent de génération "
        "de documents professionnels. Phase 3 Sprint 3 — Architecture découplée."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────

from src.api.routes.projects import router as projects_router  # noqa: E402
from src.api.routes.ws import router as ws_router  # noqa: E402

app.include_router(projects_router)
app.include_router(ws_router)


# ── Health check ─────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health_check():
    """Vérification de l'état de l'API."""
    store: ProjectStore = app.state.project_store
    providers = app.state.providers
    return {
        "status": "ok",
        "projects_count": len(store.list_projects()),
        "providers_configured": list(providers.keys()),
    }
