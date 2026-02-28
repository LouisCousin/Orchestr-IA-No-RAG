"""Routes REST pour la gestion des projets (Phase 3 Sprint 3).

Endpoints :
  POST   /api/v1/projects              → Créer un projet
  GET    /api/v1/projects               → Lister les projets
  GET    /api/v1/projects/{id}/state    → État complet du projet
  POST   /api/v1/projects/{id}/corpus   → Uploader documents / URLs
  POST   /api/v1/projects/{id}/architect → Lancer l'Architecte
  PUT    /api/v1/projects/{id}/architect → Valider l'architecture (HITL)
  POST   /api/v1/projects/{id}/generate  → Lancer la génération
  PUT    /api/v1/projects/{id}/config    → Mettre à jour la configuration
  PUT    /api/v1/projects/{id}/provider  → Configurer le fournisseur IA
"""

import asyncio
import copy
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form

from src.api.models import (
    ArchitectRequest,
    ArchitectValidation,
    ConfigUpdate,
    CorpusStatus,
    CorpusURLs,
    GenerateAccepted,
    GenerateRequest,
    PlanOut,
    PlanSectionOut,
    ProjectCreate,
    ProjectCreated,
    ProjectInfo,
    ProjectState as ProjectStateOut,
    ProviderConfig,
)
from src.core.orchestrator import ProjectState
from src.core.project_state import ProjectStore
from src.utils.config import load_default_config
from src.utils.file_utils import sanitize_filename

logger = logging.getLogger("orchestria")

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


def _get_store() -> ProjectStore:
    """Accès au singleton ProjectStore (injecté via app.state)."""
    from src.api.main import app
    return app.state.project_store


def _get_providers() -> dict:
    """Accès au registre des providers actifs."""
    from src.api.main import app
    return app.state.providers


# ── Lister les projets ───────────────────────────────────────────────────


@router.get("", response_model=list[ProjectInfo])
async def list_projects():
    """Liste tous les projets existants."""
    store = _get_store()
    projects = store.list_projects()
    return [ProjectInfo(**p) for p in projects]


# ── Créer un projet ─────────────────────────────────────────────────────


@router.post("", response_model=ProjectCreated, status_code=201)
async def create_project(body: ProjectCreate):
    """Crée un nouveau projet."""
    store = _get_store()

    project_id = sanitize_filename(body.name)
    if store.project_exists(project_id):
        raise HTTPException(status_code=409, detail=f"Le projet '{project_id}' existe déjà")

    # Construire la configuration
    config = copy.deepcopy(load_default_config())
    gen = config.get("generation", {})
    config.setdefault("model", config.get("default_model", "gpt-4o"))
    config.setdefault("temperature", gen.get("temperature", 0.7))
    config.setdefault("max_tokens", gen.get("max_tokens", 4096))
    config.setdefault("number_of_passes", gen.get("number_of_passes", 1))
    config["target_pages"] = body.target_pages
    config["objective"] = body.objective

    # Appliquer le profil si demandé
    if body.profile:
        from src.core.profile_manager import ProfileManager
        manager = ProfileManager()
        profiles = manager.list_profiles()
        selected = next((p for p in profiles if p["name"] == body.profile), None)
        if selected:
            profile_config = manager.get_profile_config(selected["id"])
            config.update(profile_config)

    # Appliquer les surcharges de config
    if body.config_overrides:
        config.update(body.config_overrides)

    # Configurer le provider si fourni
    if body.provider:
        config["default_provider"] = body.provider

    state = ProjectState(
        name=body.name,
        config=config,
    )
    store.create_project(project_id, state)

    # Configurer le provider si clé API fournie
    if body.provider and body.api_key:
        _configure_provider(body.provider, body.api_key)

    return ProjectCreated(project_id=project_id, name=body.name)


# ── État du projet ───────────────────────────────────────────────────────


@router.get("/{project_id}/state", response_model=ProjectStateOut)
async def get_project_state(project_id: str):
    """Récupère l'état complet du projet."""
    store = _get_store()
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable")

    data = store.get_state_dict(project_id)
    # Convertir le plan en format API
    plan_out = None
    if data.get("plan"):
        plan_data = data["plan"]
        sections = [
            PlanSectionOut(
                id=s.get("id", ""),
                title=s.get("title", ""),
                level=s.get("level", 1),
                description=s.get("description", ""),
                page_budget=s.get("page_budget"),
                status=s.get("status", "pending"),
                dependencies=s.get("dependencies", []),
            )
            for s in plan_data.get("sections", [])
        ]
        plan_out = PlanOut(
            title=plan_data.get("title", ""),
            objective=plan_data.get("objective", ""),
            sections=sections,
        )

    return ProjectStateOut(
        name=data.get("name", ""),
        current_step=data.get("current_step", "init"),
        plan=plan_out,
        generated_sections=data.get("generated_sections", {}),
        section_summaries=data.get("section_summaries", []),
        cost_report=data.get("cost_report", {}),
        quality_reports=data.get("quality_reports", {}),
        factcheck_reports=data.get("factcheck_reports", {}),
        rag_coverage=data.get("rag_coverage", {}),
        deferred_sections=data.get("deferred_sections", []),
        config=data.get("config", {}),
        is_generating=data.get("is_generating", False),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )


# ── Corpus ───────────────────────────────────────────────────────────────


@router.post("/{project_id}/corpus/upload", response_model=CorpusStatus)
async def upload_corpus_files(
    project_id: str,
    files: list[UploadFile] = File(...),
):
    """Upload de fichiers dans le corpus du projet."""
    store = _get_store()
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable")

    corpus_dir = store.get_project_dir(project_id) / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    from src.core.corpus_acquirer import CorpusAcquirer, AcquisitionReport
    acquirer = CorpusAcquirer(corpus_dir)
    report = AcquisitionReport()

    for uploaded in files:
        safe_name = sanitize_filename(uploaded.filename or "document")
        temp_path = corpus_dir / f"_temp_{safe_name}"
        content = await uploaded.read()
        temp_path.write_bytes(content)
        acquirer.acquire_local_files([temp_path], report)
        if temp_path.exists():
            temp_path.unlink()

    return CorpusStatus(
        total_documents=report.successful + report.failed,
        total_tokens=0,
        successful=report.successful,
        failed=report.failed,
        details=[{"source": s.source, "status": s.status, "message": s.message}
                 for s in report.statuses],
    )


@router.post("/{project_id}/corpus/urls", response_model=CorpusStatus)
async def acquire_corpus_urls(project_id: str, body: CorpusURLs):
    """Acquiert des documents depuis des URLs."""
    store = _get_store()
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable")

    corpus_dir = store.get_project_dir(project_id) / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    state = store.load_state(project_id)
    config = state.config or {}
    acq_config = config.get("corpus_acquisition", {})

    if body.slow_mode:
        conn_timeout = acq_config.get("slow_mode_connection_timeout", 30)
        read_timeout = acq_config.get("slow_mode_read_timeout", 120)
    else:
        conn_timeout = acq_config.get("connection_timeout", 15)
        read_timeout = acq_config.get("read_timeout", 60)

    from src.core.corpus_acquirer import CorpusAcquirer, AcquisitionReport
    acquirer = CorpusAcquirer(
        corpus_dir=corpus_dir,
        connection_timeout=conn_timeout,
        read_timeout=read_timeout,
        throttle_delay=acq_config.get("throttle_delay", 1.0),
        user_agent=acq_config.get("user_agent", "Mozilla/5.0"),
    )
    report = AcquisitionReport()
    acquirer.acquire_urls(body.urls, report)
    acquirer.close()

    return CorpusStatus(
        total_documents=report.successful + report.failed,
        total_tokens=0,
        successful=report.successful,
        failed=report.failed,
        details=[{"source": s.source, "status": s.status, "message": s.message}
                 for s in report.statuses],
    )


# ── Architecte ───────────────────────────────────────────────────────────


@router.post("/{project_id}/architect")
async def run_architect(project_id: str, body: ArchitectRequest = None):
    """Lance l'Architecte Agent pour analyser le plan et les dépendances."""
    store = _get_store()
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable")

    state = store.load_state(project_id)
    providers = _get_providers()
    if not providers:
        raise HTTPException(status_code=400, detail="Aucun fournisseur IA configuré")

    if body and body.objective:
        state.config["objective"] = body.objective

    if not state.plan:
        raise HTTPException(status_code=400, detail="Aucun plan défini pour ce projet")

    from src.core.agent_framework import AgentConfig
    from src.core.multi_agent_orchestrator import MultiAgentOrchestrator

    agent_config = AgentConfig.from_config(state.config)
    orchestrator = MultiAgentOrchestrator(
        project_state=state,
        agent_config=agent_config,
        providers=providers,
    )

    architecture = await orchestrator.run_architect_phase()
    state.agent_architecture = architecture
    store.save_state(project_id, state)

    return {"architecture": architecture, "message": "Architecture générée. Validez via PUT."}


@router.put("/{project_id}/architect")
async def validate_architect(project_id: str, body: ArchitectValidation):
    """Valide ou modifie l'architecture proposée (HITL)."""
    store = _get_store()
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable")

    state = store.load_state(project_id)

    if not state.agent_architecture:
        raise HTTPException(status_code=400, detail="Aucune architecture en attente de validation")

    if not body.approved:
        state.agent_architecture = None
        state.current_step = "plan"
        store.save_state(project_id, state)
        return {"message": "Architecture rejetée. Relancez POST /architect."}

    if body.modifications:
        from src.core.plan_parser import PlanParser
        parser = PlanParser()
        new_plan = parser.parse_text(body.modifications)
        if state.config.get("objective"):
            new_plan.objective = state.config["objective"]
        target_pages = state.config.get("target_pages")
        if target_pages:
            parser.distribute_page_budget(new_plan, target_pages)
        state.plan = new_plan

    state.current_step = "generation"
    store.save_state(project_id, state)

    return {"message": "Architecture validée. Lancez POST /generate."}


# ── Génération ───────────────────────────────────────────────────────────


async def _run_generation_task(project_id: str) -> None:
    """Tâche de fond : exécute le pipeline multi-agents de génération."""
    from src.api.routes.ws import register_bus, unregister_bus

    store = _get_store()
    try:
        state = store.load_state(project_id, force_reload=True)
        providers = _get_providers()

        from src.core.agent_framework import AgentConfig
        from src.core.multi_agent_orchestrator import MultiAgentOrchestrator

        agent_config = AgentConfig.from_config(state.config)
        orchestrator = MultiAgentOrchestrator(
            project_state=state,
            agent_config=agent_config,
            providers=providers,
        )

        # Enregistrer le MessageBus pour le relais WebSocket
        register_bus(project_id, orchestrator.bus)

        architecture = state.agent_architecture or orchestrator._default_architecture()
        result = await orchestrator.run_generation_phase(architecture)

        # Mettre à jour l'état avec les résultats
        state.generated_sections.update(result.sections)
        state.agent_verif_reports = result.verif_reports
        state.agent_eval_result = result.eval_result
        state.cost_report = orchestrator._cost_tracker.report.to_dict()
        state.current_step = "review"
        store.save_state(project_id, state)

        logger.info(
            f"Génération terminée pour {project_id} : "
            f"{len(result.sections)} sections, "
            f"coût=${result.total_cost_usd:.4f}"
        )

    except Exception as exc:
        logger.error(f"Erreur génération {project_id}: {exc}")
    finally:
        unregister_bus(project_id)
        await store.release_generation_lock(project_id)


@router.post("/{project_id}/generate", response_model=GenerateAccepted, status_code=202)
async def launch_generation(
    project_id: str,
    body: GenerateRequest = None,
    background_tasks: BackgroundTasks = None,
):
    """Lance la génération multi-agents en arrière-plan.

    Retourne immédiatement HTTP 202 (Accepted). Le client peut suivre
    la progression via WebSocket sur /api/v1/projects/{id}/ws.
    """
    store = _get_store()
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable")

    state = store.load_state(project_id)
    if not state.plan:
        raise HTTPException(status_code=400, detail="Aucun plan défini pour ce projet")

    providers = _get_providers()
    if not providers:
        raise HTTPException(status_code=400, detail="Aucun fournisseur IA configuré")

    # Vérifier le verrou de génération (HTTP 409 si déjà en cours)
    force = body.force_restart if body else False
    if not force:
        acquired = await store.acquire_generation_lock(project_id)
        if not acquired:
            raise HTTPException(
                status_code=409,
                detail="Une génération est déjà en cours pour ce projet",
            )
    else:
        await store.release_generation_lock(project_id)
        await store.acquire_generation_lock(project_id)

    state.current_step = "generation"
    store.save_state(project_id, state)

    # Lancer la génération en arrière-plan
    asyncio.create_task(_run_generation_task(project_id))

    return GenerateAccepted(
        project_id=project_id,
        ws_url=f"/api/v1/projects/{project_id}/ws",
    )


# ── Configuration ────────────────────────────────────────────────────────


@router.put("/{project_id}/config")
async def update_config(project_id: str, body: ConfigUpdate):
    """Met à jour la configuration du projet."""
    store = _get_store()
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable")

    state = store.load_state(project_id)
    state.config.update(body.config)
    store.save_state(project_id, state)

    return {"message": "Configuration mise à jour"}


@router.put("/{project_id}/provider")
async def configure_provider(project_id: str, body: ProviderConfig):
    """Configure le fournisseur IA pour le projet."""
    store = _get_store()
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable")

    _configure_provider(body.provider, body.api_key)

    state = store.load_state(project_id)
    state.config["default_provider"] = body.provider
    if body.model:
        state.config["model"] = body.model
    store.save_state(project_id, state)

    return {"message": f"Fournisseur {body.provider} configuré"}


def _configure_provider(provider_name: str, api_key: str) -> None:
    """Configure un provider IA dans le registre global."""
    from src.utils.providers_registry import create_provider
    provider = create_provider(provider_name, api_key)
    if not provider or not provider.is_available():
        raise HTTPException(status_code=400, detail=f"Clé API invalide pour {provider_name}")
    providers = _get_providers()
    providers[provider_name] = provider
