"""Schémas Pydantic pour l'API Orchestr'IA (Phase 3 Sprint 3).

Définit strictement les modèles d'entrée/sortie pour la documentation
Swagger/OpenAPI générée automatiquement par FastAPI.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Projets ──────────────────────────────────────────────────────────────────


class ProjectCreate(BaseModel):
    """Requête de création d'un nouveau projet."""

    name: str = Field(..., min_length=1, max_length=200, description="Nom du projet")
    objective: str = Field("", description="Objectif du document à produire")
    target_pages: int = Field(10, ge=1, le=500, description="Nombre de pages cibles")
    profile: Optional[str] = Field(None, description="Nom du profil à appliquer")
    provider: Optional[str] = Field(None, description="Fournisseur IA (openai, anthropic, google)")
    api_key: Optional[str] = Field(None, description="Clé API du fournisseur (session uniquement)")
    config_overrides: Optional[dict[str, Any]] = Field(
        None, description="Paramètres de configuration à fusionner"
    )


class ProjectInfo(BaseModel):
    """Résumé d'un projet."""

    id: str
    name: str
    current_step: str
    sections_generated: int = 0
    total_sections: int = 0
    created_at: str = ""
    updated_at: str = ""


class ProjectCreated(BaseModel):
    """Réponse après création d'un projet."""

    project_id: str
    name: str
    message: str = "Projet créé avec succès"


# ── Corpus ───────────────────────────────────────────────────────────────────


class CorpusURLs(BaseModel):
    """Liste d'URLs à acquérir dans le corpus."""

    urls: list[str] = Field(..., min_length=1, description="URLs à télécharger")
    slow_mode: bool = Field(False, description="Mode sites lents (timeouts étendus)")


class CorpusStatus(BaseModel):
    """Statut du corpus après acquisition."""

    total_documents: int
    total_tokens: int
    successful: int
    failed: int
    details: list[dict[str, Any]] = Field(default_factory=list)


# ── Plan / Architecte ───────────────────────────────────────────────────────


class ArchitectRequest(BaseModel):
    """Requête pour lancer l'Architecte Agent."""

    objective: Optional[str] = Field(None, description="Objectif (écrase celui du projet)")


class ArchitectValidation(BaseModel):
    """Validation HITL de l'architecture proposée."""

    approved: bool = Field(..., description="True pour valider, False pour rejeter")
    modifications: Optional[str] = Field(
        None, description="Texte du plan modifié (si modifications souhaitées)"
    )


class PlanSectionOut(BaseModel):
    """Section du plan."""

    id: str
    title: str
    level: int = 1
    description: str = ""
    page_budget: Optional[float] = None
    status: str = "pending"
    dependencies: list[str] = Field(default_factory=list)


class PlanOut(BaseModel):
    """Plan complet du document."""

    title: str = ""
    objective: str = ""
    sections: list[PlanSectionOut] = Field(default_factory=list)


# ── Génération ───────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Requête pour lancer la génération via le DAG multi-agents."""

    force_restart: bool = Field(
        False, description="Forcer le redémarrage même si une génération est en cours"
    )


class GenerateAccepted(BaseModel):
    """Réponse 202 Accepted après lancement de la génération."""

    message: str = "Génération lancée en arrière-plan"
    project_id: str
    ws_url: str = Field("", description="URL WebSocket pour le suivi temps réel")


# ── État du projet ───────────────────────────────────────────────────────────


class ProjectState(BaseModel):
    """État complet du projet exposé via l'API."""

    name: str
    current_step: str = "init"
    plan: Optional[PlanOut] = None
    generated_sections: dict[str, str] = Field(default_factory=dict)
    section_summaries: list[str] = Field(default_factory=list)
    cost_report: dict[str, Any] = Field(default_factory=dict)
    quality_reports: dict[str, Any] = Field(default_factory=dict)
    factcheck_reports: dict[str, Any] = Field(default_factory=dict)
    rag_coverage: dict[str, Any] = Field(default_factory=dict)
    deferred_sections: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    is_generating: bool = False
    created_at: str = ""
    updated_at: str = ""


# ── WebSocket Events ────────────────────────────────────────────────────────


class WSEvent(BaseModel):
    """Événement WebSocket envoyé au client."""

    type: str = Field(
        ...,
        description="Type d'événement (agent_start, agent_success, dag_error, "
        "generation_progress, generation_complete, state_update)",
    )
    section: Optional[str] = Field(None, description="ID de la section concernée")
    agent: Optional[str] = Field(None, description="Nom de l'agent")
    cost: Optional[float] = Field(None, description="Coût en USD")
    tokens: Optional[int] = Field(None, description="Nombre de tokens")
    message: Optional[str] = Field(None, description="Message descriptif")
    progress: Optional[float] = Field(None, description="Progression 0.0-1.0")
    data: Optional[dict[str, Any]] = Field(None, description="Données supplémentaires")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


# ── Configuration ────────────────────────────────────────────────────────────


class ProviderConfig(BaseModel):
    """Configuration du fournisseur IA."""

    provider: str = Field(..., description="Nom du fournisseur (openai, anthropic, google)")
    api_key: str = Field(..., description="Clé API")
    model: Optional[str] = Field(None, description="Modèle à utiliser")


class ConfigUpdate(BaseModel):
    """Mise à jour partielle de la configuration."""

    config: dict[str, Any] = Field(..., description="Paramètres à fusionner")


# ── Erreurs ──────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée."""

    detail: str
    error_code: Optional[str] = None
