"""Gestion centralisée de l'état des projets (Phase 3 Sprint 3).

Le backend FastAPI devient le gestionnaire exclusif de l'état des projets.
Un système de verrouillage empêche les exécutions concurrentes sur un même projet.
"""

import asyncio
import copy
import logging
from pathlib import Path
from typing import Optional

from src.core.orchestrator import ProjectState
from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir, load_json, save_json

logger = logging.getLogger("orchestria")

PROJECTS_DIR = ROOT_DIR / "projects"


class ProjectStore:
    """Gestionnaire centralisé des projets avec verrouillage.

    - Charge / sauvegarde les états depuis le disque (state.json).
    - Maintient un cache mémoire des états chargés.
    - Gère un verrou par projet pour empêcher les exécutions concurrentes.
    """

    def __init__(self, projects_dir: Optional[Path] = None):
        self._projects_dir = projects_dir or PROJECTS_DIR
        self._cache: dict[str, ProjectState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._generating: set[str] = set()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _project_dir(self, project_id: str) -> Path:
        return self._projects_dir / project_id

    def _state_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "state.json"

    def _get_lock(self, project_id: str) -> asyncio.Lock:
        if project_id not in self._locks:
            self._locks[project_id] = asyncio.Lock()
        return self._locks[project_id]

    # ── CRUD ─────────────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        """Liste tous les projets existants."""
        if not self._projects_dir.exists():
            return []

        projects = []
        for project_dir in sorted(self._projects_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            state_path = project_dir / "state.json"
            if not state_path.exists():
                continue
            try:
                data = load_json(state_path)
                projects.append({
                    "id": project_dir.name,
                    "name": data.get("name", project_dir.name),
                    "current_step": data.get("current_step", "init"),
                    "sections_generated": len(data.get("generated_sections", {})),
                    "total_sections": len(data.get("plan", {}).get("sections", [])),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                })
            except Exception as exc:
                logger.warning(f"Projet {project_dir.name} illisible : {exc}")
        return projects

    def create_project(self, project_id: str, state: ProjectState) -> Path:
        """Crée un nouveau projet sur le disque."""
        project_dir = self._project_dir(project_id)
        if project_dir.exists():
            raise FileExistsError(f"Le projet {project_id!r} existe déjà")
        ensure_dir(project_dir)
        ensure_dir(project_dir / "corpus")
        save_json(self._state_path(project_id), state.to_dict())
        self._cache[project_id] = state
        return project_dir

    def load_state(self, project_id: str, *, force_reload: bool = False) -> ProjectState:
        """Charge l'état d'un projet (cache mémoire + disque)."""
        if not force_reload and project_id in self._cache:
            return self._cache[project_id]

        state_path = self._state_path(project_id)
        if not state_path.exists():
            raise FileNotFoundError(f"Projet {project_id!r} introuvable")

        data = load_json(state_path)
        state = ProjectState.from_dict(data)
        self._cache[project_id] = state
        return state

    def save_state(self, project_id: str, state: Optional[ProjectState] = None) -> None:
        """Persiste l'état d'un projet sur le disque."""
        if state is None:
            state = self._cache.get(project_id)
        if state is None:
            raise ValueError(f"Aucun état en mémoire pour le projet {project_id!r}")
        save_json(self._state_path(project_id), state.to_dict())
        self._cache[project_id] = state

    def get_state_dict(self, project_id: str) -> dict:
        """Retourne l'état sérialisé d'un projet."""
        state = self.load_state(project_id)
        result = state.to_dict()
        result["is_generating"] = self.is_generating(project_id)
        return result

    def project_exists(self, project_id: str) -> bool:
        return self._state_path(project_id).exists()

    def get_project_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id)

    # ── Verrouillage de génération ───────────────────────────────────────

    def is_generating(self, project_id: str) -> bool:
        """Vérifie si une génération est en cours pour ce projet."""
        return project_id in self._generating

    async def acquire_generation_lock(self, project_id: str) -> bool:
        """Tente d'acquérir le verrou de génération.

        Returns:
            True si le verrou a été acquis, False si une génération est déjà en cours.
        """
        lock = self._get_lock(project_id)
        async with lock:
            if project_id in self._generating:
                return False
            self._generating.add(project_id)
            return True

    async def release_generation_lock(self, project_id: str) -> None:
        """Libère le verrou de génération."""
        lock = self._get_lock(project_id)
        async with lock:
            self._generating.discard(project_id)

    # ── Invalidation du cache ────────────────────────────────────────────

    def invalidate(self, project_id: str) -> None:
        """Invalide le cache mémoire d'un projet."""
        self._cache.pop(project_id, None)

    def invalidate_all(self) -> None:
        """Invalide tout le cache."""
        self._cache.clear()
