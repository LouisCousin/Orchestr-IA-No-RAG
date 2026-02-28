"""Client HTTP pour communiquer avec l'API Orchestr'IA (Phase 3 Sprint 3).

Remplace tous les imports directs de src.core par des appels HTTP
vers le backend FastAPI.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger("orchestria")

DEFAULT_API_URL = "http://localhost:8000"


class OrchestrIAClient:
    """Client HTTP synchrone pour l'API Orchestr'IA."""

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    # ── Health ───────────────────────────────────────────────────────────

    def health(self) -> dict:
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def is_available(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    # ── Projets ──────────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        resp = self._client.get("/api/v1/projects")
        resp.raise_for_status()
        return resp.json()

    def create_project(
        self,
        name: str,
        objective: str = "",
        target_pages: int = 10,
        profile: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        config_overrides: Optional[dict] = None,
    ) -> dict:
        payload = {
            "name": name,
            "objective": objective,
            "target_pages": target_pages,
        }
        if profile:
            payload["profile"] = profile
        if provider:
            payload["provider"] = provider
        if api_key:
            payload["api_key"] = api_key
        if config_overrides:
            payload["config_overrides"] = config_overrides

        resp = self._client.post("/api/v1/projects", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_project_state(self, project_id: str) -> dict:
        resp = self._client.get(f"/api/v1/projects/{project_id}/state")
        resp.raise_for_status()
        return resp.json()

    # ── Corpus ───────────────────────────────────────────────────────────

    def upload_corpus_files(self, project_id: str, files: list[tuple[str, bytes]]) -> dict:
        """Upload des fichiers au corpus.

        Args:
            project_id: ID du projet.
            files: Liste de tuples (nom_fichier, contenu_bytes).
        """
        multipart_files = [
            ("files", (name, content)) for name, content in files
        ]
        resp = self._client.post(
            f"/api/v1/projects/{project_id}/corpus/upload",
            files=multipart_files,
        )
        resp.raise_for_status()
        return resp.json()

    def acquire_corpus_urls(
        self,
        project_id: str,
        urls: list[str],
        slow_mode: bool = False,
    ) -> dict:
        resp = self._client.post(
            f"/api/v1/projects/{project_id}/corpus/urls",
            json={"urls": urls, "slow_mode": slow_mode},
        )
        resp.raise_for_status()
        return resp.json()

    # ── Architecte ───────────────────────────────────────────────────────

    def run_architect(self, project_id: str, objective: Optional[str] = None) -> dict:
        payload = {}
        if objective:
            payload["objective"] = objective
        resp = self._client.post(
            f"/api/v1/projects/{project_id}/architect",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def validate_architect(
        self,
        project_id: str,
        approved: bool = True,
        modifications: Optional[str] = None,
    ) -> dict:
        resp = self._client.put(
            f"/api/v1/projects/{project_id}/architect",
            json={"approved": approved, "modifications": modifications},
        )
        resp.raise_for_status()
        return resp.json()

    # ── Génération ───────────────────────────────────────────────────────

    def launch_generation(
        self, project_id: str, force_restart: bool = False
    ) -> dict:
        resp = self._client.post(
            f"/api/v1/projects/{project_id}/generate",
            json={"force_restart": force_restart},
        )
        resp.raise_for_status()
        return resp.json()

    # ── Configuration ────────────────────────────────────────────────────

    def update_config(self, project_id: str, config: dict) -> dict:
        resp = self._client.put(
            f"/api/v1/projects/{project_id}/config",
            json={"config": config},
        )
        resp.raise_for_status()
        return resp.json()

    def configure_provider(
        self,
        project_id: str,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
    ) -> dict:
        payload = {"provider": provider, "api_key": api_key}
        if model:
            payload["model"] = model
        resp = self._client.put(
            f"/api/v1/projects/{project_id}/provider",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    # ── WebSocket URL ────────────────────────────────────────────────────

    def get_ws_url(self, project_id: str) -> str:
        """Retourne l'URL WebSocket pour un projet."""
        ws_base = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{ws_base}/api/v1/projects/{project_id}/ws"
