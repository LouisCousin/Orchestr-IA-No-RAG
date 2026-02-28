"""Route WebSocket pour le monitoring temps réel (Phase 3 Sprint 3).

Endpoint : WS /api/v1/projects/{id}/ws

Le client se connecte au WebSocket et reçoit les trames JSON émises
par le MessageBus du MultiAgentOrchestrator au fil de l'eau :
  {"type": "agent_start",  "section": "1.1", "agent": "writer"}
  {"type": "agent_success", "section": "1.1", "cost": 0.04, "tokens": 1200}
  {"type": "dag_error",     "section": "2.0", "message": "API Timeout"}
  {"type": "generation_progress", "progress": 0.5, "section": "1.2", ...}
  {"type": "generation_complete", "cost": 1.23, ...}
  {"type": "state_update",  "data": {...}}

À la connexion, le serveur envoie immédiatement l'état courant du projet.
"""

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.project_state import ProjectStore

logger = logging.getLogger("orchestria")

router = APIRouter(tags=["websocket"])


def _get_store() -> ProjectStore:
    from src.api.main import app
    return app.state.project_store


# Registre global des bus par projet (alimenté par la route /generate)
_project_buses: dict[str, Any] = {}


def register_bus(project_id: str, bus) -> None:
    """Enregistre le MessageBus d'un projet pour le relais WebSocket."""
    _project_buses[project_id] = bus


def unregister_bus(project_id: str) -> None:
    """Désenregistre le MessageBus d'un projet."""
    _project_buses.pop(project_id, None)


def get_bus(project_id: str):
    """Récupère le MessageBus d'un projet."""
    return _project_buses.get(project_id)


@router.websocket("/api/v1/projects/{project_id}/ws")
async def project_websocket(websocket: WebSocket, project_id: str):
    """Endpoint WebSocket pour le suivi temps réel d'un projet.

    Protocole :
    1. Le serveur envoie un message 'state_update' avec l'état courant.
    2. Le serveur relaie tous les événements du MessageBus.
    3. Le client peut envoyer {"type": "ping"} ; le serveur répond {"type": "pong"}.
    4. La déconnexion du client ne stoppe pas la génération en cours.
    """
    store = _get_store()
    if not store.project_exists(project_id):
        await websocket.close(code=4004, reason="Projet introuvable")
        return

    await websocket.accept()
    logger.info(f"WS connecté : projet {project_id}")

    # Envoyer l'état initial
    try:
        state_dict = store.get_state_dict(project_id)
        await websocket.send_json({
            "type": "state_update",
            "data": state_dict,
            "timestamp": time.time(),
        })
    except Exception as exc:
        logger.warning(f"WS: erreur envoi état initial: {exc}")

    # Créer le listener pour relayer les événements du bus
    async def ws_listener(event: dict) -> None:
        try:
            await websocket.send_json(event)
        except Exception:
            raise  # Sera capté par le bus pour retirer le listener

    # Enregistrer le listener sur le bus du projet (s'il existe)
    bus = get_bus(project_id)
    if bus:
        bus.add_ws_listener(ws_listener)

    try:
        while True:
            # Écouter les messages du client (ping, etc.)
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                data = json.loads(raw)
                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
                elif msg_type == "get_state":
                    state_dict = store.get_state_dict(project_id)
                    await websocket.send_json({
                        "type": "state_update",
                        "data": state_dict,
                        "timestamp": time.time(),
                    })

            except asyncio.TimeoutError:
                # Timeout de réception : envoyer un heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
                except Exception:
                    break

                # Vérifier si un bus a été enregistré entre-temps
                new_bus = get_bus(project_id)
                if new_bus and new_bus is not bus:
                    if bus:
                        bus.remove_ws_listener(ws_listener)
                    bus = new_bus
                    bus.add_ws_listener(ws_listener)

    except WebSocketDisconnect:
        logger.info(f"WS déconnecté : projet {project_id}")
    except Exception as exc:
        logger.warning(f"WS erreur : {exc}")
    finally:
        if bus:
            bus.remove_ws_listener(ws_listener)
        logger.info(f"WS nettoyé : projet {project_id}")
