"""Page de génération — Client HTTP + WebSocket (Phase 3 Sprint 3).

Aucun import de src.core.
La génération est lancée via POST /generate (HTTP 202 Accepted).
Le suivi temps réel se fait via WebSocket.
"""

import json
import logging
import time

import streamlit as st

from src.client.api_client import OrchestrIAClient

logger = logging.getLogger("orchestria")


def _get_client() -> OrchestrIAClient:
    from src.client.app import get_client
    return get_client()


def render():
    st.title("Génération du document")
    st.info(
        "**Étape 4** — Lancez la génération du contenu. L'IA utilise votre corpus "
        "et le plan défini pour produire chaque section via le pipeline multi-agents."
    )
    st.markdown("---")

    project_id = st.session_state.get("current_project")
    if not project_id:
        st.warning("Aucun projet actif.")
        return

    client = _get_client()

    try:
        state = client.get_project_state(project_id)
    except Exception as e:
        st.error(f"Erreur : {e}")
        return

    if not state.get("plan"):
        st.warning("Aucun plan défini. Retournez à l'étape Plan.")
        return

    plan = state["plan"]
    sections = plan.get("sections", [])
    generated = state.get("generated_sections", {})
    is_generating = state.get("is_generating", False)

    # ── Indicateurs ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sections", f"{len(generated)}/{len(sections)}")
    with col2:
        cost_report = state.get("cost_report", {})
        total_cost = cost_report.get("total_cost_usd", 0)
        st.metric("Coût", f"${total_cost:.4f}")
    with col3:
        st.metric("Étape", state.get("current_step", "init").capitalize())

    if sections:
        progress = len(generated) / len(sections) if sections else 0
        st.progress(progress, text=f"{len(generated)}/{len(sections)} sections générées")

    st.markdown("---")

    # ── Lancement de la génération ──
    if is_generating:
        st.warning("Une génération est en cours pour ce projet.")
        _render_live_monitoring(client, project_id)
    else:
        col_gen, col_force = st.columns(2)
        with col_gen:
            if st.button("Lancer la génération", type="primary", use_container_width=True):
                _launch_generation(client, project_id, force=False)
        with col_force:
            if generated:
                if st.button("Relancer (forcer)", use_container_width=True):
                    _launch_generation(client, project_id, force=True)

    # ── Sections générées ──
    if generated:
        st.markdown("---")
        st.subheader("Sections générées")

        for section in sections:
            sid = section["id"]
            content = generated.get(sid, "")
            if content:
                with st.expander(f"{section['id']} — {section['title']}", expanded=False):
                    st.markdown(content)
            else:
                st.caption(f"⏳ {section['id']} — {section['title']} (en attente)")

    # ── Navigation ──
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Retour au plan", use_container_width=True):
            st.session_state.current_page = "plan"
            st.rerun()
    with col_next:
        if generated:
            if st.button("Voir le dashboard →", type="primary", use_container_width=True):
                st.session_state.current_page = "dashboard"
                st.rerun()


def _launch_generation(client: OrchestrIAClient, project_id: str, force: bool = False):
    """Lance la génération via l'API."""
    try:
        result = client.launch_generation(project_id, force_restart=force)
        st.success(result.get("message", "Génération lancée !"))
        st.info(f"WebSocket : {result.get('ws_url', '')}")
        _render_live_monitoring(client, project_id)
    except Exception as e:
        error_str = str(e)
        if "409" in error_str:
            st.warning("Une génération est déjà en cours.")
        else:
            st.error(f"Erreur : {e}")


def _render_live_monitoring(client: OrchestrIAClient, project_id: str):
    """Affiche un placeholder pour le monitoring temps réel via WebSocket.

    Le suivi complet se fait via la page Dashboard.
    Un rechargement périodique de l'état via l'API permet un suivi minimal.
    """
    st.markdown("---")
    st.subheader("Suivi en direct")
    st.caption(
        "La génération tourne en arrière-plan sur le serveur. "
        "Vous pouvez fermer cette page sans interrompre le processus. "
        "Consultez le Dashboard pour un suivi temps réel via WebSocket."
    )

    status_placeholder = st.empty()
    progress_bar = st.empty()

    # Polling léger de l'état (le vrai temps réel est sur le Dashboard via WS)
    try:
        state = client.get_project_state(project_id)
        plan = state.get("plan", {})
        sections = plan.get("sections", [])
        generated = state.get("generated_sections", {})

        if sections:
            progress = len(generated) / len(sections)
            progress_bar.progress(progress, text=f"{len(generated)}/{len(sections)}")

        if state.get("is_generating"):
            status_placeholder.info("Génération en cours... Rechargez la page pour voir les mises à jour.")
        else:
            status_placeholder.success("Génération terminée !")
    except Exception:
        status_placeholder.warning("Impossible de récupérer l'état.")
