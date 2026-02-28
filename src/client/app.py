"""Point d'entrée Streamlit — Client pur HTTP (Phase 3 Sprint 3).

Ce fichier remplace src/app.py pour l'architecture découplée.
Aucun import de src.core n'est présent : toute la logique métier
est consommée via l'API REST (src/api).

Lancement :
    streamlit run src/client/app.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.client.api_client import OrchestrIAClient

# Configuration de la page
st.set_page_config(
    page_title="Orchestr'IA",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"


def get_client() -> OrchestrIAClient:
    """Retourne le client API (singleton par session)."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = OrchestrIAClient(base_url=API_URL)
    return st.session_state.api_client


def init_session_state():
    """Initialise l'état de session Streamlit (léger, pas de logique métier)."""
    defaults = {
        "current_page": "accueil",
        "current_project": None,
        "api_url": API_URL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ── Sidebar ──────────────────────────────────────────────────────────────


def render_sidebar():
    """Affiche la sidebar de navigation."""
    client = get_client()

    with st.sidebar:
        st.markdown("## Orchestr'IA")

        # Indicateur de connexion API
        if client.is_available():
            st.success("API connectée")
        else:
            st.error("API non disponible")
            st.caption(f"URL : {st.session_state.api_url}")
            new_url = st.text_input("URL de l'API", value=st.session_state.api_url)
            if new_url != st.session_state.api_url:
                st.session_state.api_url = new_url
                st.session_state.api_client = OrchestrIAClient(base_url=new_url)
                st.rerun()

        st.markdown("---")

        # Stepper visuel de navigation
        st.markdown("### Pipeline")
        stepper_pages = [
            ("accueil", "Accueil"),
            ("configuration", "1. Configuration"),
            ("acquisition", "2. Acquisition"),
            ("plan", "3. Plan"),
            ("generation", "4. Génération"),
            ("dashboard", "5. Dashboard"),
            ("export", "6. Export"),
        ]

        current_page = st.session_state.current_page
        project_id = st.session_state.get("current_project")

        # Charger l'état du projet si un projet est actif
        project_state = None
        if project_id and client.is_available():
            try:
                project_state = client.get_project_state(project_id)
            except Exception:
                pass

        for page_id, page_label in stepper_pages:
            is_current = page_id == current_page
            has_sections = project_state and project_state.get("generated_sections")
            is_disabled = page_id == "export" and not has_sections

            if is_current:
                label = f"**▸ {page_label}**"
            elif is_disabled:
                label = f"~~{page_label}~~"
            else:
                label = page_label

            if is_disabled:
                st.markdown(
                    f"  {label}",
                    help="Générez des sections avant d'accéder à l'export",
                )
            elif st.button(label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()

        st.markdown("---")

        # Indicateur de progression (depuis l'API)
        if project_state:
            st.markdown("### Progression")
            steps = ["init", "plan", "corpus", "generation", "review", "export", "done"]
            current_step = project_state.get("current_step", "init")
            current_idx = steps.index(current_step) if current_step in steps else 0
            progress = current_idx / (len(steps) - 1)
            st.progress(progress, text=f"Étape : {current_step}")

            plan = project_state.get("plan")
            if plan:
                total = len(plan.get("sections", []))
                done = len(project_state.get("generated_sections", {}))
                st.metric("Sections générées", f"{done}/{total}")

            if project_state.get("is_generating"):
                st.info("Génération en cours...")


render_sidebar()


# ── Routeur de pages ─────────────────────────────────────────────────────

page = st.session_state.current_page

if page == "accueil":
    from src.client.pages.page_accueil import render
    render()
elif page == "configuration":
    from src.client.pages.page_configuration import render
    render()
elif page == "acquisition":
    from src.client.pages.page_acquisition import render
    render()
elif page == "plan":
    from src.client.pages.page_plan import render
    render()
elif page == "generation":
    from src.client.pages.page_generation import render
    render()
elif page == "dashboard":
    from src.client.pages.page_dashboard import render
    render()
elif page == "export":
    from src.client.pages.page_export import render
    render()
