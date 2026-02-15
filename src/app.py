"""Point d'entrée principal de l'application Streamlit Orchestr'IA."""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.utils.config import load_env, load_default_config
from src.utils.logger import ActivityLog, setup_logging

# Charger les variables d'environnement
load_env()
setup_logging()

# Configuration de la page
st.set_page_config(
    page_title="Orchestr'IA",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialise l'état de session Streamlit."""
    defaults = {
        "config": load_default_config(),
        "activity_log": ActivityLog(),
        "current_project": None,
        "project_state": None,
        "current_page": "accueil",
        "provider": None,
        "cost_tracker": None,
        "checkpoint_manager": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --- Sidebar ---
def render_sidebar():
    """Affiche la sidebar de navigation."""
    with st.sidebar:
        st.markdown("## Orchestr'IA")
        st.markdown("---")

        # Navigation
        st.markdown("### Navigation")
        pages = {
            "accueil": "Accueil",
            "configuration": "Configuration",
            "acquisition": "Acquisition du corpus",
            "plan": "Plan du document",
            "generation": "Génération",
            "export": "Export",
        }

        for page_id, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id

        st.markdown("---")

        # Indicateur de progression
        if st.session_state.project_state:
            state = st.session_state.project_state
            st.markdown("### Progression")
            steps = ["init", "plan", "corpus", "generation", "review", "export", "done"]
            current_idx = steps.index(state.current_step) if state.current_step in steps else 0
            progress = current_idx / (len(steps) - 1)
            st.progress(progress, text=f"Étape : {state.current_step}")

            if state.plan:
                total = len(state.plan.sections)
                done = len(state.generated_sections)
                st.metric("Sections générées", f"{done}/{total}")

        # Mode d'exécution
        if st.session_state.project_state:
            mode = st.session_state.project_state.config.get("mode", "manual")
            st.markdown(f"**Mode :** {'Agentique' if mode == 'agentic' else 'Manuel'}")

        # Info fournisseur
        st.markdown("---")
        st.markdown("### Fournisseur IA")
        provider = st.session_state.get("provider")
        if provider and provider.is_available():
            label = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Gemini"}.get(provider.name, provider.name)
            st.success(f"{label} connecté")
        else:
            st.warning("Non configuré")


render_sidebar()


# --- Routeur de pages ---
page = st.session_state.current_page

if page == "accueil":
    from src.pages.page_accueil import render
    render()
elif page == "configuration":
    from src.pages.page_configuration import render
    render()
elif page == "acquisition":
    from src.pages.page_acquisition import render
    render()
elif page == "plan":
    from src.pages.page_plan import render
    render()
elif page == "generation":
    from src.pages.page_generation import render
    render()
elif page == "export":
    from src.pages.page_export import render
    render()
