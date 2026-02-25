"""Point d'entrée principal de l'application Streamlit Orchestr'IA."""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.utils.config import load_env, load_default_config, ROOT_DIR
from src.utils.file_utils import save_json
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

        # Stepper visuel de navigation
        st.markdown("### Pipeline")
        stepper_pages = [
            ("accueil", "Accueil"),
            ("configuration", "1. Configuration"),
            ("acquisition", "2. Acquisition"),
            ("plan", "3. Plan"),
            ("generation", "4. Génération"),
            ("dashboard", "5. Dashboard"),
            ("bibliotheque", "6. Bibliothèque"),
            ("export", "7. Export"),
        ]

        current_page = st.session_state.current_page
        state = st.session_state.get("project_state")
        has_sections = state and state.generated_sections

        for page_id, page_label in stepper_pages:
            # Déterminer si l'étape est accessible
            is_current = page_id == current_page
            is_disabled = (page_id == "export" and not has_sections)

            if is_current:
                label = f"**▸ {page_label}**"
            elif is_disabled:
                label = f"~~{page_label}~~"
            else:
                label = page_label

            if is_disabled:
                st.markdown(f"  {label}", help="Générez des sections avant d'accéder à l'export")
            elif st.button(label, key=f"nav_{page_id}", use_container_width=True):
                # Persister l'état avant navigation
                project_id = st.session_state.get("current_project")
                if state and project_id:
                    state_path = ROOT_DIR / "projects" / project_id / "state.json"
                    save_json(state_path, state.to_dict())
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
elif page == "dashboard":
    from src.pages.page_dashboard import render
    render()
elif page == "bibliotheque":
    from src.pages.page_bibliotheque import render
    render()
elif page == "export":
    from src.pages.page_export import render
    render()
