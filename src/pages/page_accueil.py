"""Page d'accueil — Création et gestion de projets."""

import copy
import os
import streamlit as st
from pathlib import Path
from datetime import datetime

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir, save_json, load_json, sanitize_filename
from src.core.profile_manager import ProfileManager
from src.core.orchestrator import ProjectState
from src.utils.providers_registry import PROVIDERS_INFO, create_provider
from src.core.cost_tracker import CostTracker


PROJECTS_DIR = ROOT_DIR / "projects"


def render():
    st.title("Orchestr'IA")
    st.markdown("*Pipeline intelligent de génération de documents professionnels*")
    st.markdown("---")

    tab_new, tab_existing = st.tabs(["Nouveau projet", "Projets existants"])

    with tab_new:
        _render_new_project()

    with tab_existing:
        _render_existing_projects()


def _render_new_project():
    """Formulaire de création d'un nouveau projet."""
    st.subheader("Créer un nouveau projet")

    col1, col2 = st.columns(2)

    with col1:
        project_name = st.text_input("Nom du projet", placeholder="Mon rapport d'analyse")
        objective = st.text_area(
            "Objectif du document",
            placeholder="Décrivez l'objectif du document à produire...",
            height=100,
        )

    with col2:
        # Sélection du profil
        profile_mgr = ProfileManager()
        profiles = profile_mgr.list_profiles()
        profile_options = ["Aucun (configuration manuelle)"] + [p["name"] for p in profiles]
        selected_profile = st.selectbox("Profil de projet", profile_options)

        target_pages = st.number_input(
            "Taille cible (pages)", min_value=1, max_value=500, value=10, step=1,
            help="Nombre approximatif de pages du document final",
        )

    # Description du profil sélectionné
    if selected_profile != "Aucun (configuration manuelle)":
        selected = next((p for p in profiles if p["name"] == selected_profile), None)
        if selected:
            st.info(f"**{selected['name']}** — {selected['description']}")

    # Bouton de création
    if st.button("Créer le projet", type="primary", use_container_width=True):
        if not project_name:
            st.error("Veuillez saisir un nom de projet.")
            return

        project_id = sanitize_filename(project_name)
        project_dir = PROJECTS_DIR / project_id
        if project_dir.exists():
            st.error("Un projet avec ce nom existe déjà.")
            return

        ensure_dir(project_dir)
        ensure_dir(project_dir / "corpus")

        # Charger la configuration du profil (d'abord le profil, puis les choix utilisateur)
        config = copy.deepcopy(st.session_state.config)

        # Aplatir les paramètres de génération depuis la config par défaut
        # (la config YAML les imbrique sous "generation:" et utilise "default_model")
        gen = config.get("generation", {})
        config.setdefault("model", config.get("default_model", "gpt-4o"))
        config.setdefault("temperature", gen.get("temperature", 0.7))
        config.setdefault("max_tokens", gen.get("max_tokens", 4096))
        config.setdefault("number_of_passes", gen.get("number_of_passes", 1))

        if selected_profile != "Aucun (configuration manuelle)":
            selected = next((p for p in profiles if p["name"] == selected_profile), None)
            if selected:
                profile_config = profile_mgr.get_profile_config(selected["id"])
                config.update(profile_config)

        # Les choix explicites de l'utilisateur priment sur le profil
        config["target_pages"] = target_pages
        config["objective"] = objective

        # Créer l'état du projet
        state = ProjectState(
            name=project_name,
            config=config,
        )
        state_path = project_dir / "state.json"
        save_json(state_path, state.to_dict())

        # Mettre à jour la session
        st.session_state.project_state = state
        st.session_state.current_project = project_id
        st.session_state.current_page = "configuration"
        st.success(f"Projet « {project_name} » créé avec succès !")
        st.rerun()


def _render_existing_projects():
    """Liste les projets existants."""
    st.subheader("Projets existants")

    if not PROJECTS_DIR.exists():
        st.info("Aucun projet existant.")
        return

    projects = []
    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if project_dir.is_dir():
            state_path = project_dir / "state.json"
            if state_path.exists():
                try:
                    data = load_json(state_path)
                    projects.append({
                        "id": project_dir.name,
                        "name": data.get("name", project_dir.name),
                        "step": data.get("current_step", "init"),
                        "created": data.get("created_at", ""),
                        "updated": data.get("updated_at", ""),
                        "sections_done": len(data.get("generated_sections", {})),
                        "total_sections": len(data.get("plan", {}).get("sections", [])),
                    })
                except Exception:
                    pass

    if not projects:
        st.info("Aucun projet existant.")
        return

    for project in projects:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{project['name']}**")
                st.caption(f"Étape : {project['step']} | Sections : {project['sections_done']}/{project['total_sections']}")
            with col2:
                if project["updated"]:
                    st.caption(f"Dernière modification : {project['updated'][:16]}")
            with col3:
                if st.button("Ouvrir", key=f"open_{project['id']}"):
                    _load_project(project["id"])


def _load_project(project_id: str):
    """Charge un projet existant dans la session.

    Tente de restaurer le provider IA automatiquement à partir de la clé API
    en variable d'environnement. Si la restauration échoue, redirige vers
    la page Configuration avec un message explicatif.
    """
    project_dir = PROJECTS_DIR / project_id
    state_path = project_dir / "state.json"

    try:
        data = load_json(state_path)
        state = ProjectState.from_dict(data)
        st.session_state.project_state = state
        st.session_state.current_project = project_id

        # Restaurer le provider automatiquement (C1-E)
        provider_name = state.config.get("default_provider")
        provider_restored = False

        if provider_name:
            info = PROVIDERS_INFO.get(provider_name, {})
            api_key = os.environ.get(info.get("env_var", ""), "")
            if api_key and api_key != info.get("placeholder", ""):
                try:
                    provider = create_provider(provider_name, api_key)
                    if provider.is_available():
                        st.session_state.provider = provider
                        st.session_state.cost_tracker = CostTracker()
                        provider_restored = True
                except ValueError:
                    pass  # Unknown provider — will redirect to configuration page

        if provider_restored:
            # Navigation vers la page appropriée
            step_to_page = {
                "generation": "generation",
                "review": "export",
                "export": "export",
                "done": "export",
                "dashboard": "dashboard",
                "bibliotheque": "bibliotheque",
            }
            st.session_state.current_page = step_to_page.get(state.current_step, "plan")
        else:
            # Pas de provider restaurable → rediriger vers Configuration
            st.session_state.current_page = "configuration"
            st.session_state._restore_message = (
                "Le projet a été restauré. Reconfigurez votre fournisseur IA "
                "pour continuer."
            )

        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors du chargement du projet : {e}")
