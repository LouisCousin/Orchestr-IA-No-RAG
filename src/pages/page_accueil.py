"""Page d'accueil — Création et gestion de projets."""

import streamlit as st
from pathlib import Path
from datetime import datetime

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir, save_json, load_json, sanitize_filename
from src.core.profile_manager import ProfileManager
from src.core.orchestrator import ProjectState


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

        # Charger la configuration du profil
        config = st.session_state.config.copy()
        config["target_pages"] = target_pages
        config["objective"] = objective

        if selected_profile != "Aucun (configuration manuelle)":
            selected = next((p for p in profiles if p["name"] == selected_profile), None)
            if selected:
                profile_config = profile_mgr.get_profile_config(selected["id"])
                config.update(profile_config)

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
    """Charge un projet existant dans la session."""
    project_dir = PROJECTS_DIR / project_id
    state_path = project_dir / "state.json"

    try:
        data = load_json(state_path)
        state = ProjectState.from_dict(data)
        st.session_state.project_state = state
        st.session_state.current_project = project_id
        st.session_state.current_page = "generation" if state.current_step == "generation" else "plan"
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors du chargement du projet : {e}")
