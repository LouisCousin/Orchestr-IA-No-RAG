"""Page d'accueil — Création et gestion de projets (client HTTP).

Phase 3 Sprint 3 : aucun import de src.core.
Toutes les opérations passent par l'API REST.
"""

import streamlit as st

from src.client.api_client import OrchestrIAClient


def _get_client() -> OrchestrIAClient:
    from src.client.app import get_client
    return get_client()


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
        target_pages = st.number_input(
            "Taille cible (pages)", min_value=1, max_value=500, value=10, step=1,
            help="Nombre approximatif de pages du document final",
        )

        provider = st.selectbox(
            "Fournisseur IA",
            ["openai", "anthropic", "google"],
            index=0,
        )

    api_key = st.text_input(
        "Clé API (session uniquement)",
        type="password",
        help="La clé est transmise au serveur API et n'est pas sauvegardée sur le client.",
    )

    if st.button("Créer le projet", type="primary", use_container_width=True):
        if not project_name:
            st.error("Veuillez saisir un nom de projet.")
            return

        client = _get_client()
        try:
            result = client.create_project(
                name=project_name,
                objective=objective,
                target_pages=target_pages,
                provider=provider,
                api_key=api_key if api_key else None,
            )
            st.session_state.current_project = result["project_id"]
            st.session_state.current_page = "configuration"
            st.success(f"Projet « {project_name} » créé avec succès !")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de la création : {e}")


def _render_existing_projects():
    """Liste les projets existants."""
    st.subheader("Projets existants")

    client = _get_client()
    try:
        projects = client.list_projects()
    except Exception as e:
        st.error(f"Impossible de charger les projets : {e}")
        return

    if not projects:
        st.info("Aucun projet existant.")
        return

    for project in projects:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{project['name']}**")
                st.caption(
                    f"Étape : {project.get('current_step', 'init')} | "
                    f"Sections : {project.get('sections_generated', 0)}/{project.get('total_sections', 0)}"
                )
            with col2:
                if project.get("updated_at"):
                    st.caption(f"Modifié : {project['updated_at'][:16]}")
            with col3:
                if st.button("Ouvrir", key=f"open_{project['id']}"):
                    st.session_state.current_project = project["id"]
                    step = project.get("current_step", "init")
                    step_to_page = {
                        "generation": "generation",
                        "review": "export",
                        "export": "export",
                        "done": "export",
                    }
                    st.session_state.current_page = step_to_page.get(step, "configuration")
                    st.rerun()
