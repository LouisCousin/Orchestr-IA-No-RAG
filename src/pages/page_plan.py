"""Page de gestion du plan du document."""

import streamlit as st
from pathlib import Path

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir, save_json
from src.core.plan_parser import PlanParser, NormalizedPlan
from src.core.text_extractor import extract
from src.core.checkpoint_manager import CheckpointManager, CheckpointConfig, CheckpointType


PROJECTS_DIR = ROOT_DIR / "projects"


def render():
    st.title("Plan du document")
    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif. Créez ou ouvrez un projet depuis la page Accueil.")
        return

    state = st.session_state.project_state

    # Si un plan existe déjà, afficher directement la vue/modification
    if state.plan:
        _render_view_plan(state)
    else:
        # Sinon, proposer import ou génération
        _render_create_plan(state)


def _render_create_plan(state):
    """Interface unifiée pour créer un plan (import ou génération)."""
    st.subheader("Créer le plan du document")

    target_pages = state.config.get("target_pages") or 10
    st.info(f"Taille cible du document : **{target_pages} pages**")

    # Import depuis fichier ou texte
    with st.expander("Importer un plan existant", expanded=True):
        _render_import_plan(state)

    # Génération IA
    with st.expander("Générer un plan avec l'IA"):
        _render_generate_plan(state)


def _render_import_plan(state):
    """Import d'un plan depuis un fichier."""
    plan_file = st.file_uploader(
        "Fichier du plan",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx"],
        help="Le plan sera analysé et normalisé automatiquement.",
    )

    plan_text = st.text_area(
        "Ou saisir/coller le plan directement",
        height=200,
        placeholder="1. Introduction\n1.1 Contexte\n1.2 Objectifs\n2. Analyse\n2.1 État des lieux\n...",
    )

    if st.button("Normaliser le plan", type="primary"):
        parser = PlanParser()

        if plan_file:
            project_id = st.session_state.current_project
            temp_path = PROJECTS_DIR / project_id / f"_temp_plan{Path(plan_file.name).suffix}"
            temp_path.write_bytes(plan_file.getvalue())

            try:
                plan = parser.parse_file(temp_path)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse du plan : {e}")
                return
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        elif plan_text.strip():
            plan = parser.parse_text(plan_text)
        else:
            st.error("Veuillez fournir un fichier ou saisir un plan.")
            return

        # Distribuer le budget de pages depuis la config persistée
        target_pages = state.config.get("target_pages")
        if target_pages:
            parser.distribute_page_budget(plan, target_pages)

        if state.config.get("objective"):
            plan.objective = state.config["objective"]

        state.plan = plan
        state.current_step = "plan"
        _save_state(state)
        st.success(f"Plan normalisé : {len(plan.sections)} sections détectées.")
        st.rerun()


def _render_generate_plan(state):
    """Génération automatique du plan par l'IA."""
    provider = st.session_state.get("provider")
    if not provider or not provider.is_available():
        st.warning("Configurez d'abord votre clé API dans la page Configuration.")
        return

    objective = st.text_area(
        "Objectif du document",
        value=state.config.get("objective", ""),
        height=100,
        placeholder="Décrivez le document que vous souhaitez produire...",
        key="plan_gen_objective",
    )

    if st.button("Générer le plan", type="primary"):
        if not objective:
            st.error("Veuillez décrire l'objectif du document.")
            return

        # Persister l'objectif dans la config
        state.config["objective"] = objective

        target_pages = state.config.get("target_pages")

        with st.spinner("Génération du plan en cours..."):
            from src.core.orchestrator import Orchestrator
            from src.core.cost_tracker import CostTracker

            project_id = st.session_state.current_project
            project_dir = PROJECTS_DIR / project_id

            orchestrator = Orchestrator(
                provider=provider,
                project_dir=project_dir,
                cost_tracker=st.session_state.get("cost_tracker") or CostTracker(),
                config=state.config,
            )

            try:
                plan = orchestrator.generate_plan_from_objective(
                    objective, target_pages, corpus=state.corpus,
                )
                state.plan = plan
                state.current_step = "plan"
                _save_state(state)
                st.success(f"Plan généré : {len(plan.sections)} sections.")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de la génération : {e}")


def _render_view_plan(state):
    """Visualisation et modification du plan."""
    plan = state.plan
    target_pages = state.config.get("target_pages")

    # En-tête avec métriques
    col1, col2, col3 = st.columns(3)
    col1.metric("Sections", len(plan.sections))
    col2.metric("Pages cibles", target_pages or "Auto")
    objective_text = plan.objective or ""
    if objective_text:
        display_obj = objective_text[:30] + "..." if len(objective_text) > 30 else objective_text
        col3.metric("Objectif", display_obj)

    st.markdown("---")

    # Affichage hiérarchique
    for section in plan.sections:
        indent = "\u3000" * (section.level - 1)
        status_icon = {
            "pending": "",
            "generating": "",
            "generated": "",
            "validated": "",
            "failed": "",
        }.get(section.status, "")

        budget_str = f" ({section.page_budget} p.)" if section.page_budget else ""
        st.markdown(f"{indent}{status_icon} **{section.id}** {section.title}{budget_str}")

        if section.description:
            st.caption(f"{indent}  _{section.description}_")

    # Modification du plan
    st.markdown("---")
    with st.expander("Modifier le plan"):
        plan_text = ""
        for s in plan.sections:
            indent = "  " * (s.level - 1)
            plan_text += f"{indent}{s.id} {s.title}\n"
            if s.description:
                plan_text += f"{indent}  {s.description}\n"

        edited_text = st.text_area("Plan (éditable)", value=plan_text, height=300)

        if st.button("Appliquer les modifications"):
            parser = PlanParser()
            new_plan = parser.parse_text(edited_text)
            if state.config.get("objective"):
                new_plan.objective = state.config["objective"]
            if target_pages:
                parser.distribute_page_budget(new_plan, target_pages)
            state.plan = new_plan
            _save_state(state)
            st.success("Plan mis à jour.")
            st.rerun()

    with st.expander("Remplacer le plan"):
        _render_create_plan(state)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Valider le plan et continuer", type="primary", use_container_width=True):
            state.current_step = "generation"
            _save_state(state)
            st.session_state.current_page = "generation"
            st.rerun()
    with col2:
        if st.button("Retour", use_container_width=True):
            st.session_state.current_page = "acquisition"
            st.rerun()


def _save_state(state):
    """Sauvegarde l'état du projet."""
    project_id = st.session_state.current_project
    if project_id:
        state_path = PROJECTS_DIR / project_id / "state.json"
        save_json(state_path, state.to_dict())
