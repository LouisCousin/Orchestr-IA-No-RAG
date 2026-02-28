"""Page du plan — Client HTTP (Phase 3 Sprint 3).

Aucun import de src.core. Le plan est géré côté API.
"""

import streamlit as st

from src.client.api_client import OrchestrIAClient


def _get_client() -> OrchestrIAClient:
    from src.client.app import get_client
    return get_client()


def render():
    st.title("Plan du document")
    st.info(
        "**Étape 3** — Définissez la structure de votre document. "
        "Vous pouvez saisir un plan manuellement ou le faire générer automatiquement."
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

    plan = state.get("plan")

    if plan and plan.get("sections"):
        _render_existing_plan(client, project_id, plan)
    else:
        _render_plan_input(client, project_id, state)

    # Navigation
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Retour à l'acquisition", use_container_width=True):
            st.session_state.current_page = "acquisition"
            st.rerun()
    with col_next:
        if plan and plan.get("sections"):
            if st.button("Continuer vers la génération →", type="primary", use_container_width=True):
                st.session_state.current_page = "generation"
                st.rerun()


def _render_plan_input(client: OrchestrIAClient, project_id: str, state: dict):
    """Formulaire de saisie du plan."""
    st.subheader("Saisir le plan")

    plan_text = st.text_area(
        "Plan du document (format hiérarchique)",
        height=300,
        placeholder=(
            "1. Introduction\n"
            "   1.1. Contexte\n"
            "   1.2. Objectifs\n"
            "2. Analyse\n"
            "   2.1. État de l'art\n"
            "   2.2. Méthodologie\n"
            "3. Résultats\n"
            "4. Conclusion"
        ),
    )

    if st.button("Valider le plan", type="primary"):
        if not plan_text.strip():
            st.error("Veuillez saisir un plan.")
            return

        try:
            # Envoyer le plan via la config
            client.update_config(project_id, {"plan_text": plan_text})
            st.success("Plan enregistré.")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")


def _render_existing_plan(client: OrchestrIAClient, project_id: str, plan: dict):
    """Affiche le plan existant."""
    st.subheader(plan.get("title", "Plan du document"))

    sections = plan.get("sections", [])
    st.markdown(f"**{len(sections)} section(s)**")

    for section in sections:
        indent = "  " * (section.get("level", 1) - 1)
        status_icon = {
            "pending": "⏳", "generating": "🔄", "generated": "✅", "failed": "❌",
        }.get(section.get("status", "pending"), "❓")

        st.markdown(f"{indent}{status_icon} **{section['id']}** — {section['title']}")

    # Bouton pour relancer l'architecte
    st.markdown("---")
    st.subheader("Architecte Agent")
    st.caption("L'Architecte analyse le plan et détermine les dépendances entre sections.")

    if st.button("Lancer l'Architecte"):
        with st.spinner("Analyse du plan en cours..."):
            try:
                result = client.run_architect(project_id)
                st.success("Architecture générée !")
                architecture = result.get("architecture", {})

                with st.expander("Architecture proposée"):
                    st.json(architecture)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Valider l'architecture", type="primary"):
                        client.validate_architect(project_id, approved=True)
                        st.success("Architecture validée.")
                        st.rerun()
                with col2:
                    if st.button("Rejeter"):
                        client.validate_architect(project_id, approved=False)
                        st.warning("Architecture rejetée.")
                        st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")
