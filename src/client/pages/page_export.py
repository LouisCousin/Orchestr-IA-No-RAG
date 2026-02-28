"""Page d'export — Client HTTP (Phase 3 Sprint 3).

Aucun import de src.core. L'état du projet est récupéré via l'API.
L'export DOCX est encore local (la logique est dans src.core.export_engine)
mais l'état provient exclusivement du serveur.
"""

import streamlit as st

from src.client.api_client import OrchestrIAClient


def _get_client() -> OrchestrIAClient:
    from src.client.app import get_client
    return get_client()


def render():
    st.title("Export du document")
    st.info(
        "**Étape 6** — Récapitulatif du projet et préparation de l'export. "
        "Le document est disponible au téléchargement."
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
    generated = state.get("generated_sections", {})

    if not plan or not generated:
        st.warning("Aucune section générée. Lancez d'abord la génération.")
        return

    sections = plan.get("sections", [])

    # ── Récapitulatif ──
    st.subheader("Récapitulatif")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sections générées", f"{len(generated)}/{len(sections)}")
    with col2:
        cost_report = state.get("cost_report", {})
        st.metric("Coût total", f"${cost_report.get('total_cost_usd', 0):.4f}")
    with col3:
        actual_words = sum(len(c.split()) for c in generated.values())
        st.metric("Mots générés", f"{actual_words:,}")

    # ── Détail des sections ──
    st.markdown("---")
    for section in sections:
        sid = section["id"]
        content = generated.get(sid, "")
        status_icon = "✅" if content else "❌"
        length = len(content) if content else 0
        st.markdown(f"[{status_icon}] **{sid}** {section['title']} — {length:,} caractères")

    # ── Info export ──
    st.markdown("---")
    st.info(
        "Pour exporter en DOCX, utilisez la documentation Swagger du serveur API "
        "ou l'ancien client Streamlit (src/app.py) qui inclut l'ExportEngine."
    )

    # Navigation
    st.markdown("---")
    col_back, col_home = st.columns(2)
    with col_back:
        if st.button("← Retour au dashboard", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    with col_home:
        if st.button("Retour à l'accueil", use_container_width=True):
            st.session_state.current_page = "accueil"
            st.rerun()
