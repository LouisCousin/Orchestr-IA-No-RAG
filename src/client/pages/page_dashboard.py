"""Dashboard de monitoring temps réel — Client HTTP + WebSocket (Phase 3 Sprint 3).

Aucun import de src.core.
Les données sont récupérées via l'API REST et les événements
temps réel via le WebSocket du projet.
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
    st.title("Dashboard")
    st.markdown("Vue centralisée de l'avancement, des coûts et de la qualité du pipeline.")

    project_id = st.session_state.get("current_project")
    if not project_id:
        st.info("Aucun projet actif.")
        return

    client = _get_client()

    try:
        state = client.get_project_state(project_id)
    except Exception as e:
        st.error(f"Erreur : {e}")
        return

    # ── WebSocket info ──
    if state.get("is_generating"):
        ws_url = client.get_ws_url(project_id)
        st.info(
            f"Génération en cours. Connectez un client WebSocket sur `{ws_url}` "
            "pour le suivi temps réel."
        )

    # ── Section A : Progression globale ──
    st.header("Progression globale")
    col1, col2, col3 = st.columns(3)

    plan = state.get("plan", {})
    sections = plan.get("sections", []) if plan else []
    total_sections = len(sections)
    generated = len(state.get("generated_sections", {}))

    with col1:
        st.metric("Sections générées", f"{generated}/{total_sections}")
    with col2:
        target_pages = state.get("config", {}).get("target_pages", 0)
        actual_words = sum(len(c.split()) for c in state.get("generated_sections", {}).values())
        actual_pages = round(actual_words / 400, 1) if actual_words else 0
        st.metric("Pages estimées", f"{actual_pages}" + (f" / {target_pages}" if target_pages else ""))
    with col3:
        st.metric("Étape courante", state.get("current_step", "init").capitalize())

    if total_sections > 0:
        progress = generated / total_sections
        st.progress(progress, text=f"{generated}/{total_sections} sections ({progress:.0%})")

    # Statut par section
    if sections:
        with st.expander("Statut détaillé par section"):
            import pandas as pd
            section_data = []
            for s in sections:
                status_icon = {
                    "pending": "⏳", "generating": "🔄", "generated": "✅",
                    "deferred": "⏸️", "failed": "❌",
                }.get(s.get("status", "pending"), "❓")
                gen_sections = state.get("generated_sections", {})
                section_data.append({
                    "ID": s["id"],
                    "Titre": s["title"],
                    "Statut": f"{status_icon} {s.get('status', 'pending')}",
                    "Longueur": len(gen_sections.get(s["id"], "")),
                })
            st.dataframe(pd.DataFrame(section_data))

    st.divider()

    # ── Section B : Consommation et coûts ──
    st.header("Consommation et coûts")
    cost_report = state.get("cost_report", {})
    entries = cost_report.get("entries", [])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tokens input", f"{cost_report.get('total_input_tokens', 0):,}")
    with col2:
        st.metric("Tokens output", f"{cost_report.get('total_output_tokens', 0):,}")
    with col3:
        total_cost = cost_report.get("total_cost_usd", 0)
        estimated = cost_report.get("estimated_cost_usd", 0)
        st.metric(
            "Coût total (USD)",
            f"${total_cost:.4f}",
            delta=f"Estimé: ${estimated:.4f}" if estimated else None,
        )

    if entries:
        import pandas as pd
        with st.expander("Détails des coûts par section"):
            chart_data = {}
            for entry in entries:
                sid = entry.get("section_id", "?")
                if sid not in chart_data:
                    chart_data[sid] = {"input": 0, "output": 0}
                chart_data[sid]["input"] += entry.get("input_tokens", 0)
                chart_data[sid]["output"] += entry.get("output_tokens", 0)

            if chart_data:
                df = pd.DataFrame.from_dict(chart_data, orient="index")
                st.bar_chart(df)

    st.divider()

    # ── Section C : Qualité et fiabilité ──
    st.header("Qualité et fiabilité")
    quality_reports = state.get("quality_reports", {})
    factcheck_reports = state.get("factcheck_reports", {})

    if quality_reports:
        import pandas as pd
        with st.expander("Scores de qualité par section", expanded=True):
            quality_data = []
            for sid, report in quality_reports.items():
                row = {"Section": sid, "Score global": report.get("global_score", 0)}
                for criterion in report.get("criteria", []):
                    row[criterion.get("name", criterion.get("id", ""))] = criterion.get("score", 0)
                quality_data.append(row)
            if quality_data:
                st.dataframe(pd.DataFrame(quality_data))
    else:
        st.info("Les rapports de qualité seront disponibles après la génération.")

    if factcheck_reports:
        import pandas as pd
        with st.expander("Fiabilité factuelle par section"):
            fc_data = []
            for sid, report in factcheck_reports.items():
                fc_data.append({
                    "Section": sid,
                    "Score (%)": report.get("reliability_score", 0),
                    "Affirmations": report.get("total_claims", 0),
                })
            st.dataframe(pd.DataFrame(fc_data))

    st.divider()

    # ── Rechargement ──
    if st.button("Rafraîchir les données", use_container_width=True):
        st.rerun()
