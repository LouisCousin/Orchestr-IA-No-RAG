"""Dashboard de métriques et graphiques — Phase 3.

Offre une vue centralisée de l'avancement, des coûts, de la qualité
et des performances du pipeline.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir


def render():
    """Affiche le dashboard de métriques."""
    st.title("Dashboard")
    st.markdown("Vue centralisée de l'avancement, des coûts et de la qualité du pipeline.")

    state = st.session_state.get("project_state")
    if not state:
        st.info("Aucun projet actif. Configurez et lancez un projet pour voir les métriques.")
        return

    # ── Section A : Progression globale ──
    st.header("Progression globale")
    col1, col2, col3 = st.columns(3)

    total_sections = len(state.plan.sections) if state.plan else 0
    generated = len(state.generated_sections)

    with col1:
        st.metric("Sections générées", f"{generated}/{total_sections}")
    with col2:
        target_pages = state.config.get("target_pages", 0)
        actual_words = sum(len(c.split()) for c in state.generated_sections.values())
        actual_pages = round(actual_words / 400, 1) if actual_words else 0
        st.metric("Pages estimées", f"{actual_pages}" + (f" / {target_pages}" if target_pages else ""))
    with col3:
        st.metric("Étape courante", state.current_step.capitalize())

    if total_sections > 0:
        progress = generated / total_sections
        st.progress(progress, text=f"{generated}/{total_sections} sections ({progress:.0%})")

    # Statut par section
    if state.plan:
        with st.expander("Statut détaillé par section"):
            section_data = []
            for s in state.plan.sections:
                status_icon = {
                    "pending": "⏳", "generating": "🔄", "generated": "✅",
                    "deferred": "⏸️", "failed": "❌",
                }.get(s.status, "❓")
                section_data.append({
                    "ID": s.id,
                    "Titre": s.title,
                    "Statut": f"{status_icon} {s.status}",
                    "Longueur": len(state.generated_sections.get(s.id, "")),
                })
            st.dataframe(pd.DataFrame(section_data), use_container_width=True)

    st.divider()

    # ── Section B : Consommation et coûts ──
    st.header("Consommation et coûts")
    cost_report = state.cost_report or {}
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
        with st.expander("Détails des coûts par section"):
            # Bar chart tokens by section
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

            # Cost by provider
            by_provider = {}
            for entry in entries:
                provider = entry.get("provider", "unknown")
                cost = entry.get("cost_usd", 0)
                by_provider[provider] = by_provider.get(provider, 0) + cost

            if by_provider:
                st.markdown("**Coût par fournisseur :**")
                for provider, cost in by_provider.items():
                    st.markdown(f"- {provider}: ${cost:.4f}")

    st.divider()

    # ── Section C : Qualité et fiabilité ──
    st.header("Qualité et fiabilité")

    quality_reports = getattr(state, "quality_reports", {})
    factcheck_reports = getattr(state, "factcheck_reports", {})

    if quality_reports:
        with st.expander("Scores de qualité par section", expanded=True):
            quality_data = []
            for sid, report in quality_reports.items():
                row = {"Section": sid, "Score global": report.get("global_score", 0)}
                for criterion in report.get("criteria", []):
                    row[criterion.get("name", criterion.get("id", ""))] = criterion.get("score", 0)
                row["{{NEEDS_SOURCE}}"] = report.get("needs_source_count", 0)
                quality_data.append(row)

            if quality_data:
                df = pd.DataFrame(quality_data)
                st.dataframe(df, use_container_width=True)

                # Heatmap-style visualization using bar chart
                if len(quality_data) > 1:
                    scores_only = df.set_index("Section").drop(columns=["{{NEEDS_SOURCE}}"], errors="ignore")
                    st.bar_chart(scores_only)
    else:
        st.info("Les rapports de qualité seront disponibles après la génération.")

    if factcheck_reports:
        with st.expander("Fiabilité factuelle par section"):
            fc_data = []
            for sid, report in factcheck_reports.items():
                fc_data.append({
                    "Section": sid,
                    "Score (%)": report.get("reliability_score", 0),
                    "Affirmations": report.get("total_claims", 0),
                    "Corroborées": report.get("status_counts", {}).get("CORROBORÉE", 0),
                    "Plausibles": report.get("status_counts", {}).get("PLAUSIBLE", 0),
                    "Non fondées": report.get("status_counts", {}).get("NON FONDÉE", 0),
                    "Contredites": report.get("status_counts", {}).get("CONTREDITE", 0),
                })
            st.dataframe(pd.DataFrame(fc_data), use_container_width=True)

    # NEEDS_SOURCE markers
    from src.core.export_engine import scan_all_sections_for_markers
    markers = scan_all_sections_for_markers(state.generated_sections)
    total_markers = sum(len(m) for m in markers.values())
    if total_markers > 0:
        st.warning(f"**{total_markers}** marqueur(s) {{{{NEEDS_SOURCE}}}} résiduel(s) dans {len(markers)} section(s).")
    elif state.generated_sections:
        st.success("Aucun marqueur {{NEEDS_SOURCE}} résiduel.")

    # RAG coverage
    if state.rag_coverage:
        with st.expander("Couverture RAG par section"):
            rag_data = []
            for sid, cov in state.rag_coverage.items():
                rag_data.append({
                    "Section": sid,
                    "Niveau": cov.get("level", ""),
                    "Score moyen": round(cov.get("avg_score", 0), 3),
                    "Blocs pertinents": cov.get("num_relevant_blocks", 0),
                })
            st.dataframe(pd.DataFrame(rag_data), use_container_width=True)

    st.divider()

    # ── Section D : Performance ──
    st.header("Performance")
    cost_tracker = st.session_state.get("cost_tracker")
    if cost_tracker and hasattr(cost_tracker, "report"):
        report = cost_tracker.report
        entries = report.entries if hasattr(report, "entries") else []
        if entries:
            total_calls = len(entries)
            st.metric("Appels API", total_calls)

    st.divider()

    # ── Section E : Activité et audit ──
    st.header("Activité et audit")
    activity_log = st.session_state.get("activity_log")
    if activity_log:
        recent = activity_log.get_recent(20)
        if recent:
            with st.expander("Journal d'activité récent", expanded=False):
                for entry in recent:
                    level = entry.get("level", "info")
                    msg = entry.get("message", "")
                    icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "•")
                    st.markdown(f"{icon} {msg}")

    # Feedback statistics
    feedback_history = getattr(state, "feedback_history", [])
    if feedback_history:
        with st.expander("Feedback loop"):
            total_fb = len(feedback_history)
            accepted = sum(1 for f in feedback_history if f.get("decision") == "accepted")
            rate = accepted / total_fb if total_fb > 0 else 0
            st.metric("Taux d'acceptation", f"{rate:.0%}")

    st.divider()

    # ── Export des métriques ──
    if st.button("Exporter les métriques (Excel)"):
        _export_dashboard_excel(state)


def _export_dashboard_excel(state):
    """Exporte les métriques du dashboard en Excel."""
    project_name = state.name or "projet"
    output_dir = ensure_dir(ROOT_DIR / "output")
    filepath = output_dir / f"dashboard_{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    try:
        with pd.ExcelWriter(str(filepath), engine="openpyxl") as writer:
            # Sections
            if state.plan:
                section_data = [{
                    "ID": s.id,
                    "Titre": s.title,
                    "Statut": s.status,
                    "Longueur": len(state.generated_sections.get(s.id, "")),
                } for s in state.plan.sections]
                pd.DataFrame(section_data).to_excel(writer, sheet_name="Sections", index=False)

            # Costs
            entries = (state.cost_report or {}).get("entries", [])
            if entries:
                pd.DataFrame(entries).to_excel(writer, sheet_name="Coûts", index=False)

            # Quality
            quality_reports = getattr(state, "quality_reports", {})
            if quality_reports:
                q_data = [{"Section": sid, **r} for sid, r in quality_reports.items()]
                pd.DataFrame(q_data).to_excel(writer, sheet_name="Qualité", index=False)

        st.success(f"Métriques exportées : {filepath}")
    except Exception as e:
        st.error(f"Erreur export : {e}")
