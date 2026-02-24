"""Page d'export du document final."""

import streamlit as st
from pathlib import Path
from datetime import datetime

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir, sanitize_filename
from src.core.export_engine import ExportEngine


PROJECTS_DIR = ROOT_DIR / "projects"
OUTPUT_DIR = ROOT_DIR / "output"


def render():
    st.title("Export du document")
    st.info(
        "**Étape 5/5** — Exportez votre document au format DOCX. Vous pouvez "
        "également exporter les métadonnées (coûts, couverture RAG) au format Excel."
    )
    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif.")
        return

    state = st.session_state.project_state

    if not state.plan or not state.generated_sections:
        st.warning("Aucune section générée. Lancez d'abord la génération.")
        return

    # Bouton retour
    if st.button("← Retour à la génération"):
        st.session_state.current_page = "generation"
        st.rerun()

    _render_docx_export(state)

    # Récapitulatif en bas de page
    st.markdown("---")
    with st.expander("Récapitulatif du projet"):
        _render_recap(state)

    # Bouton retour accueil
    st.markdown("---")
    if st.button("Retour à l'accueil", use_container_width=True):
        st.session_state.current_page = "accueil"
        st.rerun()


def _render_docx_export(state):
    """Export DOCX."""
    plan = state.plan
    total_sections = len(plan.sections)
    generated = len(state.generated_sections)

    col1, col2 = st.columns(2)
    col1.metric("Sections générées", f"{generated}/{total_sections}")

    styling = state.config.get("styling", {})
    col2.metric("Police", styling.get("font_title", "Calibri"))

    if generated < total_sections:
        st.warning(f"{total_sections - generated} section(s) non générée(s). Elles apparaîtront comme '[Section non générée]' dans le document.")

    # Options d'export
    st.markdown("---")
    filename = st.text_input(
        "Nom du fichier",
        value=sanitize_filename(state.name or "document"),
    )

    col_docx, col_meta = st.columns(2)

    with col_docx:
        if st.button("Générer le DOCX", type="primary", use_container_width=True):
            ensure_dir(OUTPUT_DIR)
            output_path = OUTPUT_DIR / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"

            with st.spinner("Génération du document DOCX..."):
                try:
                    engine = ExportEngine(styling=styling)
                    result_path = engine.export_docx(
                        plan=plan,
                        generated_sections=state.generated_sections,
                        output_path=output_path,
                        project_name=state.name,
                    )

                    st.success(f"Document généré : {result_path.name}")

                    with open(result_path, "rb") as f:
                        st.download_button(
                            "Télécharger le DOCX",
                            data=f.read(),
                            file_name=result_path.name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"Erreur lors de la génération : {e}")

    with col_meta:
        if st.button("Exporter les métadonnées (Excel)", use_container_width=True):
            ensure_dir(OUTPUT_DIR)
            xlsx_path = OUTPUT_DIR / f"{filename}_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            try:
                engine = ExportEngine(styling=styling)
                engine.export_metadata_excel(
                    plan=plan,
                    generated_sections=state.generated_sections,
                    cost_report=state.cost_report or {},
                    output_path=xlsx_path,
                    rag_coverage=getattr(state, "rag_coverage", None),
                    deferred_sections=getattr(state, "deferred_sections", None),
                )

                st.success(f"Métadonnées exportées : {xlsx_path.name}")
                with open(xlsx_path, "rb") as f:
                    st.download_button(
                        "Télécharger le fichier Excel",
                        data=f.read(),
                        file_name=xlsx_path.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as e:
                st.error(f"Erreur : {e}")


def _render_recap(state):
    """Récapitulatif du projet."""
    st.markdown(f"**Projet :** {state.name}")
    st.markdown(f"**Créé le :** {state.created_at[:16] if state.created_at else 'N/A'}")

    if state.plan:
        st.markdown(f"**Titre du document :** {state.plan.title}")
        st.markdown(f"**Sections :** {len(state.plan.sections)}")
        st.markdown(f"**Sections générées :** {len(state.generated_sections)}")

    # Métriques de coûts
    cost = state.cost_report
    if cost:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tokens input", f"{cost.get('total_input_tokens', 0):,}")
        col2.metric("Tokens output", f"{cost.get('total_output_tokens', 0):,}")
        col3.metric("Coût réel", f"${cost.get('total_cost_usd', 0):.4f}")
        col4.metric("Coût estimé", f"${cost.get('estimated_cost_usd', 0):.4f}")

    # Détail des sections
    st.markdown("---")
    if state.plan:
        for section in state.plan.sections:
            content = state.generated_sections.get(section.id, "")
            status_icon = "v" if content else "-"
            length = len(content) if content else 0
            st.markdown(f"[{status_icon}] **{section.id}** {section.title} — {length:,} caractères")
