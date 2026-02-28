"""Page d'acquisition du corpus — Client HTTP (Phase 3 Sprint 3).

Aucun import de src.core. Upload et acquisition via l'API REST.
"""

import streamlit as st

from src.client.api_client import OrchestrIAClient


def _get_client() -> OrchestrIAClient:
    from src.client.app import get_client
    return get_client()


def render():
    st.title("Acquisition du corpus")
    st.info(
        "**Étape 2** — Constituez votre corpus de sources. Uploadez des fichiers "
        "ou saisissez des URLs. Ces documents serviront de base factuelle."
    )
    st.markdown("---")

    project_id = st.session_state.get("current_project")
    if not project_id:
        st.warning("Aucun projet actif.")
        return

    client = _get_client()

    tab_files, tab_urls = st.tabs(["Fichiers", "URLs"])

    with tab_files:
        _render_file_upload(client, project_id)

    with tab_urls:
        _render_url_acquisition(client, project_id)

    # Navigation
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Retour à la configuration", use_container_width=True):
            st.session_state.current_page = "configuration"
            st.rerun()
    with col_next:
        if st.button("Continuer vers le plan →", type="primary", use_container_width=True):
            st.session_state.current_page = "plan"
            st.rerun()


def _render_file_upload(client: OrchestrIAClient, project_id: str):
    """Upload de fichiers locaux via l'API."""
    st.subheader("Fichiers locaux")

    uploaded_files = st.file_uploader(
        "Sélectionnez vos fichiers sources",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md", "html"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ajouter au corpus", type="primary"):
        files = [(f.name, f.getvalue()) for f in uploaded_files]

        with st.spinner(f"Upload de {len(files)} fichier(s)..."):
            try:
                result = client.upload_corpus_files(project_id, files)
                if result.get("successful", 0) > 0:
                    st.success(f"{result['successful']} document(s) acquis avec succès.")
                if result.get("failed", 0) > 0:
                    st.warning(f"{result['failed']} document(s) en échec.")
                for detail in result.get("details", []):
                    st.caption(f"{detail.get('source', '')} — {detail.get('message', '')}")
            except Exception as e:
                st.error(f"Erreur : {e}")


def _render_url_acquisition(client: OrchestrIAClient, project_id: str):
    """Acquisition depuis URLs via l'API."""
    st.subheader("URLs distantes")

    urls_text = st.text_area(
        "URLs (une par ligne)",
        height=120,
        placeholder="https://example.com/document.pdf\nhttps://example.com/page-web",
    )

    slow_mode = st.checkbox("Mode sites lents (timeouts étendus)")

    if st.button("Lancer l'acquisition", type="primary"):
        if not urls_text.strip():
            st.error("Veuillez saisir au moins une URL.")
            return

        urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]

        with st.spinner(f"Acquisition de {len(urls)} URL(s)..."):
            try:
                result = client.acquire_corpus_urls(project_id, urls, slow_mode=slow_mode)
                if result.get("successful", 0) > 0:
                    st.success(f"{result['successful']} document(s) acquis avec succès.")
                if result.get("failed", 0) > 0:
                    st.warning(f"{result['failed']} document(s) en échec.")
            except Exception as e:
                st.error(f"Erreur : {e}")
