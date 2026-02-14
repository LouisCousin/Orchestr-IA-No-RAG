"""Page d'acquisition du corpus documentaire."""

import streamlit as st
from pathlib import Path

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir
from src.utils.token_counter import count_tokens
from src.core.corpus_acquirer import CorpusAcquirer, AcquisitionReport
from src.core.text_extractor import extract
from src.core.corpus_deduplicator import CorpusDeduplicator
from src.core.cost_tracker import CostTracker


PROJECTS_DIR = ROOT_DIR / "projects"


def render():
    st.title("Acquisition du corpus")
    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif. Créez ou ouvrez un projet depuis la page Accueil.")
        return

    project_id = st.session_state.current_project
    corpus_dir = PROJECTS_DIR / project_id / "corpus"
    ensure_dir(corpus_dir)

    tab_upload, tab_urls, tab_recap = st.tabs([
        "Fichiers locaux", "URLs distantes", "Récapitulatif"
    ])

    with tab_upload:
        _render_file_upload(corpus_dir)

    with tab_urls:
        _render_url_acquisition(corpus_dir)

    with tab_recap:
        _render_corpus_recap(corpus_dir)


def _render_file_upload(corpus_dir: Path):
    """Upload de fichiers locaux."""
    st.subheader("Téléversement de fichiers")

    uploaded_files = st.file_uploader(
        "Sélectionnez vos fichiers sources",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md", "html"],
        accept_multiple_files=True,
        help="Formats supportés : PDF, DOCX, Excel, CSV, TXT, Markdown, HTML",
    )

    if uploaded_files and st.button("Ajouter au corpus", type="primary"):
        acquirer = CorpusAcquirer(corpus_dir)
        report = AcquisitionReport()

        progress = st.progress(0, text="Acquisition en cours...")
        for i, uploaded in enumerate(uploaded_files):
            # Sauvegarder le fichier temporairement
            temp_path = corpus_dir / f"_temp_{uploaded.name}"
            temp_path.write_bytes(uploaded.getvalue())

            acquirer.acquire_local_files([temp_path], report)

            # Supprimer le fichier temporaire
            if temp_path.exists():
                temp_path.unlink()

            progress.progress((i + 1) / len(uploaded_files), text=f"Traitement de {uploaded.name}...")

        progress.empty()
        _display_report(report)
        st.rerun()


def _render_url_acquisition(corpus_dir: Path):
    """Acquisition depuis URLs."""
    st.subheader("Téléchargement depuis URLs")

    urls_text = st.text_area(
        "URLs (une par ligne)",
        height=150,
        placeholder="https://example.com/document.pdf\nhttps://example.com/page-web",
    )

    col1, col2 = st.columns(2)
    with col1:
        url_file = st.file_uploader(
            "Ou importer depuis un fichier Excel/CSV",
            type=["xlsx", "csv"],
            key="url_file",
        )
    with col2:
        slow_mode = st.checkbox("Mode sites lents (timeouts étendus)")

    if st.button("Lancer l'acquisition", type="primary"):
        config = st.session_state.config
        acq_config = config.get("corpus_acquisition", {})

        if slow_mode:
            conn_timeout = acq_config.get("slow_mode_connection_timeout", 30)
            read_timeout = acq_config.get("slow_mode_read_timeout", 120)
        else:
            conn_timeout = acq_config.get("connection_timeout", 15)
            read_timeout = acq_config.get("read_timeout", 60)

        acquirer = CorpusAcquirer(
            corpus_dir=corpus_dir,
            connection_timeout=conn_timeout,
            read_timeout=read_timeout,
            throttle_delay=acq_config.get("throttle_delay", 1.0),
            user_agent=acq_config.get("user_agent", "Mozilla/5.0"),
        )

        report = AcquisitionReport()

        # URLs depuis le fichier
        if url_file:
            temp_path = corpus_dir / f"_temp_{url_file.name}"
            temp_path.write_bytes(url_file.getvalue())
            acquirer.acquire_urls_from_file(temp_path, report)
            temp_path.unlink()

        # URLs saisies
        if urls_text.strip():
            urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
            if urls:
                with st.spinner(f"Téléchargement de {len(urls)} URL(s)..."):
                    acquirer.acquire_urls(urls, report)

        acquirer.close()
        _display_report(report)
        st.rerun()


def _render_corpus_recap(corpus_dir: Path):
    """Récapitulatif du corpus acquis avec pré-analyse."""
    st.subheader("Corpus acquis")

    files = sorted(f for f in corpus_dir.iterdir() if f.is_file() and not f.name.startswith(".") and f.suffix != ".json")

    if not files:
        st.info("Aucun document dans le corpus. Utilisez les onglets ci-dessus pour ajouter des documents.")
        return

    # Analyse du corpus
    documents_info = []
    deduplicator = CorpusDeduplicator(corpus_dir)
    extractions = []

    with st.spinner("Analyse du corpus..."):
        for f in files:
            result = extract(f)
            extractions.append(result)

            tokens = count_tokens(result.text) if result.text else 0
            dedup_entry = deduplicator.check_duplicate(result)

            documents_info.append({
                "Fichier": f.name,
                "Pages": result.page_count,
                "Tokens": tokens,
                "Statut extraction": result.status,
                "Méthode": result.extraction_method,
                "Taille (Ko)": round(result.source_size_bytes / 1024, 1),
                "Doublon": dedup_entry.status,
            })

            if dedup_entry.status == "unique":
                deduplicator.register(dedup_entry)

    # Tableau récapitulatif
    import pandas as pd
    df = pd.DataFrame(documents_info)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Statistiques agrégées
    total_tokens = sum(d["Tokens"] for d in documents_info)
    total_files = len(documents_info)
    duplicates = sum(1 for d in documents_info if d["Doublon"] != "unique")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", total_files)
    col2.metric("Tokens totaux", f"{total_tokens:,}")
    col3.metric("Doublons détectés", duplicates)

    # Estimation de coûts
    config = st.session_state.project_state.config
    model = config.get("model", "gpt-4o")
    provider = config.get("default_provider", "openai")
    tracker = CostTracker()

    docs_for_estimate = [{"tokens": d["Tokens"]} for d in documents_info]
    estimate = tracker.estimate_corpus_cost(docs_for_estimate, provider, model)
    if "error" not in estimate:
        col4.metric("Coût input estimé", f"${estimate['estimated_input_cost_usd']:.4f}")

        if estimate.get("documents_exceeding_context", 0) > 0:
            st.warning(
                f"⚠️ {estimate['documents_exceeding_context']} document(s) dépasse(nt) "
                f"la fenêtre de contexte du modèle {model} ({estimate['context_window']:,} tokens)."
            )

    # Déduplication
    if duplicates > 0:
        st.markdown("---")
        st.subheader("Gestion des doublons")
        for doc in documents_info:
            if doc["Doublon"] != "unique":
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"**{doc['Fichier']}** — {doc['Doublon']}")
                with col_b:
                    if st.button("Supprimer", key=f"del_{doc['Fichier']}"):
                        deduplicator.remove_duplicate(doc["Fichier"])
                        st.rerun()

    # Bouton pour passer à l'étape suivante
    st.markdown("---")
    if st.button("Continuer vers le plan →", type="primary", use_container_width=True):
        st.session_state.current_page = "plan"
        st.rerun()


def _display_report(report: AcquisitionReport):
    """Affiche le rapport d'acquisition."""
    if report.successful > 0:
        st.success(f"{report.successful} document(s) acquis avec succès.")
    if report.failed > 0:
        st.warning(f"{report.failed} document(s) en échec.")

    for status in report.statuses:
        if status.status == "SUCCESS":
            st.markdown(f"✅ **{status.source}** → {status.message}")
        else:
            st.markdown(f"❌ **{status.source}** — {status.message}")
