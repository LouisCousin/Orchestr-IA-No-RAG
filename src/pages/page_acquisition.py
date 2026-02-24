"""Page d'acquisition du corpus documentaire."""

import streamlit as st
from pathlib import Path

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir, save_json
from src.utils.token_counter import count_tokens
from src.core.corpus_acquirer import CorpusAcquirer, AcquisitionReport
from src.core.text_extractor import extract
from src.core.corpus_deduplicator import CorpusDeduplicator
from src.core.cost_tracker import CostTracker
from src.utils.content_validator import is_antibot_page


PROJECTS_DIR = ROOT_DIR / "projects"


def render():
    st.title("Acquisition du corpus")
    st.info(
        "**Étape 2/5** — Constituez votre corpus de sources. Uploadez des fichiers "
        "(PDF, DOCX, etc.) ou saisissez des URLs. Ces documents serviront de base "
        "factuelle pour la génération (RAG). Plus le corpus est riche et pertinent, "
        "meilleur sera le document produit."
    )
    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif. Créez ou ouvrez un projet depuis la page Accueil.")
        return

    project_id = st.session_state.current_project
    corpus_dir = PROJECTS_DIR / project_id / "corpus"
    ensure_dir(corpus_dir)

    # Sources d'acquisition (côte à côte)
    col_upload, col_urls = st.columns(2)

    with col_upload:
        _render_file_upload(corpus_dir)

    with col_urls:
        _render_url_acquisition(corpus_dir)

    # Récapitulatif du corpus
    st.markdown("---")
    _render_corpus_recap(corpus_dir)

    # Inspection RAG
    project_dir = PROJECTS_DIR / project_id
    _render_rag_inspection(project_dir)


def _render_file_upload(corpus_dir: Path):
    """Upload de fichiers locaux."""
    st.subheader("Fichiers locaux")

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
    st.subheader("URLs distantes")

    urls_text = st.text_area(
        "URLs (une par ligne)",
        height=120,
        placeholder="https://example.com/document.pdf\nhttps://example.com/page-web",
    )

    url_file = st.file_uploader(
        "Ou importer depuis un fichier Excel/CSV",
        type=["xlsx", "csv"],
        key="url_file",
    )

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
            try:
                acquirer.acquire_urls_from_file(temp_path, report)
            finally:
                temp_path.unlink(missing_ok=True)

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
        st.info("Aucun document dans le corpus. Ajoutez des fichiers ou URLs ci-dessus.")
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

            # Évaluation de la qualité du contenu
            quality = "OK"
            if result.text:
                if is_antibot_page(result.text):
                    quality = "Suspect"
                elif len(result.text.strip()) < 200:
                    quality = "Suspect"
            elif not result.text:
                quality = "Vide"

            documents_info.append({
                "Fichier": f.name,
                "Pages": result.page_count,
                "Tokens": tokens,
                "Qualité": quality,
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
                f"{estimate['documents_exceeding_context']} document(s) dépasse(nt) "
                f"la fenêtre de contexte du modèle {model} ({estimate['context_window']:,} tokens)."
            )

    # Sources suspectes (anti-bot, contenu vide)
    suspects = [d for d in documents_info if d["Qualité"] == "Suspect"]
    if suspects:
        st.markdown("---")
        st.subheader("Sources suspectes")
        st.warning(
            f"{len(suspects)} source(s) détectée(s) comme suspecte(s) "
            "(contenu anti-bot possible ou texte trop court). "
            "Vérifiez le contenu ou supprimez les fichiers concernés."
        )
        for doc in suspects:
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{doc['Fichier']}** — Contenu suspect (anti-bot ou texte < 200 caractères)")
            with col_b:
                if st.button("Supprimer", key=f"del_suspect_{doc['Fichier']}"):
                    suspect_path = corpus_dir / doc["Fichier"]
                    if suspect_path.exists():
                        suspect_path.unlink()
                    st.rerun()

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

    # Navigation inter-étapes
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Retour à la configuration", use_container_width=True):
            st.session_state.current_page = "configuration"
            st.rerun()
    with col_next:
        if st.button("Continuer vers le plan →", type="primary", use_container_width=True):
            state = st.session_state.project_state
            if state:
                state.current_step = "plan"
                project_id = st.session_state.current_project
                if project_id:
                    save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
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
            st.markdown(f"**{status.source}** — {status.message}")
        else:
            st.markdown(f"**{status.source}** — {status.message}")


def _render_rag_inspection(project_dir: Path):
    """Section d'inspection du corpus RAG (chunks indexés)."""
    chromadb_dir = project_dir / "chromadb"
    metadata_db_path = project_dir / "metadata.db"

    has_chromadb = chromadb_dir.exists() and any(chromadb_dir.iterdir()) if chromadb_dir.exists() else False
    has_metadata = metadata_db_path.exists()

    if not has_chromadb and not has_metadata:
        return

    with st.expander("Inspection du corpus RAG"):
        # Informations de stockage
        st.markdown("**Emplacements des données**")
        st.code(f"ChromaDB : {chromadb_dir}\nMetadata SQLite : {metadata_db_path}")

        col1, col2 = st.columns(2)

        # Nombre de chunks ChromaDB
        chroma_count = 0
        if has_chromadb:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(chromadb_dir))
                collection = client.get_or_create_collection("orchestria_corpus")
                chroma_count = collection.count()
            except Exception:
                pass
        col1.metric("Chunks ChromaDB", chroma_count)

        # Nombre de chunks SQLite
        sqlite_count = 0
        if has_metadata:
            try:
                from src.core.metadata_store import MetadataStore
                store = MetadataStore(str(project_dir))
                sqlite_count = store.count_chunks()
                store.close()
            except Exception:
                pass
        col2.metric("Chunks SQLite", sqlite_count)

        # Liste des chunks par document
        if has_metadata and sqlite_count > 0:
            st.markdown("**Chunks par document**")
            try:
                import pandas as pd
                from src.core.metadata_store import MetadataStore
                store = MetadataStore(str(project_dir))
                all_chunks = store.get_all_chunks()
                store.close()

                if all_chunks:
                    chunk_data = []
                    for c in all_chunks[:100]:  # Limiter à 100 pour la performance
                        chunk_data.append({
                            "Document": c.get("doc_id", ""),
                            "Chunk ID": c.get("chunk_id", ""),
                            "Texte (extrait)": (c.get("text", "")[:200] + "...") if len(c.get("text", "")) > 200 else c.get("text", ""),
                            "Page": c.get("page_number", ""),
                            "Tokens": c.get("token_count", ""),
                        })
                    df = pd.DataFrame(chunk_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    if len(all_chunks) > 100:
                        st.caption(f"Affichage limité aux 100 premiers chunks sur {len(all_chunks)} au total.")
            except Exception as e:
                st.warning(f"Impossible de charger les chunks : {e}")

        # Recherche test dans le RAG
        if has_chromadb and chroma_count > 0:
            st.markdown("**Recherche test**")
            test_query = st.text_input("Requête de test", placeholder="Saisissez un terme pour tester la recherche RAG...")
            if test_query and st.button("Tester la recherche"):
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=str(chromadb_dir))
                    collection = client.get_or_create_collection("orchestria_corpus")
                    results = collection.query(
                        query_texts=[test_query],
                        n_results=min(5, chroma_count),
                        include=["documents", "metadatas", "distances"],
                    )
                    if results and results["documents"] and results["documents"][0]:
                        for i, doc in enumerate(results["documents"][0]):
                            distance = results["distances"][0][i] if results["distances"] else 0
                            similarity = 1.0 - distance
                            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                            source = metadata.get("source_file", metadata.get("doc_id", "inconnu"))
                            st.markdown(f"**Résultat {i+1}** (score: {similarity:.3f}) — {source}")
                            st.text(doc[:300] + ("..." if len(doc) > 300 else ""))
                    else:
                        st.info("Aucun résultat trouvé.")
                except Exception as e:
                    st.error(f"Erreur de recherche : {e}")
