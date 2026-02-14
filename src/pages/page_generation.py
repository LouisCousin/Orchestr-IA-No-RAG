"""Page de génération séquentielle du contenu."""

import streamlit as st
from pathlib import Path

from src.utils.config import ROOT_DIR
from src.utils.file_utils import save_json
from src.utils.logger import ActivityLog
from src.core.orchestrator import Orchestrator
from src.core.corpus_extractor import CorpusExtractor
from src.core.cost_tracker import CostTracker
from src.core.checkpoint_manager import CheckpointManager, CheckpointConfig


PROJECTS_DIR = ROOT_DIR / "projects"


def render():
    st.title("Génération du document")
    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif.")
        return

    state = st.session_state.project_state
    provider = st.session_state.get("provider")

    if not state.plan:
        st.warning("Aucun plan défini. Rendez-vous sur la page Plan.")
        return

    if not provider or not provider.is_available():
        st.warning("Fournisseur IA non configuré. Rendez-vous sur la page Configuration.")
        return

    _render_launch_and_progress(state, provider)

    # Section relecture (visible si des sections sont générées)
    if state.generated_sections:
        st.markdown("---")
        _render_review(state)


def _render_launch_and_progress(state, provider):
    """Interface de lancement et suivi de la génération."""
    plan = state.plan
    config = state.config

    # Métriques d'état
    total_sections = len(plan.sections)
    already_generated = len(state.generated_sections)
    failed = sum(1 for s in plan.sections if s.status == "failed")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sections", total_sections)
    col2.metric("Générées", already_generated)
    col3.metric("Modèle", config.get("model", "gpt-4o"))
    col4.metric("Pages cibles", config.get("target_pages") or "Auto")

    if total_sections > 0:
        st.progress(already_generated / total_sections, text=f"{already_generated}/{total_sections} sections")

    # Estimation des coûts
    tracker = st.session_state.get("cost_tracker") or CostTracker()

    project_id = st.session_state.current_project
    corpus_dir = PROJECTS_DIR / project_id / "corpus"
    avg_corpus_tokens = 2000

    if corpus_dir.exists() and any(corpus_dir.iterdir()):
        extractor = CorpusExtractor()
        corpus = extractor.extract_corpus(corpus_dir)
        state.corpus = corpus
        if corpus.total_chunks > 0:
            avg_corpus_tokens = corpus.total_tokens // max(1, total_sections)

    estimate = tracker.estimate_project_cost(
        section_count=total_sections,
        avg_corpus_tokens=avg_corpus_tokens,
        provider=config.get("default_provider", "openai"),
        model=config.get("model", "gpt-4o"),
        num_passes=config.get("number_of_passes", 1),
    )

    if "error" not in estimate:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Coût estimé", f"${estimate['estimated_cost_usd']:.4f}")
        col2.metric("Tokens input estimés", f"{estimate['estimated_input_tokens']:,}")
        col3.metric("Tokens output estimés", f"{estimate['estimated_output_tokens']:,}")

    # Coûts réels (si génération en cours/terminée)
    cost_report = state.cost_report
    if cost_report and cost_report.get("total_cost_usd", 0) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Tokens input réels", f"{cost_report.get('total_input_tokens', 0):,}")
        col2.metric("Tokens output réels", f"{cost_report.get('total_output_tokens', 0):,}")
        col3.metric("Coût réel", f"${cost_report.get('total_cost_usd', 0):.4f}")

    # Actions
    st.markdown("---")

    if already_generated > 0 and already_generated < total_sections:
        st.info(f"{already_generated}/{total_sections} sections déjà générées. La génération reprendra là où elle s'est arrêtée.")

    if already_generated == total_sections:
        st.success("Toutes les sections ont été générées !")
        if st.button("Passer à l'export", type="primary", use_container_width=True):
            st.session_state.current_page = "export"
            st.rerun()
    else:
        if st.button("Lancer la génération", type="primary", use_container_width=True):
            _run_generation(state, provider, tracker)

    # Journal d'activité
    with st.expander("Journal d'activité"):
        logs = st.session_state.activity_log.get_recent(20)
        if logs:
            for log in reversed(logs):
                icon = {"info": "i", "warning": "!", "error": "x", "success": "v"}.get(log["level"], "i")
                st.text(f"[{icon}] [{log['timestamp'][11:19]}] {log['message']}")
        else:
            st.info("Aucune activité enregistrée.")


def _run_generation(state, provider, tracker):
    """Exécute la génération séquentielle."""
    project_id = st.session_state.current_project
    project_dir = PROJECTS_DIR / project_id

    # Configurer les checkpoints
    cp_config = CheckpointConfig.from_dict(state.config.get("checkpoints", {}))
    checkpoint_mgr = CheckpointManager(config=cp_config)

    activity_log = st.session_state.activity_log

    orchestrator = Orchestrator(
        provider=provider,
        project_dir=project_dir,
        checkpoint_manager=checkpoint_mgr,
        cost_tracker=tracker,
        activity_log=activity_log,
        config=state.config,
    )
    orchestrator.state = state

    # Extraction du corpus si nécessaire
    if not state.corpus:
        corpus_dir = project_dir / "corpus"
        if corpus_dir.exists():
            extractor = CorpusExtractor()
            state.corpus = extractor.extract_corpus(corpus_dir)

    # Barre de progression
    progress_bar = st.progress(0, text="Démarrage de la génération...")
    status_area = st.empty()
    log_area = st.empty()

    plan = state.plan
    sections_to_generate = [s for s in plan.sections if s.status != "generated"]
    total = len(sections_to_generate)

    for i, section in enumerate(sections_to_generate):
        progress_bar.progress((i) / max(total, 1), text=f"Section {section.id}: {section.title}")
        status_area.info(f"Génération de la section **{section.id} {section.title}**...")

        # Construire et exécuter
        corpus_chunks = []
        if state.corpus:
            corpus_chunks = state.corpus.get_chunks_for_section(section.title)

        prompt = orchestrator.prompt_engine.build_section_prompt(
            section=section,
            plan=plan,
            corpus_chunks=corpus_chunks,
            previous_summaries=state.section_summaries,
            target_pages=state.config.get("target_pages"),
        )

        system_prompt = orchestrator.prompt_engine.build_system_prompt()

        try:
            response = provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=state.config.get("model", "gpt-4o"),
                temperature=state.config.get("temperature", 0.7),
                max_tokens=state.config.get("max_tokens", 4096),
            )

            tracker.record(
                section_id=section.id,
                model=state.config.get("model", "gpt-4o"),
                provider=provider.name,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                task_type="generation",
            )

            state.generated_sections[section.id] = response.content
            section.status = "generated"
            section.generated_content = response.content

            # Résumé pour le contexte
            summary = response.content[:200] + "..."
            state.section_summaries.append(f"[{section.id}] {section.title}: {summary}")

            activity_log.success(
                f"Section {section.id} générée ({response.output_tokens} tokens)",
                section=section.id,
            )

        except Exception as e:
            section.status = "failed"
            activity_log.error(f"Erreur section {section.id}: {e}", section=section.id)
            status_area.error(f"Erreur sur la section {section.id}: {e}")

        # Sauvegarder après chaque section
        state.cost_report = tracker.report.to_dict()
        save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())

        # Afficher les logs récents
        recent_logs = activity_log.get_recent(5)
        log_text = "\n".join(f"[{l['level'].upper()}] {l['message']}" for l in recent_logs)
        log_area.code(log_text, language=None)

    progress_bar.progress(1.0, text="Génération terminée !")
    status_area.success(f"Toutes les sections ont été générées. Coût total : ${tracker.report.total_cost_usd:.4f}")

    state.current_step = "review"
    save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())


def _render_review(state):
    """Relecture des sections générées."""
    st.subheader("Relecture des sections")

    plan = state.plan
    for section in plan.sections:
        content = state.generated_sections.get(section.id, "")
        if not content:
            continue

        with st.expander(f"{section.id} {section.title}", expanded=False):
            edited = st.text_area(
                f"Contenu de la section {section.id}",
                value=content,
                height=300,
                key=f"review_{section.id}",
            )

            if edited != content:
                if st.button(f"Sauvegarder les modifications", key=f"save_{section.id}"):
                    state.generated_sections[section.id] = edited
                    section.generated_content = edited
                    project_id = st.session_state.current_project
                    save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
                    st.success(f"Section {section.id} mise à jour.")

    # Passage à l'export
    st.markdown("---")
    if st.button("Passer à l'export", type="primary", use_container_width=True):
        state.current_step = "export"
        project_id = st.session_state.current_project
        save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
        st.session_state.current_page = "export"
        st.rerun()
