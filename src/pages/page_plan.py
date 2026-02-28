"""Page de gestion du plan du document."""

import streamlit as st
from pathlib import Path

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir, save_json
from src.core.plan_parser import PlanParser, NormalizedPlan


PROJECTS_DIR = ROOT_DIR / "projects"


def render():
    st.title("Plan du document")
    st.info(
        "**Étape 3/5** — Définissez la structure de votre document. Importez un plan "
        "existant ou laissez l'IA en générer un à partir de votre objectif. Vous "
        "pourrez ajuster le plan manuellement avant de lancer la génération."
    )
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

            # Phase 2.5 : indexer le corpus avant la génération du plan
            # pour activer le plan-corpus linker (Bug #2 fix)
            if state.corpus:
                orchestrator._init_rag()
                if orchestrator.rag_engine:
                    if orchestrator.rag_engine.indexed_count == 0:
                        with st.spinner("Indexation du corpus..."):
                            orchestrator.index_corpus_rag()

            # Stocker l'orchestrateur en session pour l'affichage
            # du plan_context dans _render_view_plan() (Bug #1 fix)
            st.session_state["orchestrator"] = orchestrator

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

    # Phase 2.5 : résumé du corpus si disponible
    plan_context = None
    orchestrator = st.session_state.get("orchestrator")
    if orchestrator and hasattr(orchestrator, "_last_plan_context"):
        plan_context = orchestrator._last_plan_context

    if plan_context and plan_context.corpus_summary:
        summary = plan_context.corpus_summary
        with st.expander(
            f"📚 Corpus analysé : {summary.get('total_documents', 0)} documents, "
            f"{summary.get('total_tokens', 0):,} tokens"
        ):
            if plan_context.themes:
                st.markdown("**Thèmes identifiés :**")
                for theme in plan_context.themes:
                    cov = plan_context.coverage.get(theme, {})
                    avg = cov.get("avg_score", 0)
                    nb = cov.get("nb_chunks", 0)
                    if avg >= 0.5:
                        icon = "🟢"
                    elif avg >= 0.3:
                        icon = "🟡"
                    else:
                        icon = "🔴"
                    st.markdown(f"  {icon} {theme} — score: {avg:.2f}, {nb} blocs")

    st.markdown("---")

    # Affichage hiérarchique
    for section in plan.sections:
        indent = "\u3000" * (section.level - 1)
        status_icon = {
            "pending": "⬜",
            "generating": "🔄",
            "generated": "✅",
            "validated": "✔️",
            "failed": "❌",
        }.get(section.status, "⬜")

        budget_str = f" ({section.page_budget} p.)" if section.page_budget else ""

        # Phase 2.5 : indicateur de couverture corpus
        coverage_indicator = ""
        if plan_context and plan_context.coverage:
            best_score = 0
            # B11: guard against None values on section.title and theme
            section_title_lower = (section.title or "").lower()
            for theme, cov in plan_context.coverage.items():
                theme_lower = (theme or "").lower()
                if (theme_lower in section_title_lower
                        or section_title_lower in theme_lower):
                    best_score = max(best_score, cov.get("avg_score", 0))
            if best_score > 0:
                if best_score >= 0.5:
                    coverage_indicator = " 🟢"
                elif best_score >= 0.3:
                    coverage_indicator = " 🟡"
                else:
                    coverage_indicator = " 🔴"

        st.markdown(
            f"{indent}{status_icon} **{section.id}** {section.title}"
            f"{budget_str}{coverage_indicator}"
        )

        if section.description:
            st.caption(f"{indent}  _{section.description}_")

    # Phase 2.5 : légende des indicateurs de couverture
    if plan_context and plan_context.coverage:
        st.markdown("---")
        st.caption(
            "🟢 Couverture forte (≥0.5) · "
            "🟡 Couverture partielle (0.3–0.5) · "
            "🔴 Couverture faible (<0.3)"
        )

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

    # Phase 3: Glossary sub-section after plan validation
    _render_glossary_section(state, plan)

    # Phase 3: Persona per section selector
    _render_persona_section(state, plan)

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


def _render_glossary_section(state, plan):
    """Phase 3: Glossary sub-section displayed after plan validation."""
    config = state.config
    gl_config = config.get("glossary", {})
    if not gl_config.get("enabled", False):
        return

    st.markdown("---")
    with st.expander("Glossaire terminologique (Phase 3)", expanded=False):
        project_id = st.session_state.current_project
        project_dir = PROJECTS_DIR / project_id

        from src.core.glossary_engine import GlossaryEngine
        glossary = GlossaryEngine(
            project_dir=project_dir,
            max_terms_per_prompt=gl_config.get("max_terms_per_prompt", 15),
            enabled=True,
        )

        terms = glossary.get_all_terms()
        if terms:
            st.markdown(f"**{len(terms)} terme(s) dans le glossaire**")
            for t in terms:
                abbr = f" ({t.get('abbreviation')})" if t.get("abbreviation") else ""
                st.markdown(f"- **{t['term']}{abbr}** : {t['definition']}")
                if t.get("preferred_form") and t["preferred_form"] != t["term"]:
                    st.caption(f"  Forme préférée : {t['preferred_form']}")
                if t.get("avoid_forms"):
                    st.caption(f"  Éviter : {', '.join(t['avoid_forms'])}")
        else:
            st.info("Aucun terme dans le glossaire.")

        # Auto-generate button
        provider = st.session_state.get("provider")
        if provider and provider.is_available():
            if st.button("Générer le glossaire automatiquement", key="gen_glossary"):
                with st.spinner("Génération du glossaire..."):
                    try:
                        generated = glossary.generate_from_plan(plan, provider)
                        added = glossary.apply_generated_terms(generated)
                        state.glossary = {"terms": glossary.get_all_terms()}
                        _save_state(state)
                        st.success(f"{added} terme(s) ajouté(s) au glossaire.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur : {e}")

        # Manual add
        st.markdown("---")
        st.markdown("**Ajouter un terme manuellement**")
        with st.form("add_term_form"):
            term_name = st.text_input("Terme")
            term_def = st.text_input("Définition")
            term_abbr = st.text_input("Abréviation (optionnel)")
            if st.form_submit_button("Ajouter"):
                if term_name and term_def:
                    try:
                        glossary.add_term(term=term_name, definition=term_def, abbreviation=term_abbr or None)
                        state.glossary = {"terms": glossary.get_all_terms()}
                        _save_state(state)
                        st.success(f"Terme '{term_name}' ajouté.")
                        st.rerun()
                    except ValueError as e:
                        st.warning(str(e))
                else:
                    st.warning("Renseignez le terme et la définition.")


def _render_persona_section(state, plan):
    """Phase 3: Persona per section selector (optional)."""
    config = state.config
    p_config = config.get("personas", {})
    if not p_config.get("enabled", False):
        return

    st.markdown("---")
    with st.expander("Personas par section (Phase 3)", expanded=False):
        project_id = st.session_state.current_project
        project_dir = PROJECTS_DIR / project_id

        from src.core.persona_engine import PersonaEngine
        persona_engine = PersonaEngine(project_dir=project_dir, enabled=True)

        personas = persona_engine.list_personas()
        if not personas:
            st.info("Aucun persona configuré. Créez-en dans Configuration > Phase 3 > Personas.")

            # AI suggestion button
            provider = st.session_state.get("provider")
            if provider and provider.is_available() and plan.objective:
                if st.button("Suggérer des personas par l'IA", key="suggest_personas"):
                    with st.spinner("Suggestion en cours..."):
                        try:
                            suggested = persona_engine.suggest_personas(
                                plan, plan.objective, provider,
                            )
                            for p in suggested:
                                persona_engine.create(
                                    name=p.get("name", "Persona"),
                                    profile=p.get("profile", ""),
                                    expertise_level=p.get("expertise_level", "intermédiaire"),
                                    expectations=p.get("expectations", ""),
                                    register=p.get("register", "formel"),
                                )
                            state.personas = {"personas": persona_engine.list_personas()}
                            _save_state(state)
                            st.success(f"{len(suggested)} persona(s) créé(s).")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur : {e}")
            return

        # Show personas
        primary = persona_engine.get_primary()
        primary_id = primary["id"] if primary else None

        st.markdown(f"**{len(personas)} persona(s) configuré(s)**")
        for p in personas:
            is_primary = p["id"] == primary_id
            marker = " (principal)" if is_primary else ""
            st.markdown(f"- **{p['name']}{marker}** — {p.get('profile', '')[:80]}")

        # Per-section assignment
        st.markdown("---")
        st.markdown("**Assigner un persona par section**")
        persona_options = {p["name"]: p["id"] for p in personas}
        persona_names = ["(Principal)"] + list(persona_options.keys())

        for section in plan.sections:
            current_assignment = persona_engine.get_section_assignment(section.id)
            current_persona_name = "(Principal)"
            for p in personas:
                if p["id"] == current_assignment:
                    current_persona_name = p["name"]
                    break

            selected = st.selectbox(
                f"{section.id} {section.title}",
                persona_names,
                index=persona_names.index(current_persona_name) if current_persona_name in persona_names else 0,
                key=f"persona_assign_{section.id}",
            )

            if selected != "(Principal)" and selected != current_persona_name:
                pid = persona_options.get(selected)
                if pid:
                    persona_engine.assign_to_section(section.id, pid)
            elif selected == "(Principal)" and current_assignment:
                # Remove specific assignment (use primary)
                persona_engine.clear_section_assignment(section.id)


def _save_state(state):
    """Sauvegarde l'état du projet."""
    project_id = st.session_state.current_project
    if project_id:
        state_path = PROJECTS_DIR / project_id / "state.json"
        save_json(state_path, state.to_dict())
