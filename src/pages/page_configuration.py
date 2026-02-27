"""Page de configuration — Paramètres du projet et du fournisseur IA (Phase 2 + Phase 3)."""

import os
import streamlit as st

from src.core.checkpoint_manager import CheckpointConfig
from src.core.cost_tracker import CostTracker
from src.utils.config import ROOT_DIR
from src.utils.file_utils import save_json
from src.utils.providers_registry import PROVIDERS_INFO, create_provider


PROJECTS_DIR = ROOT_DIR / "projects"


def _save_state(state):
    """Sauvegarde l'état du projet sur disque."""
    project_id = st.session_state.current_project
    if project_id:
        state_path = PROJECTS_DIR / project_id / "state.json"
        save_json(state_path, state.to_dict())


def render():
    st.title("Configuration")
    st.info(
        "**Étape 1/5** — Configurez votre fournisseur IA (clé API), choisissez le "
        "modèle de génération et ajustez les paramètres (température, tokens max). "
        "Ces réglages seront utilisés pour toute la génération du document."
    )

    # Message de restauration de projet (C1-E)
    restore_msg = st.session_state.pop("_restore_message", None)
    if restore_msg:
        st.info(restore_msg)

    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif. Créez ou ouvrez un projet depuis la page Accueil.")
        return

    _render_api_config()
    st.markdown("---")
    _render_model_config()
    st.markdown("---")
    _render_mode_config()
    st.markdown("---")

    with st.expander("Points de controle (HITL)", expanded=False):
        _render_checkpoint_config()

    with st.expander("Charte graphique", expanded=False):
        _render_styling_config()

    with st.expander("RAG et génération conditionnelle", expanded=False):
        _render_rag_config()

    with st.expander("Export / Import de configuration", expanded=False):
        _render_config_export_import()

    # Phase 3 sections
    with st.expander("Phase 3 — Intelligence du pipeline", expanded=False):
        _render_phase3_config()

    # Phase 5 — Gemini advanced config (shown only if Google provider is selected)
    config = st.session_state.project_state.config
    if config.get("default_provider") == "google":
        with st.expander("Phase 5 — Configuration Gemini Avancée", expanded=False):
            _render_gemini_advanced_config()

    # Navigation inter-étapes
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Retour à l'accueil", use_container_width=True):
            st.session_state.current_page = "accueil"
            st.rerun()
    with col_next:
        config = st.session_state.project_state.config
        provider = st.session_state.get("provider")
        if provider and provider.is_available():
            if st.button("Passer à l'acquisition du corpus →", type="primary", use_container_width=True):
                _save_state(st.session_state.project_state)
                st.session_state.current_page = "acquisition"
                st.rerun()
        else:
            st.button("Passer à l'acquisition du corpus →", type="primary", use_container_width=True, disabled=True)
            st.caption("Configurez et validez votre fournisseur IA avant de continuer.")


def _render_api_config():
    """Configuration du fournisseur et de la clé API."""
    st.subheader("Fournisseur IA")

    config = st.session_state.project_state.config
    current_provider = config.get("default_provider", "openai")

    provider_names = list(PROVIDERS_INFO.keys())
    provider_labels = [PROVIDERS_INFO[p]["label"] for p in provider_names]

    current_idx = provider_names.index(current_provider) if current_provider in provider_names else 0
    selected_label = st.selectbox("Fournisseur", provider_labels, index=current_idx)
    selected_provider = provider_names[provider_labels.index(selected_label)]

    info = PROVIDERS_INFO[selected_provider]

    # Vérifier la clé dans l'environnement
    api_key_env = os.environ.get(info["env_var"], "")
    has_env_key = bool(api_key_env and api_key_env != info["placeholder"])

    if has_env_key:
        st.success(f"Clé API {info['label']} détectée dans le fichier .env")
        masked = api_key_env[:8] + "..." + api_key_env[-4:] if len(api_key_env) > 12 else "***"
        st.code(masked)
    else:
        st.warning(f"Aucune clé API {info['label']} détectée. Saisissez votre clé ou configurez {info['env_var']} dans .env")

    api_key_input = st.text_input(
        f"Clé API {info['label']} (session uniquement)",
        type="password",
        help="La clé est stockée en session uniquement et n'est pas sauvegardée sur le disque.",
        key=f"api_key_{selected_provider}",
    )

    if st.button("Valider la connexion", type="primary"):
        key = api_key_input or api_key_env
        if not key:
            st.error("Veuillez saisir une clé API.")
            return

        provider = create_provider(selected_provider, key)
        if provider and provider.is_available():
            st.session_state.provider = provider
            st.session_state.cost_tracker = CostTracker()
            old_provider = config.get("default_provider")
            config["default_provider"] = selected_provider
            # Mettre à jour le modèle si le fournisseur a changé
            if old_provider != selected_provider:
                default_model = PROVIDERS_INFO[selected_provider]["default_model"]
                config["model"] = default_model
            st.session_state.project_state.config = config
            _save_state(st.session_state.project_state)
            st.success(f"Connexion {info['label']} validée !")
        else:
            st.error("Clé API invalide ou vide.")


def _render_model_config():
    """Configuration du modèle IA."""
    st.subheader("Modèle de génération")

    config = st.session_state.project_state.config
    current_provider_name = config.get("default_provider", "openai")
    info = PROVIDERS_INFO.get(current_provider_name, PROVIDERS_INFO["openai"])

    provider = st.session_state.get("provider")
    if provider:
        models = provider.list_models()
    else:
        models = info["models"]

    current_model = config.get("model", info["default_model"])
    if current_model not in models:
        # Display with first available model selected; actual config update
        # happens when user clicks "Sauvegarder" to avoid state mutation during render.
        current_model = models[0]

    model = st.selectbox("Modèle", models, index=models.index(current_model) if current_model in models else 0)

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Température", 0.0, 1.0,
            value=config.get("temperature", 0.7),
            step=0.1,
            help="0 = déterministe, 1 = créatif",
        )
    with col2:
        max_tokens = st.number_input(
            "Tokens max par section", 512, 16384,
            value=config.get("max_tokens", 4096),
            step=512,
        )

    num_passes = st.number_input(
        "Nombre de passes de génération",
        min_value=1, max_value=5,
        value=config.get("number_of_passes", 1),
        help="1 = brouillon seul, 2+ = brouillon + raffinement(s)",
    )

    if st.button("Sauvegarder la configuration du modèle"):
        config["model"] = model
        config["temperature"] = temperature
        config["max_tokens"] = max_tokens
        config["number_of_passes"] = num_passes
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Configuration sauvegardée.")


def _render_mode_config():
    """Configuration du mode (manuel / agentique)."""
    st.subheader("Mode d'exécution")

    config = st.session_state.project_state.config
    current_mode = config.get("mode", "manual")

    mode = st.radio(
        "Mode du pipeline",
        ["manual", "agentic"],
        index=0 if current_mode == "manual" else 1,
        format_func=lambda x: "Manuel" if x == "manual" else "Agentique",
        help=(
            "**Manuel** : contrôle étape par étape avec validation à chaque checkpoint.\n"
            "**Agentique** : l'IA orchestre le pipeline de bout en bout, "
            "avec consultation uniquement aux checkpoints activés."
        ),
        horizontal=True,
    )

    if mode == "agentic":
        st.info(
            "En mode agentique, seuls les checkpoints 'Validation du plan' et "
            "'Relecture finale' sont activés par défaut. Les erreurs API sont "
            "gérées automatiquement avec retry."
        )

    if st.button("Appliquer le mode"):
        config["mode"] = mode
        if mode == "agentic":
            config.setdefault("checkpoints", {})
            config["checkpoints"]["after_plan_validation"] = True
            config["checkpoints"]["after_corpus_acquisition"] = False
            config["checkpoints"]["after_extraction"] = False
            config["checkpoints"]["after_prompt_generation"] = False
            config["checkpoints"]["after_generation"] = False
            config["checkpoints"]["final_review"] = True
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success(f"Mode {'agentique' if mode == 'agentic' else 'manuel'} activé.")


def _render_checkpoint_config():
    """Configuration des checkpoints HITL."""
    st.markdown("Activez les étapes où vous souhaitez valider manuellement le résultat.")

    config = st.session_state.project_state.config
    checkpoints = config.get("checkpoints", {})

    cp_plan = st.checkbox("Validation du plan", value=checkpoints.get("after_plan_validation", True))
    cp_corpus = st.checkbox("Après acquisition du corpus", value=checkpoints.get("after_corpus_acquisition", False))
    cp_extraction = st.checkbox("Après extraction", value=checkpoints.get("after_extraction", False))
    cp_prompt = st.checkbox("Après génération des prompts", value=checkpoints.get("after_prompt_generation", False))
    cp_gen = st.checkbox("Après génération de chaque section", value=checkpoints.get("after_generation", False))
    cp_final = st.checkbox("Relecture finale", value=checkpoints.get("final_review", True))

    if st.button("Sauvegarder les checkpoints"):
        config["checkpoints"] = {
            "after_plan_validation": cp_plan,
            "after_corpus_acquisition": cp_corpus,
            "after_extraction": cp_extraction,
            "after_prompt_generation": cp_prompt,
            "after_generation": cp_gen,
            "final_review": cp_final,
        }
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Checkpoints sauvegardés.")


def _render_styling_config():
    """Configuration de la charte graphique."""
    config = st.session_state.project_state.config
    styling = config.get("styling", {})

    col1, col2 = st.columns(2)
    with col1:
        primary_color = st.color_picker("Couleur principale", value=styling.get("primary_color", "#F0C441"))
        font_title = st.text_input("Police des titres", value=styling.get("font_title", "Calibri"))
        font_size_title = st.number_input("Taille des titres", 10, 36, value=styling.get("font_size_title", 16))
    with col2:
        secondary_color = st.color_picker("Couleur secondaire", value=styling.get("secondary_color", "#4E4E50"))
        font_body = st.text_input("Police du texte", value=styling.get("font_body", "Calibri"))
        font_size_body = st.number_input("Taille du texte", 8, 18, value=styling.get("font_size_body", 11))

    if st.button("Sauvegarder la charte graphique"):
        config["styling"] = {
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "font_title": font_title,
            "font_body": font_body,
            "font_size_title": font_size_title,
            "font_size_body": font_size_body,
            "margin_top_cm": styling.get("margin_top_cm", 2.5),
            "margin_bottom_cm": styling.get("margin_bottom_cm", 2.5),
            "margin_left_cm": styling.get("margin_left_cm", 2.5),
            "margin_right_cm": styling.get("margin_right_cm", 2.5),
        }
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Charte graphique sauvegardée.")


def _render_rag_config():
    """Configuration du RAG et de la génération conditionnelle."""
    config = st.session_state.project_state.config

    st.markdown("**Paramètres RAG (Retrieval-Augmented Generation)**")

    rag_top_k = st.number_input(
        "Nombre de blocs RAG par section (top-K)",
        min_value=1, max_value=20,
        value=config.get("rag_top_k", 7),
        help="Nombre de blocs de corpus les plus pertinents injectés dans le prompt.",
    )

    st.markdown("**Génération conditionnelle**")

    conditional_enabled = st.checkbox(
        "Activer la génération conditionnelle",
        value=config.get("conditional_generation_enabled", True),
        help="Évalue la couverture du corpus avant de générer chaque section.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sufficient = st.number_input(
            "Seuil suffisant",
            min_value=0.0, max_value=1.0,
            value=config.get("coverage_sufficient_threshold", 0.5),
            step=0.05,
        )
    with col2:
        insufficient = st.number_input(
            "Seuil insuffisant",
            min_value=0.0, max_value=1.0,
            value=config.get("coverage_insufficient_threshold", 0.3),
            step=0.05,
        )
    with col3:
        min_blocks = st.number_input(
            "Blocs minimum",
            min_value=0, max_value=10,
            value=config.get("coverage_min_blocks", 3),
        )

    if st.button("Sauvegarder la configuration RAG"):
        config["rag_top_k"] = rag_top_k
        config["conditional_generation_enabled"] = conditional_enabled
        config["coverage_sufficient_threshold"] = sufficient
        config["coverage_insufficient_threshold"] = insufficient
        config["coverage_min_blocks"] = min_blocks
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Configuration RAG sauvegardée.")


def _render_phase3_config():
    """Configuration Phase 3 — Intelligence du pipeline."""
    config = st.session_state.project_state.config

    st.markdown("### Évaluation de la qualité")
    qe_config = config.get("quality_evaluation", {})
    qe_enabled = st.checkbox("Activer l'évaluation automatique", value=qe_config.get("enabled", True), key="qe_enabled")

    st.markdown("### Vérification factuelle")
    fc_config = config.get("factcheck", {})
    fc_enabled = st.checkbox("Activer la vérification factuelle", value=fc_config.get("enabled", True), key="fc_enabled")

    st.markdown("### Feedback loop")
    fb_config = config.get("feedback_loop", {})
    fb_enabled = st.checkbox("Activer le feedback loop", value=fb_config.get("enabled", True), key="fb_enabled")

    st.markdown("### Glossaire terminologique")
    gl_config = config.get("glossary", {})
    gl_enabled = st.checkbox("Activer le glossaire", value=gl_config.get("enabled", False), key="gl_enabled")

    st.markdown("### Citations APA")
    cit_config = config.get("citations", {})
    cit_enabled = st.checkbox("Activer les citations APA", value=cit_config.get("enabled", False), key="cit_enabled")

    st.markdown("### GROBID (extraction de métadonnées)")
    gr_config = config.get("grobid", {})
    gr_enabled = st.checkbox("Activer GROBID", value=gr_config.get("enabled", False), key="gr_enabled",
                              help="Nécessite un conteneur Docker GROBID actif")
    gr_server_url = None
    if gr_enabled:
        gr_server_url = st.text_input("URL du serveur GROBID", value=gr_config.get("server_url", "http://localhost:8070"), key="gr_url")

    st.markdown("### Personas")
    p_config = config.get("personas", {})
    p_enabled = st.checkbox("Activer les personas", value=p_config.get("enabled", False), key="p_enabled")

    if st.button("Sauvegarder la configuration Phase 3"):
        config.setdefault("quality_evaluation", {})["enabled"] = qe_enabled
        config.setdefault("factcheck", {})["enabled"] = fc_enabled
        config.setdefault("feedback_loop", {})["enabled"] = fb_enabled
        config.setdefault("glossary", {})["enabled"] = gl_enabled
        config.setdefault("citations", {})["enabled"] = cit_enabled
        config.setdefault("grobid", {})["enabled"] = gr_enabled
        if gr_server_url is not None:
            config["grobid"]["server_url"] = gr_server_url
        config.setdefault("personas", {})["enabled"] = p_enabled
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Configuration Phase 3 sauvegardée.")

    # ── Instructions persistantes (M1) ──
    st.markdown("---")
    _render_persistent_instructions_config(config)

    # ── Personas configuration (M1) ──
    if p_enabled:
        st.markdown("---")
        _render_personas_config(config)

    # ── Profils personnalisés (M1) ──
    st.markdown("---")
    _render_custom_profiles_config(config)

    # HITL Journal export
    st.markdown("### Journal HITL")
    if st.button("Exporter le journal HITL (Excel)"):
        try:
            from src.core.hitl_journal import HITLJournal
            from src.utils.file_utils import ensure_dir
            journal = HITLJournal()
            output_dir = ensure_dir(ROOT_DIR / "output")
            filepath = output_dir / "hitl_journal.xlsx"
            journal.export_to_excel(filepath)
            st.success(f"Journal exporté : {filepath}")
        except Exception as e:
            st.error(f"Erreur export journal HITL : {e}")


def _render_persistent_instructions_config(config):
    """M1: Persistent instructions editor with 4 category tabs and 3 hierarchy levels."""
    st.markdown("### Instructions persistantes")
    st.caption("Définissez des instructions par catégorie et par niveau hiérarchique (Projet > Contexte > Section).")

    project_id = st.session_state.current_project
    if not project_id:
        return

    from src.core.persistent_instructions import PersistentInstructions, CATEGORIES, CATEGORY_LABELS

    project_dir = PROJECTS_DIR / project_id

    pi = PersistentInstructions(project_dir=project_dir)

    # Category tabs
    tab_labels = [CATEGORY_LABELS[cat] for cat in CATEGORIES]
    tabs = st.tabs(tab_labels)

    for tab, category in zip(tabs, CATEGORIES):
        with tab:
            label = CATEGORY_LABELS[category]

            # Project level
            st.markdown(f"**Niveau Projet** — {label}")
            project_val = pi.get_project_instructions().get(category, "")
            new_project = st.text_area(
                f"Instruction projet ({label})",
                value=project_val,
                height=80,
                key=f"pi_project_{category}",
                label_visibility="collapsed",
            )
            if new_project != project_val:
                if st.button(f"Sauvegarder (projet/{label})", key=f"save_pi_project_{category}"):
                    pi.set_project_instruction(category, new_project)
                    st.success("Instruction sauvegardée.")

            # Context level
            st.markdown(f"**Niveau Contexte** — {label}")
            contexts = pi.list_contexts()
            if contexts:
                for ctx in contexts:
                    ctx_val = pi.get_context_instructions(ctx).get(category, "")
                    new_ctx = st.text_area(
                        f"Contexte '{ctx}' ({label})",
                        value=ctx_val,
                        height=60,
                        key=f"pi_ctx_{ctx}_{category}",
                    )
                    if new_ctx != ctx_val:
                        if st.button(f"Sauvegarder (ctx:{ctx}/{label})", key=f"save_pi_ctx_{ctx}_{category}"):
                            pi.set_context_instruction(ctx, category, new_ctx)
                            st.success("Instruction sauvegardée.")
            else:
                new_ctx_name = st.text_input(
                    f"Créer un contexte ({label})",
                    placeholder="ex: introduction, analyse, conclusion",
                    key=f"new_ctx_{category}",
                )
                if new_ctx_name and st.button(f"Créer contexte ({label})", key=f"create_ctx_{category}"):
                    pi.set_context_instruction(new_ctx_name.strip(), category, "")
                    st.success(f"Contexte '{new_ctx_name}' créé.")
                    st.rerun()


def _render_personas_config(config):
    """M1: Personas creation/edition UI with AI suggestion."""
    st.markdown("### Personas cibles")

    project_id = st.session_state.current_project
    if not project_id:
        return

    project_dir = PROJECTS_DIR / project_id

    from src.core.persona_engine import PersonaEngine
    engine = PersonaEngine(project_dir=project_dir, enabled=True)

    personas = engine.list_personas()

    # Display existing personas
    if personas:
        primary = engine.get_primary()
        primary_id = primary["id"] if primary else None

        for p in personas:
            is_primary = p["id"] == primary_id
            marker = " (Principal)" if is_primary else ""
            with st.expander(f"{p['name']}{marker}", expanded=False):
                st.markdown(f"**Profil :** {p.get('profile', '')}")
                st.markdown(f"**Expertise :** {p.get('expertise_level', 'N/A')}")
                st.markdown(f"**Attentes :** {p.get('expectations', '')}")
                st.markdown(f"**Registre :** {p.get('register', 'formel')}")

                col1, col2 = st.columns(2)
                with col1:
                    if not is_primary:
                        if st.button("Définir comme principal", key=f"set_primary_{p['id']}"):
                            engine.set_primary(p["id"])
                            st.success(f"{p['name']} est maintenant le persona principal.")
                            st.rerun()
                with col2:
                    if st.button("Supprimer", key=f"del_persona_{p['id']}"):
                        engine.delete(p["id"])
                        st.success(f"Persona '{p['name']}' supprimé.")
                        st.rerun()

        # Primary selector
        persona_names = [p["name"] for p in personas]
        current_primary_name = primary["name"] if primary else persona_names[0]
        selected_primary = st.selectbox(
            "Persona principal",
            persona_names,
            index=persona_names.index(current_primary_name) if current_primary_name in persona_names else 0,
            key="primary_persona_select",
        )
        if selected_primary != current_primary_name:
            pid = next((p["id"] for p in personas if p["name"] == selected_primary), None)
            if pid:
                engine.set_primary(pid)
                st.rerun()

    # AI suggestion
    provider = st.session_state.get("provider")
    if provider and provider.is_available():
        if st.button("Suggérer par l'IA", key="suggest_personas_config"):
            state = st.session_state.project_state
            objective = state.config.get("objective", "")
            if objective and state.plan:
                with st.spinner("Suggestion en cours..."):
                    try:
                        suggested = engine.suggest_personas(state.plan, objective, provider)
                        for p in suggested:
                            engine.create(
                                name=p.get("name", "Persona"),
                                profile=p.get("profile", ""),
                                expertise_level=p.get("expertise_level", "intermédiaire"),
                                expectations=p.get("expectations", ""),
                                register=p.get("register", "formel"),
                            )
                        st.success(f"{len(suggested)} persona(s) suggéré(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            else:
                st.warning("Définissez un objectif et un plan avant de suggérer des personas.")

    # Create form
    st.markdown("**Créer un persona**")
    with st.form("create_persona_form"):
        p_name = st.text_input("Nom")
        p_profile = st.text_area("Profil (description)", height=60)
        p_expertise = st.selectbox("Niveau d'expertise", ["débutant", "intermédiaire", "expert", "décideur"])
        p_expectations = st.text_input("Attentes principales")
        p_register = st.selectbox("Registre de communication", ["familier", "courant", "formel", "académique"])

        if st.form_submit_button("Créer le persona"):
            if p_name and p_profile:
                try:
                    engine.create(
                        name=p_name,
                        profile=p_profile,
                        expertise_level=p_expertise,
                        expectations=p_expectations,
                        register=p_register,
                    )
                    st.success(f"Persona '{p_name}' créé.")
                    st.rerun()
                except ValueError as e:
                    st.warning(str(e))
            else:
                st.warning("Renseignez le nom et le profil.")


def _render_custom_profiles_config(config):
    """M1: Custom profiles save/delete UI."""
    st.markdown("### Profils personnalisés")
    st.caption("Sauvegardez la configuration actuelle comme profil réutilisable.")

    from src.core.profile_manager import ProfileManager
    manager = ProfileManager()

    # List custom profiles
    custom_profiles = manager.list_custom_profiles()
    if custom_profiles:
        st.markdown(f"**{len(custom_profiles)} profil(s) personnalisé(s)**")
        for cp in custom_profiles:
            col_info, col_del = st.columns([4, 1])
            with col_info:
                st.markdown(f"- **{cp['name']}** : {cp.get('description', '')}")
            with col_del:
                if st.button("Supprimer", key=f"del_profile_{cp['id']}"):
                    try:
                        manager.delete_custom_profile(cp["id"])
                        st.success(f"Profil '{cp['name']}' supprimé.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
    else:
        st.info("Aucun profil personnalisé sauvegardé.")

    # Save as custom profile
    with st.form("save_custom_profile"):
        profile_name = st.text_input("Nom du profil")
        profile_desc = st.text_input("Description (optionnel)")

        if st.form_submit_button("Sauvegarder comme profil personnalisé", type="primary"):
            if profile_name:
                try:
                    manager.save_custom_profile(
                        name=profile_name,
                        description=profile_desc,
                        config=config,
                    )
                    st.success(f"Profil '{profile_name}' sauvegardé.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("Renseignez un nom pour le profil.")


def _render_config_export_import():
    """Export et import de la configuration du projet."""
    import yaml

    config = st.session_state.project_state.config

    st.markdown("**Exporter la configuration**")
    config_yaml = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)

    st.download_button(
        "Télécharger la configuration (YAML)",
        data=config_yaml,
        file_name=f"config_{st.session_state.project_state.name}.yaml",
        mime="text/yaml",
    )

    st.markdown("**Importer une configuration**")
    uploaded = st.file_uploader("Charger un fichier YAML", type=["yaml", "yml"], key="config_import")
    if uploaded is not None:
        try:
            imported_config = yaml.safe_load(uploaded.read())
            if isinstance(imported_config, dict):
                if st.button("Appliquer la configuration importée"):
                    config.update(imported_config)
                    st.session_state.project_state.config = config
                    _save_state(st.session_state.project_state)
                    st.success("Configuration importée et appliquée.")
                    st.rerun()
            else:
                st.error("Le fichier ne contient pas une configuration valide.")
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")


def _render_gemini_advanced_config():
    """Configuration avancée pour Google Gemini 3.1 — Context Caching et Thinking Level.

    Affiché uniquement lorsque le fournisseur actif est Google Gemini.
    """
    config = st.session_state.project_state.config
    gemini_cfg = config.get("gemini", {})
    current_model = config.get("model", "gemini-3-flash-preview")

    st.markdown("### Paramètres de génération Gemini")

    # ── Context Caching ──────────────────────────────────────────────────────
    st.markdown("#### Context Caching")

    caching_supported = "3.1-pro" in current_model
    if not caching_supported:
        st.info(
            "ℹ️ Context caching disponible uniquement avec **gemini-3.1-pro-preview**. "
            f"Modèle actuel : `{current_model}`."
        )

    caching_enabled = st.checkbox(
        "Activer le Context Caching",
        value=gemini_cfg.get("caching_enabled", False),
        disabled=not caching_supported,
        help=(
            "Stocke le corpus une seule fois côté Google Cloud. "
            "Réduit les coûts d'environ 85% sur les tokens d'entrée. "
            "Nécessite un corpus d'au minimum 2 048 tokens."
        ),
    )

    # Bloc d'estimation des coûts (si caching activé et corpus indexé)
    if caching_enabled and caching_supported:
        state = st.session_state.project_state
        corpus_tokens = 0
        num_sections = 0
        if state.corpus:
            corpus_tokens = getattr(state.corpus, "total_tokens", 0) or 0
        if state.plan and state.plan.sections:
            num_sections = len(state.plan.sections)

        if corpus_tokens > 0 and num_sections > 0:
            try:
                from src.core.gemini_cache_manager import GeminiCacheManager
                mgr = GeminiCacheManager()
                ttl_hours = gemini_cfg.get("cache_ttl_seconds", 7200) / 3600
                estimate = mgr.estimate_cache_cost(
                    corpus_tokens=corpus_tokens,
                    num_sections=num_sections,
                    ttl_hours=ttl_hours,
                    model=current_model,
                )

                st.markdown("**Estimation des coûts (Context Caching)**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sans cache", f"~${estimate['cost_without_cache']:.2f}")
                with col2:
                    st.metric(
                        "Avec cache",
                        f"~${estimate['cost_with_cache']:.2f}",
                        delta=f"-${estimate['savings_usd']:.2f}",
                    )
                with col3:
                    st.metric("Économie", f"{estimate['savings_percent']:.0f}%")

                st.caption(
                    f"Corpus : {corpus_tokens:,} tokens | "
                    f"Sections : {num_sections} | "
                    f"Stockage : ~${estimate['storage_cost']:.3f} | "
                    f"Break-even : {estimate['break_even_sections']} section(s)"
                )

                if corpus_tokens >= 2048 and num_sections >= estimate["break_even_sections"]:
                    st.success(
                        f"✅ Context Caching est rentable pour ce projet "
                        f"(break-even à partir de {estimate['break_even_sections']} section(s))."
                    )
                elif corpus_tokens < 2048:
                    st.warning(
                        "⚠️ Corpus trop petit pour le context caching (< 2 048 tokens). "
                        "Mode standard activé automatiquement."
                    )

                # Alerte long-context
                if corpus_tokens > 200_000:
                    st.error(
                        "⚠️ Votre corpus dépasse 200 000 tokens. Gemini applique un "
                        "tarif long-context ($4/1M input, $18/1M output). "
                        f"Coût estimé sans cache : ~${estimate['cost_without_cache']:.2f}. "
                        "Envisagez de réduire le corpus ou d'utiliser un autre provider."
                    )

            except Exception:
                st.info("Indexez votre corpus pour voir l'estimation des coûts.")
        else:
            st.info("Indexez votre corpus et configurez le plan pour voir l'estimation des coûts.")

    # TTL du cache
    if caching_enabled and caching_supported:
        ttl_minutes = st.slider(
            "TTL du cache (minutes)",
            min_value=5,
            max_value=1440,
            value=int(gemini_cfg.get("cache_ttl_seconds", 7200) / 60),
            step=5,
            help="Durée de vie du cache. Recommandé : 120 min (2h) pour couvrir une session.",
        )
        gemini_cfg["cache_ttl_seconds"] = ttl_minutes * 60
    else:
        ttl_minutes = int(gemini_cfg.get("cache_ttl_seconds", 7200) / 60)

    # ── Thinking Level ───────────────────────────────────────────────────────
    st.markdown("#### Thinking Level")
    st.caption(
        "Contrôle la profondeur de raisonnement interne de Gemini 3.1 Pro avant la réponse. "
        "Disponible uniquement avec gemini-3.1-pro-preview."
    )

    thinking_supported = "3.1" in current_model
    if not thinking_supported:
        st.info(
            f"ℹ️ Thinking Level disponible uniquement avec les modèles Gemini 3.1. "
            f"Modèle actuel : `{current_model}`."
        )

    thinking_mode = st.radio(
        "Mode Thinking Level",
        options=["Auto (recommandé)", "Manuel"],
        index=0 if gemini_cfg.get("thinking_level_mode", "auto") == "auto" else 1,
        disabled=not thinking_supported,
        help=(
            "Auto : l'orchestrateur choisit le niveau optimal selon le type de tâche. "
            "Manuel : vous choisissez un niveau fixe pour toutes les tâches."
        ),
    )

    if thinking_mode == "Manuel" and thinking_supported:
        level_options = ["minimal", "low", "medium", "high"]
        current_level = gemini_cfg.get("manual_thinking_level", "medium")
        manual_level = st.select_slider(
            "Niveau de réflexion",
            options=level_options,
            value=current_level if current_level in level_options else "medium",
            help="minimal ~2s | low ~5s | medium ~15s | high ~36s",
        )
        if manual_level == "high":
            st.warning(
                "⚠️ Le niveau 'high' est plus lent (~36s par section) mais offre "
                "une qualité maximale. Recommandé pour la génération de sections critiques."
            )
        gemini_cfg["manual_thinking_level"] = manual_level
        gemini_cfg["thinking_level_mode"] = "manual"
    else:
        gemini_cfg["thinking_level_mode"] = "auto"

    # Sauvegarder
    if st.button("Sauvegarder la configuration Gemini"):
        gemini_cfg["caching_enabled"] = caching_enabled
        config["gemini"] = gemini_cfg
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Configuration Gemini sauvegardée.")
