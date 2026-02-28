"""Page de configuration — Client HTTP (Phase 3 Sprint 3).

Aucun import de src.core. Toutes les opérations passent par l'API.
"""

import streamlit as st

from src.client.api_client import OrchestrIAClient

PROVIDERS_INFO = {
    "openai": {"label": "OpenAI", "models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]},
    "anthropic": {"label": "Anthropic", "models": ["claude-opus-4-6", "claude-sonnet-4-5-20250514"]},
    "google": {"label": "Google Gemini", "models": ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]},
}


def _get_client() -> OrchestrIAClient:
    from src.client.app import get_client
    return get_client()


def render():
    st.title("Configuration")
    st.info(
        "**Étape 1** — Configurez votre fournisseur IA (clé API), choisissez le "
        "modèle de génération et ajustez les paramètres."
    )
    st.markdown("---")

    project_id = st.session_state.get("current_project")
    if not project_id:
        st.warning("Aucun projet actif. Créez ou ouvrez un projet depuis la page Accueil.")
        return

    client = _get_client()
    try:
        state = client.get_project_state(project_id)
    except Exception as e:
        st.error(f"Erreur : {e}")
        return

    config = state.get("config", {})

    # ── Fournisseur IA ──
    st.subheader("Fournisseur IA")

    current_provider = config.get("default_provider", "openai")
    provider_names = list(PROVIDERS_INFO.keys())
    provider_labels = [PROVIDERS_INFO[p]["label"] for p in provider_names]

    current_idx = provider_names.index(current_provider) if current_provider in provider_names else 0
    selected_label = st.selectbox("Fournisseur", provider_labels, index=current_idx)
    selected_provider = provider_names[provider_labels.index(selected_label)]

    api_key = st.text_input(
        f"Clé API {PROVIDERS_INFO[selected_provider]['label']}",
        type="password",
        help="La clé est transmise au serveur API.",
    )

    if st.button("Valider la connexion", type="primary"):
        if not api_key:
            st.error("Veuillez saisir une clé API.")
        else:
            try:
                client.configure_provider(project_id, selected_provider, api_key)
                st.success(f"Connexion {PROVIDERS_INFO[selected_provider]['label']} validée !")
            except Exception as e:
                st.error(f"Erreur : {e}")

    # ── Modèle ──
    st.markdown("---")
    st.subheader("Modèle de génération")

    models = PROVIDERS_INFO.get(selected_provider, {}).get("models", [])
    current_model = config.get("model", models[0] if models else "")
    if current_model not in models:
        current_model = models[0] if models else ""

    model = st.selectbox("Modèle", models, index=models.index(current_model) if current_model in models else 0)

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Température", 0.0, 1.0, value=config.get("temperature", 0.7), step=0.1,
        )
    with col2:
        max_tokens = st.number_input(
            "Tokens max par section", 512, 16384, value=config.get("max_tokens", 4096), step=512,
        )

    num_passes = st.number_input(
        "Nombre de passes", min_value=1, max_value=5, value=config.get("number_of_passes", 1),
    )

    if st.button("Sauvegarder la configuration"):
        try:
            client.update_config(project_id, {
                "default_provider": selected_provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "number_of_passes": num_passes,
            })
            st.success("Configuration sauvegardée.")
        except Exception as e:
            st.error(f"Erreur : {e}")

    # ── Mode ──
    st.markdown("---")
    st.subheader("Mode d'exécution")

    current_mode = config.get("mode", "manual")
    mode = st.radio(
        "Mode du pipeline",
        ["manual", "agentic"],
        index=0 if current_mode == "manual" else 1,
        format_func=lambda x: "Manuel" if x == "manual" else "Agentique",
        horizontal=True,
    )

    if st.button("Appliquer le mode"):
        try:
            client.update_config(project_id, {"mode": mode})
            st.success(f"Mode {'agentique' if mode == 'agentic' else 'manuel'} activé.")
        except Exception as e:
            st.error(f"Erreur : {e}")

    # ── Navigation ──
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Retour à l'accueil", use_container_width=True):
            st.session_state.current_page = "accueil"
            st.rerun()
    with col_next:
        if st.button("Passer à l'acquisition du corpus →", type="primary", use_container_width=True):
            st.session_state.current_page = "acquisition"
            st.rerun()
