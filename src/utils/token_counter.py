"""Comptage de tokens pour l'estimation des coûts."""


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Compte le nombre de tokens d'un texte pour un modèle donné.

    Utilise tiktoken pour les modèles OpenAI, estimation heuristique sinon.
    """
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return _heuristic_count(text)


def _heuristic_count(text: str) -> int:
    """Estimation heuristique du nombre de tokens (1 token ≈ 4 caractères en français)."""
    # B15: empty text should return 0 tokens, not 1
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_pages(token_count: int, tokens_per_page: int = 400) -> float:
    """Estime le nombre de pages à partir d'un nombre de tokens."""
    if tokens_per_page <= 0:
        return 0.0
    # B36: clamp negative token counts to 0
    return max(0, token_count) / tokens_per_page
