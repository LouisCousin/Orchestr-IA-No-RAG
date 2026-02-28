"""Estimation et suivi des coûts d'utilisation des API — Phase 5.

Gère quatre cas de calcul pour Gemini 3.1 :
  CAS 1 — Tokens standard (pas de cache)
  CAS 2 — Tokens cachés (lu depuis le cache, 90% de réduction)
  CAS 3 — Tokens long-context (input > 200 000 tokens, repricing)
  CAS 4 — Stockage cache (comptabilisé à la création)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.utils.config import load_model_pricing
from src.utils.token_counter import count_tokens

logger = logging.getLogger("orchestria")


@dataclass
class CostEntry:
    """Entrée de coût pour un appel API."""
    section_id: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    task_type: str = ""  # "generation", "summary", "plan", "cached_call", "cache_storage"
    cached_tokens: int = 0
    is_long_context: bool = False


@dataclass
class CostReport:
    """Rapport de coûts cumulés."""
    entries: list[CostEntry] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    # Phase 5 — statistiques du cache Gemini
    gemini_cache_stats: dict = field(default_factory=dict)

    def add(self, entry: CostEntry) -> None:
        self.entries.append(entry)
        self.total_input_tokens += entry.input_tokens
        self.total_output_tokens += entry.output_tokens
        self.total_cost_usd += entry.cost_usd

    def to_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "gemini_cache_stats": self.gemini_cache_stats,
            "entries": [
                {
                    "section_id": e.section_id,
                    "model": e.model,
                    "provider": e.provider,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "cost_usd": round(e.cost_usd, 6),
                    "task_type": e.task_type,
                    "cached_tokens": e.cached_tokens,
                    "is_long_context": e.is_long_context,
                }
                for e in self.entries
            ],
        }


class CostTracker:
    """Estimation et suivi des coûts API — Phase 5 avec support Gemini caching."""

    def __init__(self):
        self._pricing = load_model_pricing()
        self._report = CostReport()

    def get_model_pricing(self, provider: str, model: str) -> Optional[dict]:
        """Retourne les tarifs d'un modèle."""
        provider_pricing = self._pricing.get(provider, {})
        return provider_pricing.get(model)

    def calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calcule le coût d'un appel API standard (CAS 1)."""
        pricing = self.get_model_pricing(provider, model)
        if not pricing:
            logger.warning(f"Tarifs non trouvés pour {provider}/{model}")
            return 0.0

        # CAS 3 : vérifier repricing long-context
        long_context_threshold = pricing.get("long_context_threshold", float("inf"))
        if input_tokens > long_context_threshold:
            input_rate = pricing.get("input_long_context", pricing.get("input", 0))
            output_rate = pricing.get("output_long_context", pricing.get("output", 0))
        else:
            input_rate = pricing.get("input", 0)
            output_rate = pricing.get("output", 0)

        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        return input_cost + output_cost

    def record(
        self,
        section_id: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str = "generation",
    ) -> CostEntry:
        """Enregistre un appel API standard et retourne l'entrée de coût."""
        is_long = self.check_long_context_threshold(provider, model, input_tokens)
        cost = self.calculate_cost(provider, model, input_tokens, output_tokens)
        entry = CostEntry(
            section_id=section_id,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_type=task_type,
            is_long_context=is_long,
        )
        self._report.add(entry)
        return entry

    def track_cached_call(
        self,
        model: str,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        section_id: str = "",
    ) -> float:
        """Enregistre un appel avec tokens mixtes (CAS 2).

        Args:
            model: Modèle Gemini utilisé.
            input_tokens: Tokens non cachés dans cette requête.
            cached_tokens: Tokens lus depuis le cache.
            output_tokens: Tokens générés en sortie.
            section_id: Identifiant de la section (pour le rapport).

        Returns:
            Coût total en USD.
        """
        pricing = self.get_model_pricing("google", model)
        if not pricing:
            return 0.0

        total_input = input_tokens + cached_tokens
        is_long = total_input > pricing.get("long_context_threshold", float("inf"))

        if is_long:
            input_rate = pricing.get("input_long_context", pricing.get("input", 0))
            output_rate = pricing.get("output_long_context", pricing.get("output", 0))
        else:
            input_rate = pricing.get("input", 0)
            output_rate = pricing.get("output", 0)

        cached_rate = pricing.get("input_cached", input_rate)

        cost = (
            (input_tokens / 1_000_000) * input_rate
            + (cached_tokens / 1_000_000) * cached_rate
            + (output_tokens / 1_000_000) * output_rate
        )

        entry = CostEntry(
            section_id=section_id,
            model=model,
            provider="google",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_type="cached_call",
            cached_tokens=cached_tokens,
            is_long_context=is_long,
        )
        self._report.add(entry)

        # Mettre à jour les stats de cache
        stats = self._report.gemini_cache_stats
        stats["cache_hits"] = stats.get("cache_hits", 0) + 1
        stats["tokens_read_from_cache"] = stats.get("tokens_read_from_cache", 0) + cached_tokens

        return cost

    def track_cache_storage(
        self,
        model: str,
        cached_tokens: int,
        ttl_hours: float,
    ) -> float:
        """Comptabilise le coût de stockage d'un cache (CAS 4).

        Args:
            model: Modèle Gemini utilisé.
            cached_tokens: Nombre de tokens dans le cache.
            ttl_hours: Durée de vie du cache en heures.

        Returns:
            Coût de stockage en USD.
        """
        pricing = self.get_model_pricing("google", model)
        if not pricing:
            return 0.0

        storage_rate = pricing.get("cache_storage_per_hour", 0.50)
        cost = (cached_tokens / 1_000_000) * storage_rate * ttl_hours

        entry = CostEntry(
            section_id="__cache_storage__",
            model=model,
            provider="google",
            input_tokens=0,
            output_tokens=0,
            cost_usd=cost,
            task_type="cache_storage",
        )
        self._report.add(entry)

        # Mettre à jour les stats de cache
        stats = self._report.gemini_cache_stats
        stats["storage_cost_usd"] = stats.get("storage_cost_usd", 0.0) + cost

        return cost

    def check_long_context_threshold(
        self,
        provider: str,
        model: str,
        total_input_tokens: int,
    ) -> bool:
        """Retourne True si total_input_tokens dépasse le seuil long-context.

        Déclenche un warning loggé en cas de dépassement.

        Args:
            provider: Fournisseur du modèle.
            model: Identifiant du modèle.
            total_input_tokens: Nombre total de tokens en entrée.

        Returns:
            True si le seuil long-context est dépassé.
        """
        pricing = self.get_model_pricing(provider, model)
        if not pricing:
            return False

        threshold = pricing.get("long_context_threshold", float("inf"))
        if total_input_tokens > threshold:
            logger.warning(
                f"⚠️ Seuil long-context dépassé pour {model} : "
                f"{total_input_tokens} > {threshold} tokens. "
                f"Tarif long-context appliqué ($4/1M input, $18/1M output)."
            )
            return True
        return False

    def init_cache_stats(
        self,
        cache_name: str,
        tokens_cached: int,
    ) -> None:
        """Initialise les statistiques du cache dans le rapport.

        Args:
            cache_name: Nom du cache Gemini.
            tokens_cached: Nombre de tokens dans le cache.
        """
        self._report.gemini_cache_stats = {
            "cache_name": cache_name,
            "tokens_cached": tokens_cached,
            "cache_hits": 0,
            "cache_misses": 0,
            "storage_cost_usd": 0.0,
            "savings_vs_no_cache_usd": 0.0,
            "savings_percent": 0.0,
        }

    def estimate_project_cost(
        self,
        section_count: int,
        avg_corpus_tokens: int,
        provider: str,
        model: str,
        num_passes: int = 1,
    ) -> dict:
        """Estime le coût total d'un projet avant lancement."""
        pricing = self.get_model_pricing(provider, model)
        if not pricing:
            return {"error": f"Tarifs non trouvés pour {provider}/{model}"}

        # Estimation par section : prompt (~1500 tokens système + corpus) + réponse (~1000 tokens)
        system_tokens = 1500
        avg_input = system_tokens + avg_corpus_tokens
        avg_output = 1000  # Estimation moyenne de sortie

        total_input = avg_input * section_count * num_passes
        total_output = avg_output * section_count * num_passes

        # Ajouter les résumés inter-sections
        summary_input = 500 * section_count
        summary_output = 100 * section_count
        total_input += summary_input
        total_output += summary_output

        cost = self.calculate_cost(provider, model, total_input, total_output)

        self._report.estimated_cost_usd = cost
        return {
            "provider": provider,
            "model": model,
            "section_count": section_count,
            "num_passes": num_passes,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": round(cost, 4),
            "context_window": pricing.get("context_window", 0),
        }

    def estimate_multi_model(
        self,
        section_count: int,
        avg_corpus_tokens: int,
        num_passes: int = 1,
    ) -> list[dict]:
        """Estime le coût pour tous les modèles configurés."""
        estimates = []
        for provider_name, models in self._pricing.items():
            for model_name in models:
                est = self.estimate_project_cost(
                    section_count, avg_corpus_tokens, provider_name, model_name, num_passes
                )
                if "error" not in est:
                    estimates.append(est)

        return sorted(estimates, key=lambda x: x["estimated_cost_usd"])

    def estimate_corpus_cost(
        self, documents: list[dict], provider: str, model: str
    ) -> dict:
        """Estime le coût d'input pour un corpus donné."""
        pricing = self.get_model_pricing(provider, model)
        if not pricing:
            return {"error": f"Tarifs non trouvés pour {provider}/{model}"}

        total_tokens = sum(doc.get("tokens", 0) for doc in documents)
        input_cost = (total_tokens / 1_000_000) * pricing.get("input", 0)

        return {
            "total_documents": len(documents),
            "total_tokens": total_tokens,
            "estimated_input_cost_usd": round(input_cost, 6),
            "context_window": pricing.get("context_window", 0),
            "documents_exceeding_context": sum(
                1 for doc in documents
                if doc.get("tokens", 0) > pricing.get("context_window", float("inf"))
            ),
        }

    @property
    def report(self) -> CostReport:
        return self._report

    def reset(self) -> None:
        """Réinitialise le suivi."""
        self._report = CostReport()
