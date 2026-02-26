"""Estimation et suivi des coûts d'utilisation des API."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
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
    task_type: str = ""  # "generation", "summary", "plan", etc.


@dataclass
class CostReport:
    """Rapport de coûts cumulés."""
    entries: list[CostEntry] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0

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
            "entries": [
                {
                    "section_id": e.section_id,
                    "model": e.model,
                    "provider": e.provider,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "cost_usd": round(e.cost_usd, 6),
                    "task_type": e.task_type,
                }
                for e in self.entries
            ],
        }


class CostTracker:
    """Estimation et suivi des coûts API."""

    def __init__(self):
        self._pricing = load_model_pricing()
        self._report = CostReport()

    def get_model_pricing(self, provider: str, model: str) -> Optional[dict]:
        """Retourne les tarifs d'un modèle."""
        provider_pricing = self._pricing.get(provider, {})
        return provider_pricing.get(model)

    def calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calcule le coût d'un appel API."""
        pricing = self.get_model_pricing(provider, model)
        if not pricing:
            logger.warning(f"Tarifs non trouvés pour {provider}/{model}")
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
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
        """Enregistre un appel API et retourne l'entrée de coût."""
        cost = self.calculate_cost(provider, model, input_tokens, output_tokens)
        entry = CostEntry(
            section_id=section_id,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_type=task_type,
        )
        self._report.add(entry)
        return entry

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
