"""Agent Évaluateur — scoring global du document.

Phase 7 : attribue un score de qualité global au document assemblé sur 6 critères
           et décide si une passe de correction est nécessaire.
"""

import json
import logging
import re

from src.core.agent_framework import AgentResult, BaseAgent

logger = logging.getLogger("orchestria")

EVALUATOR_SYSTEM_PROMPT = """Tu es un évaluateur de qualité documentaire expert. Ton rôle est d'évaluer
la qualité globale d'un document sur 6 critères et de décider si des corrections sont nécessaires.

Tu réponds UNIQUEMENT en JSON valide avec la structure demandée."""

EVALUATOR_PROMPT_TEMPLATE = """═══ DOCUMENT À ÉVALUER ═══
{document_summary}

═══ RAPPORTS DE VÉRIFICATION ═══
{verif_summary}

═══ SEUIL DE QUALITÉ ═══
Score global minimum : {quality_threshold}/5.0
Score par section minimum : {section_threshold}/5.0

═══ INSTRUCTIONS ═══
Évalue ce document et retourne un JSON avec :

1. "score_global" : float entre 1.0 et 5.0 (moyenne pondérée des critères)

2. "scores_par_critere" : {{
     "pertinence_corpus": float 1.0-5.0,
     "precision_factuelle": float 1.0-5.0,
     "coherence_interne": float 1.0-5.0,
     "qualite_redactionnelle": float 1.0-5.0,
     "completude": float 1.0-5.0,
     "respect_plan": float 1.0-5.0
   }}

3. "scores_par_section" : {{section_id: float 1.0-5.0}} pour chaque section

4. "sections_a_corriger" : liste des section_id dont le score < seuil par section

5. "recommandation" : "exporter" | "corriger" | "regenerer"
   - "exporter" si score_global >= seuil
   - "corriger" si certaines sections sont sous le seuil
   - "regenerer" si le document est globalement insuffisant (score < 2.0)

6. "commentaire" : résumé des points forts et faibles du document

Retourne UNIQUEMENT le JSON, sans commentaires."""


class EvaluatorAgent(BaseAgent):
    """Agent Évaluateur : scoring global et décision de correction."""

    async def _execute(self, task: dict) -> AgentResult:
        sections = task.get("sections", {})
        verif_reports = task.get("verif_reports", {})
        quality_threshold = task.get("quality_threshold", 3.5)
        section_threshold = task.get("section_correction_threshold", 3.0)

        if not sections:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="Aucune section à évaluer",
            )

        self._update_state("running", "Évaluation globale du document", 0.2)

        # Construire le résumé du document (pas le corpus complet)
        document_summary = self._build_document_summary(sections)
        verif_summary = self._build_verif_summary(verif_reports)

        prompt = EVALUATOR_PROMPT_TEMPLATE.format(
            document_summary=document_summary,
            verif_summary=verif_summary,
            quality_threshold=quality_threshold,
            section_threshold=section_threshold,
        )

        response = await self._call_provider(
            prompt=prompt,
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
        )

        self._update_state("running", "Analyse du scoring", 0.8)

        eval_result = self._parse_evaluation(
            response.content, sections, quality_threshold, section_threshold
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            structured_data=eval_result,
            token_input=response.input_tokens,
            token_output=response.output_tokens,
        )

    def _build_system_prompt(self, task: dict) -> str:
        return EVALUATOR_SYSTEM_PROMPT

    def _build_document_summary(self, sections: dict) -> str:
        """Construit un résumé structuré du document."""
        parts = []
        for sid, content in sections.items():
            # Résumé de chaque section (~300 mots max)
            truncated = content[:1200] if len(content) > 1200 else content
            parts.append(f"[{sid}]\n{truncated}\n")
        return "\n".join(parts)

    def _build_verif_summary(self, verif_reports: dict) -> str:
        """Résume les rapports de vérification."""
        if not verif_reports:
            return "Aucun rapport de vérification disponible."

        parts = []
        for sid, report in verif_reports.items():
            verdict = report.get("verdict", "?")
            problems = report.get("problemes", [])
            coherence = report.get("score_coherence", 0)
            problem_types = [p.get("type", "?") for p in problems]
            parts.append(
                f"[{sid}] verdict={verdict}, cohérence={coherence:.1f}, "
                f"problèmes={len(problems)} ({', '.join(problem_types) if problem_types else 'aucun'})"
            )
        return "\n".join(parts)

    def _parse_evaluation(
        self,
        content: str,
        sections: dict,
        quality_threshold: float,
        section_threshold: float,
    ) -> dict:
        """Parse le résultat de l'évaluation JSON."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return self._default_evaluation(sections, quality_threshold)
            else:
                return self._default_evaluation(sections, quality_threshold)

        # Valider et compléter
        if "score_global" not in data:
            data["score_global"] = 3.5
        if "scores_par_critere" not in data:
            data["scores_par_critere"] = {
                "pertinence_corpus": 3.5,
                "precision_factuelle": 3.5,
                "coherence_interne": 3.5,
                "qualite_redactionnelle": 3.5,
                "completude": 3.5,
                "respect_plan": 3.5,
            }
        if "scores_par_section" not in data:
            # Fallback : attribuer le score global à chaque section
            data["scores_par_section"] = {sid: data["score_global"] for sid in sections}
        if "sections_a_corriger" not in data:
            # Déduire les sections à corriger depuis scores_par_section
            data["sections_a_corriger"] = [
                sid for sid, score in data["scores_par_section"].items()
                if score < section_threshold
            ]
        if "recommandation" not in data:
            data["recommandation"] = (
                "exporter" if data["score_global"] >= quality_threshold else "corriger"
            )
        if "commentaire" not in data:
            data["commentaire"] = ""

        return data

    def _default_evaluation(self, sections: dict, quality_threshold: float) -> dict:
        """Évaluation par défaut si le parsing échoue."""
        return {
            "score_global": 3.5,
            "scores_par_critere": {
                "pertinence_corpus": 3.5,
                "precision_factuelle": 3.5,
                "coherence_interne": 3.5,
                "qualite_redactionnelle": 3.5,
                "completude": 3.5,
                "respect_plan": 3.5,
            },
            "scores_par_section": {sid: 3.5 for sid in sections},
            "sections_a_corriger": [],
            "recommandation": "exporter",
            "commentaire": "Évaluation par défaut (parsing LLM échoué).",
        }
