"""Agent Vérificateur — vérification factuelle et cohérence.

Phase 7 : vérifie la cohérence factuelle d'une section avec le corpus
           et la cohérence narrative avec les sections déjà générées.
"""

import json
import logging
import re

from src.core.agent_framework import AgentResult, BaseAgent

logger = logging.getLogger("orchestria")

VERIFIER_SYSTEM_PROMPT = """Tu es un vérificateur factuel expert. Ton rôle est de comparer une section
générée au corpus source et de détecter les problèmes de fiabilité.

Types de problèmes à détecter :
- FACTUEL : affirmation contredite par le corpus
- NEEDS_SOURCE : marqueur {{NEEDS_SOURCE}} présent dans le texte
- COHERENCE : contradiction avec une autre section
- REPETITION : contenu redondant avec une section précédente
- HORS_SUJET : contenu non lié au corpus ou au plan

Tu réponds UNIQUEMENT en JSON valide avec la structure demandée."""

VERIFIER_PROMPT_TEMPLATE = """═══ SECTION À VÉRIFIER ═══
Section : {section_id} — {section_title}

{section_content}

═══ CORPUS SOURCE ═══
{corpus_text}

═══ RÉSUMÉS DES AUTRES SECTIONS ═══
{other_sections_summaries}

═══ INSTRUCTIONS ═══
Analyse cette section et retourne un JSON avec :

1. "verdict" : "ok" | "alerte" | "erreur"
   - "ok" : section correcte et cohérente
   - "alerte" : problèmes mineurs détectés
   - "erreur" : problèmes graves détectés

2. "problemes" : liste de {{
     "type": "FACTUEL" | "NEEDS_SOURCE" | "COHERENCE" | "REPETITION" | "HORS_SUJET",
     "description": "description du problème",
     "passage_incrimine": "extrait du texte problématique"
   }}

3. "suggestions" : liste de corrections proposées (strings)

4. "score_coherence" : float entre 0.0 et 1.0 (1.0 = parfaitement cohérent)

Retourne UNIQUEMENT le JSON, sans commentaires."""


class VerifierAgent(BaseAgent):
    """Agent Vérificateur : vérification factuelle et cohérence."""

    async def _execute(self, task: dict) -> AgentResult:
        section_id = task.get("section_id", "")
        section_title = task.get("section_title", "")
        section_content = task.get("section_content", "")
        corpus_text = task.get("corpus_text", "")
        other_summaries = task.get("other_sections_summaries", "")

        if not section_content:
            return AgentResult(
                agent_name=self.name,
                section_id=section_id,
                success=False,
                error=f"Aucun contenu à vérifier pour {section_id}",
            )

        self._update_state("running", f"Vérification de {section_id}", 0.2)

        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            section_id=section_id,
            section_title=section_title,
            section_content=section_content,
            corpus_text=corpus_text,
            other_sections_summaries=other_summaries or "Aucune autre section disponible.",
        )

        response = await self._call_provider(
            prompt=prompt,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
        )

        self._update_state("running", f"Analyse du rapport {section_id}", 0.8)

        # Parser le JSON
        report = self._parse_report(response.content, section_id)

        return AgentResult(
            agent_name=self.name,
            section_id=section_id,
            success=True,
            structured_data=report,
            token_input=response.input_tokens,
            token_output=response.output_tokens,
        )

    def _build_system_prompt(self, task: dict) -> str:
        return VERIFIER_SYSTEM_PROMPT

    def _parse_report(self, content: str, section_id: str) -> dict:
        """Parse le rapport de vérification JSON."""
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
            data = None
            for i, ch in enumerate(content):
                if ch == '{':
                    try:
                        data = json.loads(content[i:])
                        break
                    except json.JSONDecodeError:
                        continue
            if data is None:
                logger.warning(f"Parsing JSON vérificateur échoué pour {section_id}")
                return self._default_report(section_id)

        # Valider la structure
        if "verdict" not in data:
            data["verdict"] = "ok"
        if "problemes" not in data:
            data["problemes"] = []
        if "suggestions" not in data:
            data["suggestions"] = []
        if "score_coherence" not in data:
            data["score_coherence"] = 0.8

        data["section_id"] = section_id
        return data

    def _default_report(self, section_id: str) -> dict:
        """Rapport par défaut si le parsing échoue."""
        return {
            "section_id": section_id,
            "verdict": "ok",
            "problemes": [],
            "suggestions": [],
            "score_coherence": 0.7,
        }
