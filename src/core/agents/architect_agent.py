"""Agent Architecte — planification et dépendances entre sections.

Phase 7 : analyse le plan fourni et produit un plan enrichi avec
           dépendances, contraintes de longueur, ton attendu et zones à risque.
"""

import json
import logging

from src.core.agent_framework import AgentResult, BaseAgent

logger = logging.getLogger("orchestria")

ARCHITECT_SYSTEM_PROMPT = """Tu es un architecte documentaire expert. Ton rôle est d'analyser un plan de document
et de produire un plan enrichi avec des informations structurelles pour guider la génération multi-agents.

Tu dois :
1. Identifier les dépendances entre sections (quelles sections dépendent d'autres pour la cohérence)
2. Détecter les zones à risque d'hallucination (sections nécessitant des données précises)
3. Définir le ton et le type de chaque section (analytique, narratif, descriptif, conclusif)
4. Estimer la longueur cible de chaque section
5. Formuler un system prompt global pour les Rédacteurs

Tu réponds UNIQUEMENT en JSON valide, sans commentaires ni texte autour du JSON."""

ARCHITECT_PROMPT_TEMPLATE = """═══ PLAN DU DOCUMENT ═══
{plan_text}

═══ OBJECTIF DU DOCUMENT ═══
{objective}

═══ CORPUS SOURCE (début) ═══
{corpus_preview}
═══ FIN DU CORPUS ═══

═══ INSTRUCTIONS ═══
Analyse ce plan et retourne un JSON structuré avec :

1. "sections" : liste d'objets avec :
   - "id" : identifiant de la section (ex: "s01", "s02")
   - "title" : titre de la section
   - "longueur_cible" : nombre de mots cible (integer)
   - "ton" : "analytique" | "narratif" | "descriptif" | "conclusif" | "introductif"
   - "type" : "introduction" | "fond" | "methodologie" | "resultats" | "discussion" | "conclusion" | "annexe"

2. "dependances" : dictionnaire {{section_id: [ids_des_sections_requises_avant]}}
   - Une section sans dépendance a une liste vide []
   - L'introduction n'a jamais de dépendance
   - La conclusion dépend de toutes les sections de fond
   - Les sections de fond peuvent dépendre les unes des autres si nécessaire

3. "zones_risque" : liste de {{"section_id": "...", "description": "..."}} pour les sections
   nécessitant une vérification factuelle renforcée

4. "system_prompt_global" : instruction système synthétisée pour les Rédacteurs,
   incluant le ton global, les consignes de style et les points d'attention

Retourne UNIQUEMENT le JSON, sans commentaires."""


class ArchitectAgent(BaseAgent):
    """Agent Architecte : planification et structuration des dépendances."""

    async def _execute(self, task: dict) -> AgentResult:
        plan = task.get("plan")
        corpus_text = task.get("corpus_text", "")
        objective = task.get("objective", "")

        if not plan:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="Aucun plan fourni à l'Architecte",
            )

        # Construire le texte du plan
        plan_text = self._format_plan(plan)

        # Limiter le corpus pour l'Architecte (vue globale)
        corpus_preview = corpus_text[:50000] if len(corpus_text) > 50000 else corpus_text

        prompt = ARCHITECT_PROMPT_TEMPLATE.format(
            plan_text=plan_text,
            objective=objective,
            corpus_preview=corpus_preview,
        )

        self._update_state("running", "Analyse du plan et des dépendances", 0.3)

        response = await self._call_provider(
            prompt=prompt,
            system_prompt=ARCHITECT_SYSTEM_PROMPT,
        )

        self._update_state("running", "Parsing du résultat", 0.8)

        # Parser le JSON de la réponse
        architecture = self._parse_architecture(response.content, plan)

        if architecture is None:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="Impossible de parser le résultat de l'Architecte",
                token_input=response.input_tokens,
                token_output=response.output_tokens,
            )

        return AgentResult(
            agent_name=self.name,
            success=True,
            structured_data=architecture,
            token_input=response.input_tokens,
            token_output=response.output_tokens,
        )

    def _build_system_prompt(self, task: dict) -> str:
        return ARCHITECT_SYSTEM_PROMPT

    def _format_plan(self, plan) -> str:
        """Formate le NormalizedPlan en texte lisible."""
        lines = []
        if hasattr(plan, "title") and plan.title:
            lines.append(f"Titre : {plan.title}")
        if hasattr(plan, "objective") and plan.objective:
            lines.append(f"Objectif : {plan.objective}")
        lines.append("")

        if hasattr(plan, "sections"):
            for section in plan.sections:
                indent = "  " * (section.level - 1)
                desc = f" — {section.description}" if section.description else ""
                lines.append(f"{indent}{section.id}. {section.title}{desc}")

        return "\n".join(lines)

    def _parse_architecture(self, content: str, plan) -> dict | None:
        """Parse le JSON retourné par le LLM avec fallback."""
        # Nettoyer le contenu
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
            # Tenter d'extraire le JSON du texte
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.error("Impossible de parser le JSON de l'Architecte")
                    return self._build_default_architecture(plan)
            else:
                return self._build_default_architecture(plan)

        # Valider et compléter la structure
        if "sections" not in data:
            data["sections"] = []
        if "dependances" not in data:
            data["dependances"] = {}
        if "zones_risque" not in data:
            data["zones_risque"] = []
        if "system_prompt_global" not in data:
            data["system_prompt_global"] = ""

        return data

    def _build_default_architecture(self, plan) -> dict:
        """Architecture par défaut si le parsing échoue."""
        sections = []
        dependances = {}
        all_fond_ids = []

        if hasattr(plan, "sections"):
            for i, section in enumerate(plan.sections):
                sid = section.id
                is_intro = i == 0
                is_conclusion = i == len(plan.sections) - 1 and len(plan.sections) > 2

                section_type = "fond"
                ton = "analytique"
                if is_intro:
                    section_type = "introduction"
                    ton = "introductif"
                elif is_conclusion:
                    section_type = "conclusion"
                    ton = "conclusif"
                else:
                    all_fond_ids.append(sid)

                sections.append({
                    "id": sid,
                    "title": section.title,
                    "longueur_cible": 500,
                    "ton": ton,
                    "type": section_type,
                })

                if is_intro:
                    dependances[sid] = []
                elif is_conclusion:
                    dependances[sid] = list(all_fond_ids)
                else:
                    dependances[sid] = []

        return {
            "sections": sections,
            "dependances": dependances,
            "zones_risque": [],
            "system_prompt_global": (
                "Rédige en français professionnel. Base-toi exclusivement sur le corpus fourni. "
                "Utilise le marqueur {{NEEDS_SOURCE}} pour toute information non sourcée."
            ),
        }
