"""Agent Correcteur — raffinement ciblé des sections insuffisantes.

Phase 7 : corrige les sections identifiées par l'Évaluateur en utilisant
           le function calling (search_corpus, get_section, flag_unresolvable).
"""

import logging

from src.core.agent_framework import AgentResult, BaseAgent

logger = logging.getLogger("orchestria")

CORRECTOR_SYSTEM_PROMPT = """Tu es un correcteur documentaire expert. Ton rôle est de corriger et raffiner
une section de document identifiée comme insuffisante.

Tu disposes de trois outils que tu peux appeler en insérant un bloc JSON dans ta réponse :

1. search_corpus : recherche des passages dans le corpus source
   Usage : {{"tool": "search_corpus", "arguments": {{"query": "termes à chercher"}}}}

2. get_section : récupère le contenu d'une autre section
   Usage : {{"tool": "get_section", "arguments": {{"section_id": "s03"}}}}

3. flag_unresolvable : signale une correction impossible (info absente du corpus)
   Usage : {{"tool": "flag_unresolvable", "arguments": {{"section_id": "s03", "reason": "description"}}}}

Stratégie de correction :
1. Identifie les passages problématiques
2. Utilise search_corpus pour retrouver les passages sourceurs manquants
3. Utilise get_section pour vérifier la cohérence avec les sections voisines
4. Réécris les passages problématiques (pas de réécriture totale)
5. Si une information manque du corpus, utilise flag_unresolvable

Quand tu as terminé la correction, retourne le contenu corrigé complet de la section.
Ne retourne PAS d'appel de tool dans ta réponse finale — uniquement le texte corrigé."""

CORRECTOR_PROMPT_TEMPLATE = """═══ SECTION À CORRIGER ═══
Section : {section_id} — {section_title}

═══ CONTENU ACTUEL ═══
{section_content}

═══ RAPPORT DE VÉRIFICATION ═══
{verif_report}

═══ SCORE ET CRITÈRES DÉFAILLANTS ═══
{eval_info}

═══ RÉSUMÉ DES AUTRES SECTIONS ═══
{other_sections}

═══ INSTRUCTIONS ═══
Corrige cette section en :
1. Résolvant les problèmes identifiés par le Vérificateur
2. Améliorant les critères défaillants signalés par l'Évaluateur
3. Conservant les éléments de qualité existants
4. N'utilisant PAS de titres Markdown (# ou ##)

Si tu as besoin d'informations du corpus, utilise l'outil search_corpus.
Si tu as besoin de vérifier la cohérence avec une autre section, utilise get_section.
Si une correction est impossible faute d'information, utilise flag_unresolvable.

Retourne le contenu corrigé complet de la section."""


class CorrectorAgent(BaseAgent):
    """Agent Correcteur : raffinement ciblé avec function calling."""

    def __init__(self, *args, tool_dispatcher=None, max_tool_calls=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_dispatcher = tool_dispatcher
        self.max_tool_calls = max_tool_calls

    async def _execute(self, task: dict) -> AgentResult:
        section_id = task.get("section_id", "")
        section_title = task.get("section_title", "")
        section_content = task.get("section_content", "")
        verif_report = task.get("verif_report", {})
        eval_info = task.get("eval_info", "")
        other_sections = task.get("other_sections_summary", "")

        if not section_content:
            return AgentResult(
                agent_name=self.name,
                section_id=section_id,
                success=False,
                error=f"Aucun contenu à corriger pour {section_id}",
            )

        self._update_state("running", f"Correction de {section_id}", 0.1)

        # Formater le rapport de vérification
        verif_text = self._format_verif_report(verif_report)

        prompt = CORRECTOR_PROMPT_TEMPLATE.format(
            section_id=section_id,
            section_title=section_title,
            section_content=section_content,
            verif_report=verif_text,
            eval_info=eval_info,
            other_sections=other_sections or "Aucun résumé disponible.",
        )

        system_prompt = CORRECTOR_SYSTEM_PROMPT

        # Utiliser la boucle de function calling si dispatcher disponible
        if self.tool_dispatcher:
            self._update_state("running", f"Correction avec tools {section_id}", 0.3)
            corrected_content, total_input, total_output = (
                await self.tool_dispatcher.run_agent_with_tools(
                    agent=self,
                    initial_prompt=prompt,
                    system_prompt=system_prompt,
                    max_tool_calls=self.max_tool_calls,
                )
            )
        else:
            response = await self._call_provider(
                prompt=prompt,
                system_prompt=system_prompt,
            )
            corrected_content = response.content
            total_input = response.input_tokens
            total_output = response.output_tokens

        corrected_content = corrected_content.strip()

        if not corrected_content:
            return AgentResult(
                agent_name=self.name,
                section_id=section_id,
                success=False,
                error=f"Correcteur a produit un texte vide pour {section_id}",
                token_input=total_input,
                token_output=total_output,
            )

        self._update_state("done", f"Section {section_id} corrigée", 1.0)

        return AgentResult(
            agent_name=self.name,
            section_id=section_id,
            success=True,
            content=corrected_content,
            token_input=total_input,
            token_output=total_output,
        )

    def _build_system_prompt(self, task: dict) -> str:
        return CORRECTOR_SYSTEM_PROMPT

    def _format_verif_report(self, report: dict) -> str:
        """Formate le rapport de vérification en texte lisible."""
        if not report:
            return "Aucun rapport de vérification disponible."

        parts = [f"Verdict : {report.get('verdict', '?')}"]
        parts.append(f"Score cohérence : {report.get('score_coherence', 0):.2f}")

        problems = report.get("problemes", [])
        if problems:
            parts.append(f"\nProblèmes ({len(problems)}) :")
            for p in problems:
                parts.append(
                    f"  [{p.get('type', '?')}] {p.get('description', '')}"
                )
                if p.get("passage_incrimine"):
                    parts.append(f"    Passage : \"{p['passage_incrimine']}\"")

        suggestions = report.get("suggestions", [])
        if suggestions:
            parts.append("\nSuggestions :")
            for s in suggestions:
                parts.append(f"  - {s}")

        return "\n".join(parts)
