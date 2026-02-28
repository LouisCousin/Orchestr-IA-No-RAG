"""Agent Rédacteur — génération du contenu d'une section.

Phase 7 : génère le contenu Markdown d'une section en s'appuyant sur
           le corpus injecté en contexte et les consignes de l'Architecte.
"""

import logging

from src.core.agent_framework import AgentResult, BaseAgent

logger = logging.getLogger("orchestria")

WRITER_SYSTEM_PROMPT = """Tu es un rédacteur professionnel expert. Tu rédiges des sections de documents
structurés, précis et de haute qualité.

Règles :
- Rédige en français sauf indication contraire.
- Base-toi EXCLUSIVEMENT sur le corpus fourni pour toute information factuelle.
- Ne fabrique JAMAIS de données, statistiques ou citations.
- Si une information n'est pas dans le corpus, utilise le marqueur {{{{NEEDS_SOURCE: [description]}}}}
- N'inclus PAS le titre de la section dans ta réponse.
- N'utilise PAS de titres Markdown (# ou ##). Utilise des sous-titres en gras (**Sous-titre**).
- Style professionnel, clair et concis.

{global_instructions}"""

WRITER_PROMPT_TEMPLATE = """═══ SECTION À RÉDIGER ═══
Titre : {section_title}
Type : {section_type}
Ton attendu : {section_ton}
Longueur cible : ~{longueur_cible} mots

{prerequisite_context}

═══ CORPUS SOURCE ═══
{corpus_text}

═══ INSTRUCTIONS ═══
Rédige le contenu de cette section en respectant les consignes ci-dessus.
Le texte doit être structuré, professionnel et directement exploitable dans un document final.
Respecte la longueur cible de ~{longueur_cible} mots.
"""


class WriterAgent(BaseAgent):
    """Agent Rédacteur : génération du contenu d'une section."""

    async def _execute(self, task: dict) -> AgentResult:
        section_id = task.get("section_id", "")
        section_title = task.get("section_title", "")
        section_type = task.get("section_type", "fond")
        section_ton = task.get("section_ton", "analytique")
        longueur_cible = task.get("longueur_cible", 500)
        corpus_text = task.get("corpus_text", "")
        global_instructions = task.get("system_prompt_global", "")
        prerequisite_sections = task.get("prerequisite_sections", {})

        self._update_state("running", f"Rédaction de {section_id}: {section_title}", 0.1)

        # Construire le contexte des sections prérequises
        prerequisite_context = ""
        if prerequisite_sections:
            parts = ["═══ CONTEXTE DES SECTIONS PRÉCÉDENTES ═══"]
            for sid, summary in prerequisite_sections.items():
                # Limiter chaque résumé à ~500 tokens
                truncated = summary[:2000] if len(summary) > 2000 else summary
                parts.append(f"[{sid}] : {truncated}")
            prerequisite_context = "\n".join(parts)

        prompt = WRITER_PROMPT_TEMPLATE.format(
            section_title=section_title,
            section_type=section_type,
            section_ton=section_ton,
            longueur_cible=longueur_cible,
            corpus_text=corpus_text,
            prerequisite_context=prerequisite_context,
        )

        system_prompt = WRITER_SYSTEM_PROMPT.format(
            global_instructions=global_instructions,
        )

        self._update_state("running", f"Appel LLM pour {section_id}", 0.3)

        response = await self._call_provider(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        content = response.content.strip()

        if not content:
            return AgentResult(
                agent_name=self.name,
                section_id=section_id,
                success=False,
                error=f"Rédacteur a produit un texte vide pour {section_id}",
                token_input=response.input_tokens,
                token_output=response.output_tokens,
            )

        self._update_state("running", f"Section {section_id} rédigée", 1.0)

        return AgentResult(
            agent_name=self.name,
            section_id=section_id,
            success=True,
            content=content,
            token_input=response.input_tokens,
            token_output=response.output_tokens,
        )

    def _build_system_prompt(self, task: dict) -> str:
        return WRITER_SYSTEM_PROMPT.format(
            global_instructions=task.get("system_prompt_global", ""),
        )
