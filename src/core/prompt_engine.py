"""Génération et gestion des prompts pour le pipeline."""

import logging
from typing import Optional

from src.core.plan_parser import PlanSection, NormalizedPlan
from src.core.corpus_extractor import CorpusChunk

logger = logging.getLogger("orchestria")


SYSTEM_PROMPT_TEMPLATE = """Tu es un rédacteur professionnel expert. Tu rédiges des documents structurés, précis et de haute qualité.

Règles générales :
- Rédige en français sauf indication contraire.
- Utilise un style professionnel, clair et concis.
- Respecte strictement la structure et les consignes fournies.
- Base-toi exclusivement sur le corpus fourni pour les informations factuelles.
- Ne fabrique pas de données ou de statistiques.
{persistent_instructions}"""

SECTION_PROMPT_TEMPLATE = """## Objectif du document
{objective}

## Section à rédiger
Titre : {section_title}
Niveau hiérarchique : {section_level}
{section_description}

## Consignes de longueur
{length_instruction}

## Contexte des sections précédentes
{previous_context}

## Corpus source pertinent
{corpus_content}

## Instructions
Rédige le contenu de cette section en respectant les consignes ci-dessus.
Le texte doit être structuré, professionnel et directement exploitable dans un document final.
N'inclus pas le titre de la section dans ta réponse (il sera ajouté automatiquement).
"""

PLAN_GENERATION_PROMPT = """À partir de l'objectif suivant, génère un plan structuré détaillé pour un document professionnel.

## Objectif
{objective}

## Taille cible
{target_pages} pages environ.
{corpus_section}
## Instructions
- Propose un plan hiérarchique avec des sections numérotées (1. / 1.1 / 1.1.1).
- Chaque section doit avoir un titre clair et descriptif.
- Le plan doit être logique, progressif et couvrir l'ensemble du sujet.
- Adapte le nombre de sections à la taille cible demandée.
{corpus_instruction}- Retourne uniquement le plan, sans commentaires ni explications.
"""


class PromptEngine:
    """Génère les prompts pour chaque étape du pipeline."""

    def __init__(self, persistent_instructions: str = ""):
        self.persistent_instructions = persistent_instructions

    def build_system_prompt(self) -> str:
        """Construit le prompt système avec les instructions persistantes."""
        instructions = ""
        if self.persistent_instructions:
            instructions = f"\n\nInstructions spécifiques au projet :\n{self.persistent_instructions}"
        return SYSTEM_PROMPT_TEMPLATE.format(persistent_instructions=instructions)

    def build_section_prompt(
        self,
        section: PlanSection,
        plan: NormalizedPlan,
        corpus_chunks: list[CorpusChunk],
        previous_summaries: list[str],
        target_pages: Optional[float] = None,
    ) -> str:
        """Construit le prompt pour générer une section."""
        # Description de la section
        description = ""
        if section.description:
            description = f"Description : {section.description}"

        # Instruction de longueur
        if section.page_budget:
            length_instruction = f"Environ {section.page_budget} page(s) ({int(section.page_budget * 400)} tokens approximativement)."
        elif target_pages:
            length_instruction = f"Ajuste la longueur proportionnellement à la taille cible du document ({target_pages} pages)."
        else:
            length_instruction = "Longueur adaptée au contenu à couvrir."

        # Contexte des sections précédentes
        previous_context = "Aucune section précédente." if not previous_summaries else "\n".join(
            f"- {s}" for s in previous_summaries[-5:]  # Limiter aux 5 derniers résumés
        )

        # Corpus pertinent
        if corpus_chunks:
            corpus_parts = []
            for i, chunk in enumerate(corpus_chunks):
                corpus_parts.append(f"[Source {i + 1}: {chunk.source_file}]\n{chunk.text}")
            corpus_content = "\n\n---\n\n".join(corpus_parts)
        else:
            corpus_content = "Aucun corpus source fourni. Rédige à partir de tes connaissances générales."

        return SECTION_PROMPT_TEMPLATE.format(
            objective=plan.objective or plan.title or "Document professionnel",
            section_title=section.title,
            section_level=section.level,
            section_description=description,
            length_instruction=length_instruction,
            previous_context=previous_context,
            corpus_content=corpus_content,
        )

    def build_plan_generation_prompt(
        self,
        objective: str,
        target_pages: Optional[float] = None,
        corpus_digest: Optional[list[dict]] = None,
    ) -> str:
        """Construit le prompt pour générer un plan automatiquement.

        Args:
            objective: Objectif du document.
            target_pages: Nombre de pages cible.
            corpus_digest: Liste de dicts {"source_file", "excerpt"} produits
                par StructuredCorpus.get_corpus_digest().
        """
        if corpus_digest:
            parts = []
            for i, doc in enumerate(corpus_digest, 1):
                parts.append(f"[Document {i} : {doc['source_file']}]\n{doc['excerpt']}")
            corpus_text = "\n\n---\n\n".join(parts)
            corpus_section = (
                f"\n## Extraits du corpus documentaire disponible\n"
                f"Voici un extrait représentatif de chaque document source "
                f"({len(corpus_digest)} document(s)) :\n\n{corpus_text}\n\n"
            )
            corpus_instruction = (
                "- Tiens compte du contenu des documents fournis pour structurer le plan "
                "de manière pertinente par rapport aux informations disponibles.\n"
            )
        else:
            corpus_section = ""
            corpus_instruction = ""

        return PLAN_GENERATION_PROMPT.format(
            objective=objective,
            target_pages=target_pages or 10,
            corpus_section=corpus_section,
            corpus_instruction=corpus_instruction,
        )

    def build_summary_prompt(self, section_title: str, content: str) -> str:
        """Construit le prompt pour résumer une section (contexte pour les suivantes)."""
        return (
            f"Résume en 2-3 phrases le contenu principal de la section \"{section_title}\" "
            f"pour fournir du contexte aux sections suivantes du document :\n\n{content[:2000]}"
        )
