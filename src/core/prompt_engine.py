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


REFINEMENT_PROMPT_TEMPLATE = """## Objectif du document
{objective}

## Section à raffiner
Titre : {section_title}
Niveau hiérarchique : {section_level}
{section_description}

## Consignes de longueur
{length_instruction}

## Contexte des sections précédentes
{previous_context}

## Corpus source pertinent
{corpus_content}

## Brouillon actuel à améliorer
{draft_content}
{extra_instruction}
## Instructions de raffinement
Améliore le brouillon ci-dessus en :
- Renforçant la précision et la richesse du contenu à partir du corpus source.
- Améliorant la structure, la clarté et la fluidité du texte.
- Corrigeant les erreurs factuelles, grammaticales ou stylistiques.
- Respectant la longueur cible.
- Conservant les éléments de qualité du brouillon.
Retourne uniquement la version améliorée, sans commentaires ni explications.
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
        corpus_chunks: list,
        previous_summaries: list[str],
        target_pages: Optional[float] = None,
        extra_instruction: str = "",
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
                source = getattr(chunk, "source_file", "inconnu")
                text = getattr(chunk, "text", str(chunk))
                corpus_parts.append(f"[Source {i + 1}: {source}]\n{text}")
            corpus_content = "\n\n---\n\n".join(corpus_parts)
        else:
            corpus_content = "Aucun corpus source fourni. Rédige à partir de tes connaissances générales."

        prompt = SECTION_PROMPT_TEMPLATE.format(
            objective=plan.objective or plan.title or "Document professionnel",
            section_title=section.title,
            section_level=section.level,
            section_description=description,
            length_instruction=length_instruction,
            previous_context=previous_context,
            corpus_content=corpus_content,
        )
        if extra_instruction:
            prompt += f"\n\n## Consigne supplémentaire\n{extra_instruction}\n"
        return prompt

    def build_refinement_prompt(
        self,
        section: PlanSection,
        plan: NormalizedPlan,
        draft_content: str,
        corpus_chunks: list,
        previous_summaries: list[str],
        target_pages: Optional[float] = None,
        extra_instruction: str = "",
    ) -> str:
        """Construit le prompt de raffinement pour une section existante."""
        description = ""
        if section.description:
            description = f"Description : {section.description}"

        if section.page_budget:
            length_instruction = f"Environ {section.page_budget} page(s) ({int(section.page_budget * 400)} tokens approximativement)."
        elif target_pages:
            length_instruction = f"Ajuste la longueur proportionnellement à la taille cible du document ({target_pages} pages)."
        else:
            length_instruction = "Longueur adaptée au contenu à couvrir."

        previous_context = "Aucune section précédente." if not previous_summaries else "\n".join(
            f"- {s}" for s in previous_summaries[-5:]
        )

        if corpus_chunks:
            corpus_parts = []
            for i, chunk in enumerate(corpus_chunks):
                source = getattr(chunk, "source_file", "inconnu")
                text = getattr(chunk, "text", str(chunk))
                corpus_parts.append(f"[Source {i + 1}: {source}]\n{text}")
            corpus_content = "\n\n---\n\n".join(corpus_parts)
        else:
            corpus_content = "Aucun corpus source fourni."

        extra_block = f"\n## Consigne supplémentaire\n{extra_instruction}" if extra_instruction else ""

        return REFINEMENT_PROMPT_TEMPLATE.format(
            objective=plan.objective or plan.title or "Document professionnel",
            section_title=section.title,
            section_level=section.level,
            section_description=description,
            length_instruction=length_instruction,
            previous_context=previous_context,
            corpus_content=corpus_content,
            draft_content=draft_content or "[Aucun brouillon disponible]",
            extra_instruction=extra_block,
        )

    def build_plan_generation_prompt(
        self,
        objective: str,
        target_pages: Optional[float] = None,
        corpus_digest: Optional[dict] = None,
    ) -> str:
        """Construit le prompt pour générer un plan automatiquement.

        Args:
            objective: Objectif du document.
            target_pages: Nombre de pages cible.
            corpus_digest: Dict produit par ``StructuredCorpus.get_corpus_digest()``
                avec les clés ``tier``, ``num_documents``, ``entries``, et
                optionnellement ``all_filenames``.
        """
        if corpus_digest and corpus_digest.get("entries"):
            corpus_section = self._format_corpus_digest(corpus_digest)
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

    @staticmethod
    def _format_corpus_digest(digest: dict) -> str:
        """Formate le digest du corpus selon le palier utilisé."""
        tier = digest["tier"]
        num_docs = digest["num_documents"]
        entries = digest["entries"]

        if tier == "full_excerpts":
            parts = []
            for i, entry in enumerate(entries, 1):
                parts.append(f"[Document {i} : {entry['source_file']}]\n{entry['text']}")
            corpus_text = "\n\n---\n\n".join(parts)
            return (
                f"\n## Extraits du corpus documentaire disponible\n"
                f"Voici un extrait représentatif de chaque document source "
                f"({num_docs} document(s)) :\n\n{corpus_text}\n\n"
            )

        elif tier == "first_sentences":
            lines = []
            for i, entry in enumerate(entries, 1):
                kw = entry.get("keywords", [])
                kw_str = f" [{', '.join(kw)}]" if kw else ""
                lines.append(f"{i}. {entry['source_file']} — {entry['text']}{kw_str}")
            listing = "\n".join(lines)
            return (
                f"\n## Documents disponibles ({num_docs} documents)\n\n"
                f"{listing}\n\n"
            )

        else:  # sampled
            # Liste de tous les fichiers avec mots-clés
            all_files_kw = digest.get("all_files_keywords", [])
            if all_files_kw:
                file_lines = []
                for fkw in all_files_kw:
                    kw = fkw.get("keywords", [])
                    kw_str = f" [{', '.join(kw)}]" if kw else ""
                    file_lines.append(f"- {fkw['source_file']}{kw_str}")
                files_listing = "\n".join(file_lines)
            else:
                all_filenames = digest.get("all_filenames", [])
                files_listing = ", ".join(all_filenames)

            parts = []
            for i, entry in enumerate(entries, 1):
                parts.append(f"[{entry['source_file']}]\n{entry['text']}")
            excerpts_text = "\n\n---\n\n".join(parts)
            return (
                f"\n## Corpus disponible ({num_docs} documents)\n\n"
                f"### Liste des sources et thématiques\n{files_listing}\n\n"
                f"### Extraits représentatifs (échantillon de {len(entries)} documents)\n\n"
                f"{excerpts_text}\n\n"
            )

    def build_summary_prompt(self, section_title: str, content: str) -> str:
        """Construit le prompt pour résumer une section (contexte pour les suivantes)."""
        return (
            f"Résume en 2-3 phrases le contenu principal de la section \"{section_title}\" "
            f"pour fournir du contexte aux sections suivantes du document :\n\n{content[:2000]}"
        )
