"""Génération et gestion des prompts pour le pipeline.

Phase 2.5 : ajout des garde-fous anti-hallucination et du marqueur {{NEEDS_SOURCE}}.
"""

import logging
from typing import Optional

from src.core.plan_parser import PlanSection, NormalizedPlan
from src.core.corpus_extractor import CorpusChunk

logger = logging.getLogger("orchestria")


# Phase 2.5 : Bloc anti-hallucination injecté dans chaque prompt de génération
ANTI_HALLUCINATION_BLOCK = """
═══ RÈGLES DE FIABILITÉ (NON NÉGOCIABLES) ═══

1. SOURCES EXCLUSIVES : Tu ne peux utiliser QUE les blocs de corpus
   fournis ci-dessous (entre les balises --- SOURCE --- et --- FIN SOURCE ---).
   Ne fabrique JAMAIS d'information absente du corpus.

2. MARQUEUR D'INSUFFISANCE : Si tu veux développer un point mais
   qu'aucun bloc de corpus ne le soutient, écris EXACTEMENT :
   {{NEEDS_SOURCE: [description du point à sourcer]}}
   Cela signale à l'utilisateur qu'il doit compléter le corpus.
   NE REMPLACE PAS ce marqueur par du contenu inventé.

3. ATTRIBUTION : Quand tu utilises une information d'un document source,
   cite-le par sa référence bibliographique complète au format APA
   (ex: Dupont, 2024) ou, à défaut, par son nom de fichier
   (ex: selon le document rapport_analyse.pdf). N'utilise JAMAIS de
   numéro de source comme [Source 1] ou [Source 5].

4. TRANSPARENCE : Si le corpus est insuffisant pour une section
   complète, écris une section plus courte plutôt que de compléter
   avec des informations non sourcées.
═══ FIN DES RÈGLES ═══
"""


SYSTEM_PROMPT_TEMPLATE = """Tu es un rédacteur professionnel expert. Tu rédiges des documents structurés, précis et de haute qualité.

Règles générales :
- Rédige en français sauf indication contraire.
- Utilise un style professionnel, clair et concis.
- Respecte strictement la structure et les consignes fournies.
- Base-toi exclusivement sur le corpus fourni pour les informations factuelles.
- Ne fabrique pas de données ou de statistiques.
{persistent_instructions}{anti_hallucination}"""

SECTION_PROMPT_TEMPLATE = """═══ OBJECTIF DU DOCUMENT ═══
{objective}

═══ SECTION À RÉDIGER ═══
Titre : {section_title}
Niveau hiérarchique : {section_level}
{section_description}

═══ CONSIGNES DE LONGUEUR ═══
{length_instruction}

═══ CONTEXTE DES SECTIONS PRÉCÉDENTES ═══
{previous_context}

═══ CORPUS SOURCE PERTINENT ═══
{corpus_content}

═══ INSTRUCTIONS ═══
Rédige le contenu de cette section en respectant les consignes ci-dessus.
Le texte doit être structuré, professionnel et directement exploitable dans un document final.
N'inclus pas le titre de la section dans ta réponse (il sera ajouté automatiquement).
N'utilise pas de titres Markdown (# ou ##) dans ta réponse. Si tu as besoin de sous-titres internes, utilise le format en gras : **Sous-titre**.
"""

PLAN_GENERATION_PROMPT = """À partir de l'objectif suivant, génère un plan structuré détaillé pour un document professionnel.

═══ OBJECTIF ═══
{objective}

═══ TAILLE CIBLE ═══
{target_pages} pages environ.
{corpus_section}
═══ INSTRUCTIONS ═══
- Propose un plan hiérarchique avec des sections numérotées (1. / 1.1 / 1.1.1).
- Chaque section doit avoir un titre clair et descriptif.
- Le plan doit être logique, progressif et couvrir l'ensemble du sujet.
- Adapte le nombre de sections à la taille cible demandée.
{corpus_instruction}- Retourne uniquement le plan, sans commentaires ni explications.
"""


REFINEMENT_PROMPT_TEMPLATE = """═══ OBJECTIF DU DOCUMENT ═══
{objective}

═══ SECTION À RAFFINER ═══
Titre : {section_title}
Niveau hiérarchique : {section_level}
{section_description}

═══ CONSIGNES DE LONGUEUR ═══
{length_instruction}

═══ CONTEXTE DES SECTIONS PRÉCÉDENTES ═══
{previous_context}

═══ CORPUS SOURCE PERTINENT ═══
{corpus_content}

═══ BROUILLON ACTUEL À AMÉLIORER ═══
{draft_content}
{extra_instruction}
═══ INSTRUCTIONS DE RAFFINEMENT ═══
Améliore le brouillon ci-dessus en :
- Renforçant la précision et la richesse du contenu à partir du corpus source.
- Améliorant la structure, la clarté et la fluidité du texte.
- Corrigeant les erreurs factuelles, grammaticales ou stylistiques.
- Respectant la longueur cible.
- Conservant les éléments de qualité du brouillon.
- N'utilisant pas de titres Markdown (# ou ##). Utilise des sous-titres en gras (**Sous-titre**) si nécessaire.
Retourne uniquement la version améliorée, sans commentaires ni explications.
"""


class PromptEngine:
    """Génère les prompts pour chaque étape du pipeline.

    Phase 2.5 : injection systématique du bloc anti-hallucination.
    """

    def __init__(self, persistent_instructions: str = "", anti_hallucination_enabled: bool = True):
        self.persistent_instructions = persistent_instructions
        self.anti_hallucination_enabled = anti_hallucination_enabled

    def build_system_prompt(self, has_corpus: bool = True) -> str:
        """Construit le prompt système avec instructions persistantes et garde-fous.

        Args:
            has_corpus: Si False, désactive le bloc anti-hallucination pour éviter
                la contradiction avec l'instruction d'utiliser les connaissances générales.
        """
        instructions = ""
        if self.persistent_instructions:
            instructions = f"\n\nInstructions spécifiques au projet :\n{self.persistent_instructions}"

        anti_hallucination = ""
        if self.anti_hallucination_enabled and has_corpus:
            anti_hallucination = f"\n{ANTI_HALLUCINATION_BLOCK}"

        return SYSTEM_PROMPT_TEMPLATE.format(
            persistent_instructions=instructions,
            anti_hallucination=anti_hallucination,
        )

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

        # Corpus pertinent (regroupé par document source)
        if corpus_chunks:
            corpus_content = self._format_corpus_chunks_grouped(corpus_chunks)
        else:
            corpus_content = (
                "Aucun corpus source disponible pour cette section. "
                "Signale chaque point nécessitant une source avec le marqueur "
                "{{NEEDS_SOURCE: [description du point]}}."
            )

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
            prompt += f"\n\n═══ CONSIGNE SUPPLÉMENTAIRE ═══\n{extra_instruction}\n"
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
            corpus_content = self._format_corpus_chunks_grouped(corpus_chunks)
        else:
            corpus_content = (
                "Aucun corpus source disponible pour cette section. "
                "Signale chaque point nécessitant une source avec le marqueur "
                "{{NEEDS_SOURCE: [description du point]}}."
            )

        extra_block = f"\n═══ CONSIGNE SUPPLÉMENTAIRE ═══\n{extra_instruction}" if extra_instruction else ""

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
                f"\n═══ EXTRAITS DU CORPUS DOCUMENTAIRE DISPONIBLE ═══\n"
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
                f"\n═══ DOCUMENTS DISPONIBLES ({num_docs} documents) ═══\n\n"
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
                f"\n═══ CORPUS DISPONIBLE ({num_docs} documents) ═══\n\n"
                f"─── Liste des sources et thématiques ───\n{files_listing}\n\n"
                f"─── Extraits représentatifs (échantillon de {len(entries)} documents) ───\n\n"
                f"{excerpts_text}\n\n"
            )

    def build_summary_prompt(self, section_title: str, content: str) -> str:
        """Construit le prompt pour résumer une section (contexte pour les suivantes)."""
        return (
            f"Résume en 2-3 phrases le contenu principal de la section \"{section_title}\" "
            f"pour fournir du contexte aux sections suivantes du document :\n\n{content[:2000]}"
        )

    @staticmethod
    def _get_chunk_attr(chunk, key, default=""):
        """Accède à un attribut d'un chunk, qu'il soit un dict ou un objet."""
        if isinstance(chunk, dict):
            return chunk.get(key, default)
        return getattr(chunk, key, default)

    @staticmethod
    def _format_corpus_chunks_grouped(corpus_chunks: list) -> str:
        """Regroupe les chunks par document source pour le prompt.

        Au lieu de numéroter chaque chunk individuellement ([Source 1], [Source 2]...),
        les regroupe par fichier source avec des références APA si disponibles.
        Supporte les chunks sous forme de dict ou d'objet (formats hétérogènes).
        """
        from collections import OrderedDict

        _get = PromptEngine._get_chunk_attr

        grouped = OrderedDict()
        for chunk in corpus_chunks:
            source = _get(chunk, "source_file", "inconnu")
            apa_ref = _get(chunk, "apa_reference", None)
            text = _get(chunk, "text", str(chunk))
            key = source
            if key not in grouped:
                grouped[key] = {"apa_reference": apa_ref, "extracts": []}
            grouped[key]["extracts"].append(text)
            # Prefer non-None apa_reference
            if apa_ref and not grouped[key]["apa_reference"]:
                grouped[key]["apa_reference"] = apa_ref

        parts = []
        for source_file, data in grouped.items():
            # Build document header with APA reference if available
            if data["apa_reference"]:
                header = f"[Document : {data['apa_reference']}]"
            else:
                header = f"[Document : {source_file}]"

            extracts = []
            for i, ext_text in enumerate(data["extracts"], 1):
                extracts.append(f"--- Extrait {i} ---\n{ext_text}")

            parts.append(f"{header}\n" + "\n".join(extracts))

        return "\n\n════════════════\n\n".join(parts)
