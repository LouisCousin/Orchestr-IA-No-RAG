"""Corpus Atlas — indexation structurée pour documents extrêmes (v4.1).

Gère les corpus de 200-400 pages (20-40M tokens) en construisant un
index structuré JSON (l'Atlas) à ~5% du volume original.

L'Atlas permet de sélectionner les Top-K documents pertinents par section,
réduisant le contexte de 20-40M à ~500K tokens par appel LLM.

Architecture :
  1. AtlasBuilder indexe chaque document via un appel LLM parallèle
  2. L'index croisé (CrossIndex) agrège entités, thèmes et contradictions
  3. Le scoring Top-K sélectionne les documents pertinents par section
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.utils.token_counter import count_tokens

logger = logging.getLogger("orchestria")


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class AtlasFact:
    """Fait extrait d'un document avec localisation."""
    text: str
    source_file: str
    page: Optional[int] = None
    confidence: float = 1.0


@dataclass
class AtlasDocumentEntry:
    """Fiche d'indexation d'un document dans l'Atlas."""
    source_file: str
    doc_index: int
    title: str = ""
    summary: str = ""
    themes: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    key_facts: list[AtlasFact] = field(default_factory=list)
    token_count: int = 0
    factual_density: float = 0.0  # facts / 1000 tokens

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "doc_index": self.doc_index,
            "title": self.title,
            "summary": self.summary,
            "themes": self.themes,
            "entities": self.entities,
            "key_facts": [
                {"text": f.text, "source_file": f.source_file,
                 "page": f.page, "confidence": f.confidence}
                for f in self.key_facts
            ],
            "token_count": self.token_count,
            "factual_density": self.factual_density,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AtlasDocumentEntry":
        facts = [
            AtlasFact(
                text=f["text"],
                source_file=f.get("source_file", ""),
                page=f.get("page"),
                confidence=f.get("confidence", 1.0),
            )
            for f in data.get("key_facts", [])
        ]
        return cls(
            source_file=data["source_file"],
            doc_index=data.get("doc_index", 0),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            themes=data.get("themes", []),
            entities=data.get("entities", []),
            key_facts=facts,
            token_count=data.get("token_count", 0),
            factual_density=data.get("factual_density", 0.0),
        )


@dataclass
class AtlasCrossIndex:
    """Index croisé inter-documents."""
    entity_to_docs: dict[str, list[str]] = field(default_factory=dict)
    theme_to_docs: dict[str, list[str]] = field(default_factory=dict)
    contradictions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entity_to_docs": self.entity_to_docs,
            "theme_to_docs": self.theme_to_docs,
            "contradictions": self.contradictions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AtlasCrossIndex":
        return cls(
            entity_to_docs=data.get("entity_to_docs", {}),
            theme_to_docs=data.get("theme_to_docs", {}),
            contradictions=data.get("contradictions", []),
        )


@dataclass
class CorpusAtlas:
    """Atlas complet du corpus — représentation structurée à ~5% du volume."""
    documents: list[AtlasDocumentEntry] = field(default_factory=list)
    cross_index: AtlasCrossIndex = field(default_factory=AtlasCrossIndex)
    total_corpus_tokens: int = 0
    atlas_tokens: int = 0
    compression_ratio: float = 0.0
    indexation_model: str = ""
    indexation_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "documents": [d.to_dict() for d in self.documents],
            "cross_index": self.cross_index.to_dict(),
            "total_corpus_tokens": self.total_corpus_tokens,
            "atlas_tokens": self.atlas_tokens,
            "compression_ratio": round(self.compression_ratio, 2),
            "indexation_model": self.indexation_model,
            "indexation_cost_usd": round(self.indexation_cost_usd, 6),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CorpusAtlas":
        docs = [AtlasDocumentEntry.from_dict(d) for d in data.get("documents", [])]
        cross = AtlasCrossIndex.from_dict(data.get("cross_index", {}))
        return cls(
            documents=docs,
            cross_index=cross,
            total_corpus_tokens=data.get("total_corpus_tokens", 0),
            atlas_tokens=data.get("atlas_tokens", 0),
            compression_ratio=data.get("compression_ratio", 0.0),
            indexation_model=data.get("indexation_model", ""),
            indexation_cost_usd=data.get("indexation_cost_usd", 0.0),
        )

    def get_doc_by_source(self, source_file: str) -> Optional[AtlasDocumentEntry]:
        """Retourne l'entrée Atlas d'un document par nom de fichier."""
        for doc in self.documents:
            if doc.source_file == source_file:
                return doc
        return None

    def format_for_prompt(self, max_tokens: int = 200_000) -> str:
        """Formate l'Atlas pour injection dans un prompt LLM."""
        parts = ["═══ CORPUS ATLAS ═══"]
        parts.append(f"Corpus : {len(self.documents)} documents, "
                     f"~{self.total_corpus_tokens:,} tokens total")
        parts.append("")

        for doc in self.documents:
            doc_part = [f"── [{doc.source_file}] ──"]
            if doc.title:
                doc_part.append(f"  Titre : {doc.title}")
            doc_part.append(f"  Résumé : {doc.summary}")
            if doc.themes:
                doc_part.append(f"  Thèmes : {', '.join(doc.themes)}")
            if doc.entities:
                doc_part.append(f"  Entités : {', '.join(doc.entities[:15])}")
            if doc.key_facts:
                doc_part.append(f"  Faits clés ({len(doc.key_facts)}) :")
                for fact in doc.key_facts[:10]:
                    doc_part.append(f"    - {fact.text}")
            parts.append("\n".join(doc_part))

        if self.cross_index.contradictions:
            parts.append("\n═══ CONTRADICTIONS DÉTECTÉES ═══")
            for c in self.cross_index.contradictions:
                parts.append(f"  ⚠ {c.get('description', '')}")

        result = "\n\n".join(parts)

        # Tronquer si dépasse le budget
        if count_tokens(result) > max_tokens:
            result = result[:max_tokens * 4]  # ~4 chars per token
        return result


# ── Prompt d'indexation ──────────────────────────────────────────────────────

INDEXATION_SYSTEM_PROMPT = """Tu es un indexeur documentaire expert. Tu analyses un document et produis une fiche structurée JSON.

Règles :
- Extrais les informations factuelles, pas les opinions vagues.
- Les entités sont des noms propres (personnes, organisations, lieux, produits).
- Les thèmes sont des sujets conceptuels (2-5 mots chacun).
- Les faits clés sont des affirmations vérifiables avec données précises.
- Le résumé doit capturer l'essence du document en 3-5 phrases.
- Réponds UNIQUEMENT en JSON valide, sans commentaires."""

INDEXATION_PROMPT_TEMPLATE = """Analyse ce document et produis une fiche d'indexation JSON :

═══ DOCUMENT ═══
Fichier : {source_file}
Tokens : ~{token_count}

{document_text}

═══ FORMAT ATTENDU ═══
{{
  "title": "Titre déduit du document",
  "summary": "Résumé factuel en 3-5 phrases",
  "themes": ["thème1", "thème2", ...],
  "entities": ["entité1", "entité2", ...],
  "key_facts": [
    {{"text": "fait vérifiable", "page": null}},
    ...
  ]
}}

Retourne UNIQUEMENT le JSON."""


# ── AtlasBuilder ─────────────────────────────────────────────────────────────


class AtlasBuilder:
    """Construit un CorpusAtlas à partir d'un StructuredCorpus.

    Chaque document est indexé via un appel LLM parallèle. L'index croisé
    est construit localement sans appel LLM supplémentaire.
    """

    def __init__(
        self,
        provider,
        model: str = "gemini-3-flash-preview",
        max_concurrent: int = 4,
        max_doc_tokens_for_indexation: int = 800_000,
        config: Optional[dict] = None,
    ):
        self.provider = provider
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_doc_tokens = max_doc_tokens_for_indexation
        self._config = config or {}
        self._cost_tracker = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def set_cost_tracker(self, tracker) -> None:
        """Injecte un CostTracker pour comptabiliser les coûts d'indexation."""
        self._cost_tracker = tracker

    async def build_atlas(self, corpus) -> CorpusAtlas:
        """Construit l'Atlas complet du corpus.

        Args:
            corpus: StructuredCorpus avec chunks regroupés par document.

        Returns:
            CorpusAtlas avec fiches par document et index croisé.
        """
        atlas = CorpusAtlas(indexation_model=self.model)

        # Regrouper les chunks par document source
        docs_by_source = self._group_chunks_by_source(corpus)
        atlas.total_corpus_tokens = sum(
            c.token_estimate for c in corpus.chunks
        ) if hasattr(corpus, "chunks") else 0

        logger.info(
            f"[Atlas] Indexation de {len(docs_by_source)} documents "
            f"(~{atlas.total_corpus_tokens:,} tokens)"
        )

        # Indexer chaque document en parallèle (avec sémaphore)
        sem = asyncio.Semaphore(self.max_concurrent)
        tasks = []
        for doc_idx, (source_file, chunks) in enumerate(docs_by_source.items()):
            tasks.append(
                self._index_document(sem, source_file, chunks, doc_idx)
            )

        entries = await asyncio.gather(*tasks, return_exceptions=True)

        for entry in entries:
            if isinstance(entry, Exception):
                logger.error(f"[Atlas] Erreur indexation : {entry}")
                continue
            if entry is not None:
                atlas.documents.append(entry)

        # Construire l'index croisé (local, pas d'appel LLM)
        atlas.cross_index = self._build_cross_index(atlas.documents)

        # Calculer les stats
        atlas_text = atlas.format_for_prompt()
        atlas.atlas_tokens = count_tokens(atlas_text)
        if atlas.total_corpus_tokens > 0:
            atlas.compression_ratio = (
                atlas.total_corpus_tokens / max(atlas.atlas_tokens, 1)
            )

        # Coût d'indexation
        if self._cost_tracker:
            atlas.indexation_cost_usd = self._cost_tracker.calculate_cost(
                provider="google",
                model=self.model,
                input_tokens=self._total_input_tokens,
                output_tokens=self._total_output_tokens,
            )

        logger.info(
            f"[Atlas] Atlas construit : {len(atlas.documents)} fiches, "
            f"~{atlas.atlas_tokens:,} tokens atlas "
            f"(ratio {atlas.compression_ratio:.1f}:1)"
        )

        return atlas

    async def _index_document(
        self,
        sem: asyncio.Semaphore,
        source_file: str,
        chunks: list,
        doc_index: int,
    ) -> Optional[AtlasDocumentEntry]:
        """Indexe un seul document via un appel LLM."""
        async with sem:
            # Assembler le texte du document
            doc_text = "\n\n".join(
                getattr(c, "text", "") for c in chunks
            )
            doc_tokens = sum(
                getattr(c, "token_estimate", 0) for c in chunks
            )

            # Tronquer si le document dépasse la limite d'indexation
            if doc_tokens > self.max_doc_tokens:
                char_limit = self.max_doc_tokens * 4  # ~4 chars/token
                doc_text = doc_text[:char_limit]
                doc_tokens = self.max_doc_tokens

            prompt = INDEXATION_PROMPT_TEMPLATE.format(
                source_file=source_file,
                token_count=doc_tokens,
                document_text=doc_text,
            )

            try:
                response = await self.provider.generate_async(
                    prompt=prompt,
                    system_prompt=INDEXATION_SYSTEM_PROMPT,
                    model=self.model,
                    temperature=0.1,
                    max_tokens=2048,
                )

                self._total_input_tokens += response.input_tokens
                self._total_output_tokens += response.output_tokens

                # Tracker le coût si disponible
                if self._cost_tracker:
                    self._cost_tracker.track_atlas_indexation(
                        source_file=source_file,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        model=self.model,
                        provider="google",
                    )

                # Parser le JSON
                entry = self._parse_indexation_response(
                    response.content, source_file, doc_index, doc_tokens
                )
                return entry

            except Exception as e:
                logger.error(
                    f"[Atlas] Échec indexation {source_file}: {e}"
                )
                # Fallback : fiche minimale
                return AtlasDocumentEntry(
                    source_file=source_file,
                    doc_index=doc_index,
                    title=source_file,
                    summary=f"[Indexation échouée : {e}]",
                    token_count=doc_tokens,
                )

    def _parse_indexation_response(
        self,
        content: str,
        source_file: str,
        doc_index: int,
        doc_tokens: int,
    ) -> AtlasDocumentEntry:
        """Parse la réponse JSON du LLM en AtlasDocumentEntry."""
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
            # Tenter d'extraire le JSON
            data = None
            for i, ch in enumerate(content):
                if ch == '{':
                    try:
                        data = json.loads(content[i:])
                        break
                    except json.JSONDecodeError:
                        continue
            if data is None:
                logger.warning(
                    f"[Atlas] JSON invalide pour {source_file}, fiche minimale"
                )
                return AtlasDocumentEntry(
                    source_file=source_file,
                    doc_index=doc_index,
                    title=source_file,
                    summary="[Parsing échoué]",
                    token_count=doc_tokens,
                )

        # Construire les faits
        key_facts = []
        for fact_data in data.get("key_facts", []):
            if isinstance(fact_data, str):
                key_facts.append(AtlasFact(text=fact_data, source_file=source_file))
            elif isinstance(fact_data, dict):
                key_facts.append(AtlasFact(
                    text=fact_data.get("text", ""),
                    source_file=source_file,
                    page=fact_data.get("page"),
                    confidence=fact_data.get("confidence", 1.0),
                ))

        # Densité factuelle
        factual_density = (len(key_facts) / max(doc_tokens / 1000, 1))

        return AtlasDocumentEntry(
            source_file=source_file,
            doc_index=doc_index,
            title=data.get("title", source_file),
            summary=data.get("summary", ""),
            themes=data.get("themes", []),
            entities=data.get("entities", []),
            key_facts=key_facts,
            token_count=doc_tokens,
            factual_density=round(factual_density, 3),
        )

    def _build_cross_index(
        self, documents: list[AtlasDocumentEntry]
    ) -> AtlasCrossIndex:
        """Construit l'index croisé sans appel LLM."""
        cross = AtlasCrossIndex()

        for doc in documents:
            for entity in doc.entities:
                entity_lower = entity.lower()
                cross.entity_to_docs.setdefault(entity_lower, [])
                if doc.source_file not in cross.entity_to_docs[entity_lower]:
                    cross.entity_to_docs[entity_lower].append(doc.source_file)

            for theme in doc.themes:
                theme_lower = theme.lower()
                cross.theme_to_docs.setdefault(theme_lower, [])
                if doc.source_file not in cross.theme_to_docs[theme_lower]:
                    cross.theme_to_docs[theme_lower].append(doc.source_file)

        # Détection de contradictions basique : même entité dans 2+ docs avec
        # des faits divergents (heuristique simple basée sur les faits clés)
        self._detect_contradictions(documents, cross)

        return cross

    def _detect_contradictions(
        self,
        documents: list[AtlasDocumentEntry],
        cross: AtlasCrossIndex,
    ) -> None:
        """Détection heuristique de contradictions entre documents.

        Cherche des entités communes dont les faits mentionnent des valeurs
        numériques différentes (dates, montants, pourcentages).
        """
        import re

        # Regrouper les faits par entité
        entity_facts: dict[str, list[tuple[str, str]]] = {}
        for doc in documents:
            for entity in doc.entities:
                entity_lower = entity.lower()
                for fact in doc.key_facts:
                    if entity_lower in fact.text.lower():
                        entity_facts.setdefault(entity_lower, []).append(
                            (doc.source_file, fact.text)
                        )

        # Chercher des divergences numériques
        num_pattern = re.compile(r'\d+[\.,]?\d*\s*[%€$M]?')
        for entity, facts in entity_facts.items():
            if len(facts) < 2:
                continue

            # Extraire les nombres de chaque fait
            doc_numbers: dict[str, set[str]] = {}
            for source, text in facts:
                numbers = set(num_pattern.findall(text))
                if numbers:
                    doc_numbers.setdefault(source, set()).update(numbers)

            sources = list(doc_numbers.keys())
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    nums_i = doc_numbers[sources[i]]
                    nums_j = doc_numbers[sources[j]]
                    if nums_i and nums_j and nums_i != nums_j:
                        cross.contradictions.append({
                            "entity": entity,
                            "doc_a": sources[i],
                            "doc_b": sources[j],
                            "description": (
                                f"Données divergentes sur '{entity}' : "
                                f"{sources[i]} mentionne {nums_i}, "
                                f"{sources[j]} mentionne {nums_j}"
                            ),
                        })

    @staticmethod
    def _group_chunks_by_source(corpus) -> dict[str, list]:
        """Regroupe les chunks par fichier source."""
        docs: dict[str, list] = {}
        if hasattr(corpus, "chunks"):
            for chunk in corpus.chunks:
                source = getattr(chunk, "source_file", "unknown")
                docs.setdefault(source, []).append(chunk)
        return docs

    # ── Top-K Selection ──────────────────────────────────────────────────────

    @staticmethod
    def select_top_k_documents(
        atlas: CorpusAtlas,
        section_title: str,
        section_themes: list[str],
        section_entities: list[str],
        k: int = 5,
        weights: Optional[dict] = None,
    ) -> list[str]:
        """Sélectionne les K documents les plus pertinents pour une section.

        Scoring multi-critères :
          - thematic (0.4)   : chevauchement de thèmes
          - entity (0.3)     : chevauchement d'entités
          - factual (0.2)    : densité factuelle du document
          - contradiction (0.1) : bonus si le document est impliqué dans une contradiction

        Args:
            atlas: CorpusAtlas construit.
            section_title: Titre de la section à rédiger.
            section_themes: Thèmes attendus pour cette section.
            section_entities: Entités attendues.
            k: Nombre de documents à sélectionner.
            weights: Pondérations personnalisées.

        Returns:
            Liste des source_file des K documents les plus pertinents.
        """
        if not atlas.documents:
            return []

        w = weights or {
            "thematic": 0.4,
            "entity": 0.3,
            "factual": 0.2,
            "contradiction": 0.1,
        }

        section_title_lower = section_title.lower()
        section_themes_lower = {t.lower() for t in section_themes}
        section_entities_lower = {e.lower() for e in section_entities}

        # Docs impliqués dans des contradictions
        contradiction_docs = set()
        for c in atlas.cross_index.contradictions:
            contradiction_docs.add(c.get("doc_a", ""))
            contradiction_docs.add(c.get("doc_b", ""))

        # Max factual density pour normalisation
        max_density = max(
            (d.factual_density for d in atlas.documents), default=1.0
        ) or 1.0

        scores: list[tuple[str, float]] = []
        for doc in atlas.documents:
            # Score thématique
            doc_themes_lower = {t.lower() for t in doc.themes}
            if section_themes_lower:
                theme_overlap = len(
                    section_themes_lower & doc_themes_lower
                ) / len(section_themes_lower)
            else:
                # Fallback : chercher les thèmes du doc dans le titre de section
                theme_overlap = sum(
                    1 for t in doc.themes if t.lower() in section_title_lower
                ) / max(len(doc.themes), 1)

            # Score entités
            doc_entities_lower = {e.lower() for e in doc.entities}
            if section_entities_lower:
                entity_overlap = len(
                    section_entities_lower & doc_entities_lower
                ) / len(section_entities_lower)
            else:
                entity_overlap = 0.0

            # Score densité factuelle (normalisé)
            factual_score = doc.factual_density / max_density

            # Bonus contradiction
            contradiction_bonus = 1.0 if doc.source_file in contradiction_docs else 0.0

            total_score = (
                w["thematic"] * theme_overlap
                + w["entity"] * entity_overlap
                + w["factual"] * factual_score
                + w["contradiction"] * contradiction_bonus
            )

            scores.append((doc.source_file, total_score))

        # Trier par score décroissant
        scores.sort(key=lambda x: -x[1])
        return [source for source, _ in scores[:k]]
