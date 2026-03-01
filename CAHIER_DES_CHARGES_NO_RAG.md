# CAHIER DES CHARGES — Orchestr'IA v4.0 : Architecture Full Context Natif

**Projet** : Orchestr'IA
**Version** : 4.0 — Suppression native du RAG / Full Long-Context
**Date** : 2026-03-01
**Auteur** : Analyse architecturale approfondie (Claude Sonnet 4.6)
**Branche cible** : `claude/update-specifications-PyOrT`
**Périmètre** : Réécriture partielle du pipeline core, suppression complète du RAG, activation conditionnelle du Long Context Cache

---

## RÉSUMÉ EXÉCUTIF

Ce cahier des charges définit la transformation d'Orchestr'IA d'une architecture **RAG-first** (ChromaDB + embeddings locaux + reranking) vers une architecture **Full Context native**. Le système doit désormais injecter l'intégralité du corpus dans la fenêtre de contexte du LLM, en utilisant le **Context Caching de Gemini** comme mécanisme principal d'optimisation des coûts, et une **compression sémantique** comme filet de sécurité pour les corpus dépassant les limites physiques de la fenêtre.

**Deux seuils déclencheurs** :
- **≥ 650 000 tokens en Input** → Activation obligatoire du mode Full Context Cache
- **≥ 50 000 tokens en Output estimé** → Activation obligatoire du mode Full Context Cache (même pour un petit corpus)

---

## 1. ANALYSE DE L'EXISTANT

### 1.1 État Actuel du Code

L'analyse complète de la base de code (`src/core/`, 75+ fichiers) révèle l'état suivant :

**Ce qui existe déjà (à conserver et adapter) :**

| Module | Fichier | Rôle actuel | Action |
|--------|---------|-------------|--------|
| GeminiCacheManager | `core/gemini_cache_manager.py` | Phase 5 — caching optionnel | **Promouvoir en module central** |
| CostTracker | `core/cost_tracker.py` | Tracking coûts avec 4 cas | **Enrichir** (cas cache storage) |
| MultiAgentOrchestrator | `core/multi_agent_orchestrator.py` | DAG Phase 7 + Phase 8 cache | **Adapter** (supprimer dépendance RAG) |
| CorpusExtractor | `core/corpus_extractor.py` | Parsing structuré du corpus | **Conserver** |
| TokenCounter | `utils/token_counter.py` | Estimation tokens | **Conserver et enrichir** |
| ProjectState | `core/orchestrator.py` | État complet du projet | **Étendre** (strategy_mode, cache stats) |
| CheckpointManager | `core/checkpoint_manager.py` | HITL | **Conserver** |

**Ce qui doit être supprimé :**

| Module | Fichier | Raison |
|--------|---------|--------|
| RAGEngine | `core/rag_engine.py` | Remplacé par ContextManager |
| LocalEmbedder | `core/local_embedder.py` | Dépendance inutile sans RAG |
| Reranker | `core/reranker.py` | Inutile sans candidats vectoriels |
| SemanticChunker | `core/semantic_chunker.py` | Chunking RAG — remplacé par CompressorSémantique |
| PlanCorpusLinker | `core/plan_corpus_linker.py` | Liait plan à chunks vectoriels |

**Ce qui doit être créé :**

| Module | Fichier cible | Rôle |
|--------|---------------|------|
| StrategySelector | `core/strategy_selector.py` | Décision dynamique : standard vs full-context-cache |
| ContextManager | `core/context_manager.py` | Remplace RAGEngine ; gère le contexte complet |
| SemanticCompressor | `core/semantic_compressor.py` | Compression corpus >limite fenêtre |
| CacheHeartbeat | `core/cache_heartbeat.py` | Renouvellement TTL automatique |

### 1.2 Dépendances à Supprimer

**requirements.txt — lignes à retirer :**
```
chromadb>=0.5.0
fastembed>=0.2.0
sentence-transformers>=3.0
```

**Impact estimé** : réduction de ~4 Go de dépendances (modèles ONNX, index HNSW, torch).

### 1.3 Contraintes Techniques Confirmées (Gemini API)

D'après la configuration existante du projet (`config/model_pricing.yaml`) et la documentation Gemini (mars 2026) :

| Contrainte | Modèle | Valeur | Impact |
|-----------|--------|--------|--------|
| Minimum tokens pour caching | `gemini-3.1-pro-preview` | **2 048 tokens** | Pratiquement toujours dépassé (contrainte C4 déjà dans le code) |
| Minimum tokens pour caching | `gemini-3-flash-preview` | **2 048 tokens** | Identique au Pro |
| Fenêtre contexte | `gemini-3.1-pro-preview` | **1 000 000 tokens** | Limite physique absolue |
| Fenêtre contexte | `gemini-3-flash-preview` | **1 000 000 tokens** | Identique au Pro |
| **Max output par appel** | Les deux | **65 536 tokens** | ⚠️ Critique : un doc >65K tokens = obligatoirement multi-passes |
| Coût tokens cachés | `gemini-3.1-pro-preview` | **$0.20/M** (vs $2.00 standard) = **-90%** | Objectif économique principal |
| Coût tokens cachés | `gemini-3-flash-preview` | **$0.05/M** (vs $0.50 standard) = **-90%** | Option économique pour compression |
| Coût stockage cache | `gemini-3.1-pro-preview` | **$0.50 / heure / M tokens** | À monitorer via CostTracker |
| Coût stockage cache | `gemini-3-flash-preview` | **$1.00 / heure / M tokens** | Plus cher à stocker qu'à lire |
| TTL par défaut | Les deux | **1 heure** | À prolonger via CacheHeartbeat |
| Long context repricing | `gemini-3.1-pro-preview` | **>200K tokens** → $4.00/M input | Contrainte C6 déjà dans gemini_cache_manager.py |
| Long context repricing | `gemini-3-flash-preview` | **Aucun** (pas de repricing Flash) | Avantage Flash pour les gros corpus |

> **Conséquence directe du max_output_tokens = 65 536** : pour un document estimé à 50K+ tokens, la génération est obligatoirement multi-passes (une section à la fois). C'est exactement pourquoi le seuil output de 50K tokens déclenche le mode cache : il faut ancrer le contexte global pour éviter le drift entre passes.

---

## 2. OBJECTIFS ET PÉRIMÈTRE

### 2.1 Objectif Principal

Transformer Orchestr'IA en un système **nativement Full Context** où :

1. **Le corpus n'est jamais découpé en vecteurs** pour la recherche de similarité
2. **Chaque agent** a accès à l'intégralité du corpus via un cache contexte
3. **La décision de stratégie** (simple injection vs cache) est automatique et basée sur des seuils mesurables
4. **La compression sémantique** est le seul fallback si le corpus dépasse la fenêtre physique

### 2.2 Périmètre Fonctionnel

**IN SCOPE :**
- Suppression complète de ChromaDB, fastembed, sentence-transformers
- Création du StrategySelector avec logique à double seuil (650K / 50K)
- Création du ContextManager (remplace RAGEngine)
- Création du SemanticCompressor (fallback fenêtre dépassée)
- Création du CacheHeartbeat (TTL management)
- Adaptation des 5 agents (Architect, Writer, Verifier, Evaluator, Corrector)
- Mise à jour config.yaml et model_pricing.yaml
- Adaptation du CostTracker (cache storage tracking)
- Adaptation du ProjectState (strategy_mode, cache lifecycle)
- Mise à jour des tests unitaires et d'intégration

**OUT OF SCOPE (traité par CAHIER_DES_CHARGES_BUGS.md) :**
- Correction des bugs concurrentiels existants (B01–B36)
- Refactoring de l'UI Streamlit
- Export DOCX

---

## 3. SPÉCIFICATIONS FONCTIONNELLES

### 3.1 Règle de Décision Stratégique (StrategySelector)

Le système doit appliquer la logique de décision suivante, à **deux moments clés** du pipeline :

#### Moment 1 : Post-Acquisition du Corpus (Input Check)

```
ENTRÉE : total_corpus_tokens (calculé par TokenCounter sur le corpus brut)

SI total_corpus_tokens >= 650 000 :
    → strategy = "HIGH_VOLUME_CACHE"
    → Raison : corpus trop volumineux pour injection directe
    → Action : Créer le cache Gemini immédiatement, avant la planification

SINON :
    → strategy = "STANDARD" (provisoire — peut évoluer après planification)
    → Action : Corpus conservé en mémoire pour injection directe
```

#### Moment 2 : Post-Planification (Output Check)

```
ENTRÉE : estimated_output_tokens = sum(section.target_word_count * 1.35) pour toutes les sections du plan

SI estimated_output_tokens >= 50 000 :
    → strategy = "HIGH_VOLUME_CACHE" (forçage)
    → Raison : Génération longue = risque de drift sémantique si pas de contexte ancré
    → Action : Créer le cache Gemini MÊME SI le corpus est petit (<650K tokens)
              Le cache contient : corpus + system_prompt global + plan complet détaillé

SI estimated_output_tokens < 50 000 ET strategy == "STANDARD" :
    → strategy reste "STANDARD"
    → Action : Injection directe du corpus dans chaque appel agent
```

#### Règle de Priorité

```
HIGH_VOLUME_CACHE > STANDARD
Une fois HIGH_VOLUME_CACHE activé, il ne peut pas être rétrogradé à STANDARD.
```

### 3.2 Comportement par Stratégie

#### Mode STANDARD (< 650K tokens input ET < 50K tokens output)

- Le corpus formaté en XML structuré est injecté directement dans le `system_instruction` de chaque appel LLM
- Aucun cache explicite créé (le caching implicite Gemini 3.x peut s'appliquer automatiquement selon l'historique de requêtes)
- Aucune dépendance à ChromaDB ou embeddings
- Compatible avec tous les providers (OpenAI, Anthropic, Gemini)

#### Mode HIGH_VOLUME_CACHE (≥ 650K tokens input OU ≥ 50K tokens output)

- Création d'un cache Gemini explicite contenant le corpus XML structuré + `system_instruction` globale
- Tous les agents reçoivent uniquement le `cache_name` — le corpus n'est jamais réenvoyé
- Heartbeat automatique pour maintenir le TTL pendant toute la génération
- Suppression explicite du cache en fin de génération (ou sur crash via `try...finally`)
- **Provider imposé** : Gemini (seul provider supportant le context caching explicite)

> **Note sur Anthropic** : Claude supporte un extended context (200K tokens) mais pas le caching explicite équivalent à Gemini. En mode HIGH_VOLUME_CACHE, Gemini est le provider par défaut. OpenAI est disponible en fallback pour les agents ne nécessitant pas l'accès corpus (ex: Corrector en mode révision stylistique).

### 3.3 Mode de Compression Sémantique (Fallback Extrême)

**Déclencheur** : `total_corpus_tokens > 900 000` (marge de sécurité avant la limite physique de 1M)

La compression sémantique est un pipeline de réduction de densité informationnelle qui permet de rester en mode Full Context même pour des corpus géants.

#### Pipeline de Compression (3 niveaux, appliqués séquentiellement jusqu'à atteindre la cible)

**Niveau 1 — Déduplication et élagage (cible : -20% à -30%)**
- Suppression des doublons textuels exacts (hash SHA-256, déjà partiellement dans CorpusDeduplicator)
- Suppression des passages redondants inter-documents via similarité cosine légère (modèle local ONNX minimal, NON ChromaDB)
- Suppression des blocs purement décoratifs (en-têtes de page, mentions légales répétées, tables des matières de source)

**Niveau 2 — Extractivité (cible : -40% à -60%)**
- Extraction des faits et affirmations clés en format liste structurée JSON :
  ```json
  {
    "doc_id": "001",
    "titre": "Rapport XYZ",
    "faits_clés": [
      "La croissance du marché est de 12% en 2025.",
      "Le principal frein identifié est le coût d'adoption.",
      ...
    ],
    "chiffres_clés": {"marché": "12%", "investissement": "3M€"},
    "citations_importantes": ["..."]
  }
  ```
- Techniquement : appel LLM léger (Gemini Flash ou Haiku) sur chaque document, prompt d'extraction de faits

**Niveau 3 — Compression Télégraphique (cible : -60% à -75%)**
- Application de la **Telegraphic Semantic Compression (TSC)** : suppression de la structure grammaticale tout en préservant l'information factuelle
- Exemple : "Le rapport de l'année 2024 indique que les ventes ont augmenté de manière significative de 15%" → "Ventes +15% (rapport 2024)"
- Tous les chiffres, entités nommées, dates et faits sont préservés
- Aucune paraphrase — uniquement compression de la forme linguistique

#### Métriques de Succès

```
OBJECTIF : corpus_compressé_tokens <= 800 000
MESURES :
  - taux_compression = (tokens_avant - tokens_après) / tokens_avant
  - ratio_preservation_faits (évalué sur échantillon par LLM) >= 0.90
  - couverture_thématique (% de thèmes du plan couverts) >= 0.95
```

Si après les 3 niveaux le corpus dépasse encore 900K tokens, le système **avertit l'utilisateur** via l'UI et propose une sélection manuelle de documents à exclure.

---

## 4. SPÉCIFICATIONS TECHNIQUES DÉTAILLÉES

### 4.1 Module : StrategySelector (`core/strategy_selector.py`)

**Nouveau module à créer.**

```python
# Interface publique attendue

class StrategySelector:
    THRESHOLD_INPUT_TOKENS: int = 650_000
    THRESHOLD_OUTPUT_TOKENS: int = 50_000
    COMPRESSION_THRESHOLD_TOKENS: int = 900_000

    def select_strategy(
        self,
        corpus_tokens: int,
        estimated_output_tokens: int = 0
    ) -> GenerationStrategy:
        """
        Retourne:
          - GenerationStrategy.STANDARD
          - GenerationStrategy.HIGH_VOLUME_CACHE
          - GenerationStrategy.SEMANTIC_COMPRESSION_REQUIRED
        """

    def estimate_output_tokens(self, plan: NormalizedPlan) -> int:
        """
        Estime les tokens output = sum(section.target_word_count * 1.35).
        Le facteur 1.35 compense la conversion mots → tokens en français.
        """

    def get_strategy_report(self) -> dict:
        """
        Retourne un rapport lisible pour l'UI:
        {
            "strategy": "HIGH_VOLUME_CACHE",
            "reason": "corpus_input_650k",
            "corpus_tokens": 712000,
            "estimated_output_tokens": 35000,
            "cache_cost_estimate_usd": 0.42,
            "savings_vs_no_cache_usd": 3.80
        }
        """
```

**Tests requis :**
- `test_strategy_input_below_threshold()` → STANDARD
- `test_strategy_input_above_threshold()` → HIGH_VOLUME_CACHE
- `test_strategy_output_above_threshold()` → HIGH_VOLUME_CACHE (même si corpus petit)
- `test_strategy_both_thresholds()` → HIGH_VOLUME_CACHE
- `test_strategy_compression_required()` → SEMANTIC_COMPRESSION_REQUIRED
- `test_estimate_output_tokens_accuracy()` → écart < 15%

### 4.2 Module : ContextManager (`core/context_manager.py`)

**Nouveau module, remplace `rag_engine.py`.**

Ce module est la pièce centrale qui remplace toute la logique RAG. Son rôle est de **fournir le contexte corpus aux agents**, quelle que soit la stratégie active.

```python
class ContextManager:
    """
    Interface unifiée d'accès au corpus pour tous les agents.
    Remplace RAGEngine.
    """

    def __init__(self, strategy: GenerationStrategy, config: dict):
        self.strategy = strategy
        self._cache_manager: Optional[GeminiCacheManager] = None
        self._corpus_xml: Optional[str] = None  # Corpus formaté en mémoire

    async def prepare(
        self,
        corpus: StructuredCorpus,
        system_instruction: str,
        plan: Optional[NormalizedPlan] = None
    ) -> None:
        """
        Prépare le contexte selon la stratégie :
        - STANDARD : formate le corpus en XML, le stocke en mémoire
        - HIGH_VOLUME_CACHE : crée le cache Gemini
        - SEMANTIC_COMPRESSION_REQUIRED : compresse d'abord, puis cache
        """

    def get_context_for_agent(
        self,
        section_id: Optional[str] = None,
        task_type: str = "generation"
    ) -> AgentContextPayload:
        """
        Retourne le payload à injecter dans l'agent :
        - En mode STANDARD : {"corpus_text": str, "cache_name": None}
        - En mode HIGH_VOLUME_CACHE : {"corpus_text": None, "cache_name": str}
        """

    async def cleanup(self) -> None:
        """Supprime le cache Gemini si actif."""

    def get_cache_id(self) -> Optional[str]:
        """Retourne le cache_name Gemini si actif, None sinon."""
```

**Logique interne `prepare()` :**

```
SI strategy == STANDARD :
    1. Appel à format_corpus_xml(corpus)
    2. Stockage en mémoire (self._corpus_xml)
    3. FIN (pas de cache)

SI strategy == HIGH_VOLUME_CACHE :
    1. Appel à format_corpus_xml(corpus)
    2. SI plan fourni ET estimated_output > 50K :
       XML enrichi = corpus_xml + "\n\n" + plan_xml + "\n\n" + writing_instructions_xml
    3. Appel à GeminiCacheManager.create_corpus_cache(xml, system_instruction, ttl)
    4. Stockage du cache_id dans self._cache_id
    5. Lancement du CacheHeartbeat en tâche asyncio de fond

SI strategy == SEMANTIC_COMPRESSION_REQUIRED :
    1. Appel à SemanticCompressor.compress(corpus)
    2. Puis même logique que HIGH_VOLUME_CACHE avec le corpus compressé
```

### 4.3 Module : SemanticCompressor (`core/semantic_compressor.py`)

**Nouveau module à créer.**

```python
class SemanticCompressor:
    """
    Pipeline de compression sémantique à 3 niveaux.
    Utilisé uniquement quand corpus_tokens > 900 000.
    """

    TARGET_TOKENS: int = 800_000
    MIN_FACT_PRESERVATION_RATIO: float = 0.90

    async def compress(
        self,
        corpus: StructuredCorpus,
        target_tokens: int = 800_000,
        llm_provider: Optional[BaseProvider] = None
    ) -> CompressedCorpus:
        """
        Applique les niveaux de compression séquentiellement jusqu'à
        atteindre target_tokens. Retourne un CompressedCorpus avec
        les statistiques de compression.
        """

    def _level1_dedup_and_prune(self, corpus: StructuredCorpus) -> StructuredCorpus:
        """
        Niveau 1 : Déduplication hash + suppression redondances évidentes.
        100% local, pas d'appel LLM. Rapide (<1s).
        """

    async def _level2_extractive(
        self, corpus: StructuredCorpus,
        llm_provider: BaseProvider
    ) -> StructuredCorpus:
        """
        Niveau 2 : Extraction des faits clés en JSON via LLM léger (Flash/Haiku).
        1 appel LLM par document. Peut être parallélisé.
        """

    async def _level3_telegraphic(
        self, corpus: StructuredCorpus,
        llm_provider: BaseProvider
    ) -> StructuredCorpus:
        """
        Niveau 3 : Compression télégraphique (TSC).
        Supprime structure grammaticale, préserve faits/chiffres/entités.
        """

    def get_compression_report(self) -> CompressionReport:
        """
        {
            "original_tokens": int,
            "compressed_tokens": int,
            "compression_ratio": float,
            "levels_applied": List[int],
            "estimated_fact_preservation": float,
            "cost_usd": float,
            "time_seconds": float
        }
        """
```

### 4.4 Module : CacheHeartbeat (`core/cache_heartbeat.py`)

**Nouveau module à créer.** Gère le renouvellement automatique du TTL Gemini.

```python
class CacheHeartbeat:
    """
    Tâche asyncio de fond qui renouvelle le TTL du cache Gemini
    toutes les RENEWAL_INTERVAL secondes pour éviter l'expiration
    pendant une génération longue.
    """

    RENEWAL_INTERVAL_SECONDS: int = 1800  # 30 minutes
    RENEWAL_TTL_EXTENSION_SECONDS: int = 7200  # +2h à chaque renouvellement

    def __init__(self, cache_manager: GeminiCacheManager, cache_name: str):
        self._cache_manager = cache_manager
        self._cache_name = cache_name
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Lance la tâche de heartbeat en fond."""
        self._task = asyncio.create_task(self._heartbeat_loop())

    def stop(self) -> None:
        """Annule la tâche de heartbeat."""
        if self._task and not self._task.done():
            self._task.cancel()

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.RENEWAL_INTERVAL_SECONDS)
            try:
                await self._cache_manager.extend_cache_ttl(
                    self._cache_name,
                    ttl_seconds=self.RENEWAL_TTL_EXTENSION_SECONDS
                )
                logger.info(f"[Heartbeat] Cache {self._cache_name} TTL étendu de +2h")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[Heartbeat] Échec renouvellement TTL: {e}")
                # Non fatal — on réessaie au prochain cycle
```

### 4.5 Adaptation du `ProjectState` (dans `core/orchestrator.py`)

**Modifications à apporter au dataclass `ProjectState` :**

Ajouter les champs suivants :

```python
@dataclass
class ProjectState:
    # ... champs existants ...

    # --- NOUVEAUX CHAMPS v4.0 ---

    # Stratégie active
    processing_strategy: str = "standard"  # "standard" | "high_volume_cache" | "semantic_compression"

    # Statistiques de sélection de stratégie
    token_stats: dict = field(default_factory=lambda: {
        "total_input_corpus": 0,
        "estimated_output": 0,
        "strategy_trigger": None,  # "input_threshold" | "output_threshold" | "both" | None
        "compression_applied": False,
        "compression_ratio": None
    })

    # Lifecycle du cache (remplace cache_id existant)
    cache_lifecycle: dict = field(default_factory=lambda: {
        "cache_name": None,
        "created_at": None,
        "expires_at": None,
        "last_renewed_at": None,
        "renewal_count": 0,
        "deleted_at": None
    })

    # --- CHAMPS EXISTANTS À SUPPRIMER ---
    # chromadb_path  → SUPPRIMER (ChromaDB n'existe plus)
    # rag_indexed    → SUPPRIMER
    # chunk_count    → SUPPRIMER
```

**Modification de `to_dict()` / `from_dict()`** : Sérialiser les nouveaux champs, ignorer gracieusement les anciens champs RAG sur les projets existants (rétrocompatibilité lecture).

### 4.6 Adaptation de l'Orchestrateur Principal (`core/orchestrator.py`)

**Méthode `index_corpus_rag()` → renommer en `prepare_corpus_context()`**

```python
async def prepare_corpus_context(self) -> None:
    """
    Remplace index_corpus_rag().
    Décide de la stratégie et prépare le contexte selon les seuils.
    """
    # 1. Compter les tokens du corpus
    total_tokens = sum(
        self.token_counter.count(doc.text)
        for doc in self.state.corpus.docs
    )
    self.state.token_stats["total_input_corpus"] = total_tokens

    # 2. Sélectionner la stratégie (Input Check)
    strategy = self.strategy_selector.select_strategy(total_tokens)

    # 3. Si compression requise → compresser avant tout
    if strategy == GenerationStrategy.SEMANTIC_COMPRESSION_REQUIRED:
        compressed = await self.semantic_compressor.compress(
            self.state.corpus,
            llm_provider=self._get_provider(task="compression")
        )
        self.state.corpus = compressed.corpus  # Remplacer par version compressée
        self.state.token_stats["compression_applied"] = True
        self.state.token_stats["compression_ratio"] = compressed.ratio
        total_tokens = compressed.compressed_tokens
        strategy = GenerationStrategy.HIGH_VOLUME_CACHE

    # 4. Initialiser le ContextManager
    self.context_manager = ContextManager(strategy, self.config)
    await self.context_manager.prepare(
        corpus=self.state.corpus,
        system_instruction=self._build_global_system_prompt()
    )

    # 5. Mettre à jour l'état
    self.state.processing_strategy = strategy.value
    if strategy == GenerationStrategy.HIGH_VOLUME_CACHE:
        self.state.cache_lifecycle["cache_name"] = self.context_manager.get_cache_id()
        self.state.cache_lifecycle["created_at"] = datetime.utcnow().isoformat()

    logger.info(f"[Context] Stratégie sélectionnée: {strategy.value} ({total_tokens:,} tokens)")
```

**Gestion du seuil Output (Post-Planification) :**

```python
async def finalize_strategy_after_plan(self) -> None:
    """
    Appelé après generate_plan_from_objective().
    Vérifie si l'output estimé force le mode HIGH_VOLUME_CACHE.
    """
    if self.state.processing_strategy == "high_volume_cache":
        return  # Déjà en mode cache, vérification inutile

    estimated_output = self.strategy_selector.estimate_output_tokens(
        self.state.plan
    )
    self.state.token_stats["estimated_output"] = estimated_output

    if estimated_output >= StrategySelector.THRESHOLD_OUTPUT_TOKENS:
        logger.info(
            f"[Strategy] Output estimé {estimated_output:,} tokens >= 50 000 "
            f"→ Forçage HIGH_VOLUME_CACHE"
        )
        self.state.token_stats["strategy_trigger"] = "output_threshold"
        self.state.processing_strategy = "high_volume_cache"

        # Créer le cache maintenant (corpus + plan) même si corpus petit
        await self.context_manager.upgrade_to_cache(
            plan=self.state.plan,
            system_instruction=self._build_global_system_prompt()
        )
```

### 4.7 Adaptation des Agents

#### BaseAgent (`core/agent_framework.py`)

Modifier la signature de `execute()` pour accepter un `AgentContextPayload` au lieu de `corpus_chunks` :

```python
@dataclass
class AgentContextPayload:
    """
    Contient le contexte corpus à injecter dans l'agent.
    Un seul des deux champs est non-null selon la stratégie.
    """
    corpus_text: Optional[str] = None   # Mode STANDARD : texte XML complet
    cache_name: Optional[str] = None    # Mode HIGH_VOLUME_CACHE : ID du cache Gemini
    strategy: str = "standard"

class BaseAgent(ABC):
    async def execute(
        self,
        task: AgentTask,
        context: AgentContextPayload,  # Remplace corpus_chunks: List[str]
        **kwargs
    ) -> AgentResult:
        ...

    def _build_provider_call_args(
        self,
        prompt: str,
        context: AgentContextPayload
    ) -> dict:
        """
        Construit les arguments d'appel au provider selon le mode :
        - Mode STANDARD : ajoute corpus_text dans system_instruction
        - Mode HIGH_VOLUME_CACHE : ajoute cached_content=cache_name
        """
        if context.cache_name:
            return {"cached_content": context.cache_name, "prompt": prompt}
        else:
            return {
                "system": context.corpus_text + "\n\n" + self._base_system_prompt,
                "prompt": prompt
            }
```

#### ArchitectAgent (`core/agents/architect_agent.py`)

- **Avant** : Analysait uniquement le plan et des sections thématiques fournies
- **Après** : Analyse le corpus complet (via `context.corpus_text` ou `context.cache_name`)
- Prompt enrichi : "Analyse l'intégralité du corpus disponible pour identifier les thèmes transversaux, les tensions entre documents, et les zones de risque d'hallucination"
- Output enrichi : Ajouter `cross_document_insights` (insights qui ne peuvent émerger qu'avec une vision globale)

#### WriterAgent (`core/agents/writer_agent.py`)

- **Avant** : Recevait `corpus_chunks: List[str]` (fragments RAG sélectionnés)
- **Après** : Reçoit `context: AgentContextPayload` (accès total via cache ou texte)
- Supprimer de tous les prompts : "Voici les extraits pertinents de corpus :"
- Ajouter dans les prompts : "Tu as accès à l'intégralité du corpus. Cite précisément les documents par leur ID (ex: [doc_001])."
- Règle anti-hallucination renforcée : L'absence d'une information dans le corpus = affirmation impossible, pas de déduction

#### VerifierAgent (`core/agents/verifier_agent.py`)

- **Avant** : Comparait le texte généré avec des chunks vectoriels pré-sélectionnés
- **Après** : Compare avec l'intégralité du corpus → fact-checking de meilleure qualité
- Nouveau type de problème à détecter : `INTER_DOCUMENT_CONTRADICTION` (une affirmation contredit un autre document du corpus)
- Cela n'était pas possible avec le RAG (qui ne fournissait qu'une partie du corpus)

#### EvaluatorAgent et CorrectorAgent

Aucune modification majeure requise. Ils opèrent sur le texte généré et les rapports de vérification. Adapter uniquement la signature `execute()` pour accepter `AgentContextPayload`.

### 4.8 Adaptation du MultiAgentOrchestrator

**Modifications dans `core/multi_agent_orchestrator.py` :**

#### Suppression des références RAG :

```python
# AVANT (à supprimer)
rag_chunks = self.rag_engine.get_corpus_chunks_for_section(section_id)
writer_result = await self.writer_agent.execute(task, corpus_chunks=rag_chunks)

# APRÈS
context_payload = self.context_manager.get_context_for_agent(
    section_id=section_id,
    task_type="generation"
)
writer_result = await self.writer_agent.execute(task, context=context_payload)
```

#### Gestion des outputs > 50 000 tokens (Rolling Architectural State) :

Pour les documents très longs, le simple résumé des sections précédentes ne suffit pas. Implémenter un **Architectural State** :

```python
class ArchitecturalState:
    """
    Maintient la cohérence narrative sur un document de 50K+ tokens.
    Injecté dans chaque prompt de WriterAgent.
    """
    plan_xml: str                    # Plan global complet (non tronqué)
    completed_sections: List[str]    # IDs des sections terminées
    section_summaries: Dict[str, str]  # Résumés des sections terminées (max 200 tokens chacun)
    running_themes: List[str]        # Thèmes transversaux identifiés par l'Architecte
    tone_guide: str                  # Instructions de style global
    word_count_progress: int         # Tokens générés jusqu'ici

    def get_context_injection(self, current_section_id: str) -> str:
        """
        Produit le bloc à injecter avant chaque section :
        - Le plan global (rappel de la structure)
        - Les 3 sections précédentes (résumés)
        - La position de la section actuelle dans le plan
        """
```

**Activation du Rolling Architectural State :**
```python
if self.state.token_stats["estimated_output"] >= 50_000:
    self.use_architectural_state = True
    self.arch_state = ArchitecturalState(
        plan_xml=plan_to_xml(self.state.plan),
        running_themes=architect_result.cross_document_insights,
        tone_guide=architect_result.system_prompt_global
    )
```

#### Parallélisme renforcé pour les outputs > 50K tokens :

```python
# Avec HIGH_VOLUME_CACHE : tous les agents partagent le même cache_name
# → Thread-safe chez Google, pas de conflit
# → Augmenter max_parallel_writers à 8 (au lieu de 4) car pas de charge locale d'embeddings
if self.state.processing_strategy == "high_volume_cache":
    effective_max_writers = min(
        self.config.get("max_parallel_writers", 4) * 2,
        8  # Plafond de sécurité pour éviter le rate-limiting Gemini
    )
```

---

## 5. GESTION DES COÛTS

### 5.1 Mise à Jour du CostTracker

Ajouter un **Cas 5** dans `cost_tracker.py` : **Compression Sémantique**

```python
# CAS 5 : Coût de la compression sémantique (appels LLM aux niveaux 2 et 3)
def track_compression_cost(
    self,
    level: int,
    input_tokens: int,
    output_tokens: int,
    model: str,
    provider: str
) -> None:
    """Enregistre le coût d'une étape de compression sémantique."""
```

Enrichir le `CostReport` avec :
```python
compression_stats: dict = {
    "applied": bool,
    "levels_used": List[int],
    "tokens_saved": int,
    "compression_cost_usd": float,
    "net_savings_vs_no_compression": float
}
```

### 5.2 Affichage des Coûts (UI Streamlit — page_generation.py)

Ajouter un bloc "Analyse de Coûts Stratégique" :

| Ligne | Valeur |
|-------|--------|
| Stratégie active | HIGH_VOLUME_CACHE |
| Corpus (tokens) | 712 000 tokens |
| Cache créé | Oui (gemini-3.1-pro-preview) |
| Économies cache | -90% sur tokens input |
| Coût cache storage | ~$0.38 (durée estimée 45min) |
| Coût compression | $0.00 (non requise) |
| Coût total estimé | $1.24 (vs $8.90 sans cache) |
| Économies totales | **-86%** |

### 5.3 Calcul Break-Even du Cache

Le cache Gemini devient rentable à partir du 2e appel (approximativement). La formule :

```
break_even_sections = ceil(
    cache_creation_cost_usd / (cost_per_section_without_cache - cost_per_section_with_cache)
)
```

Afficher ce calcul dans l'UI avant le lancement de la génération : "Le cache devient rentable à partir de la section N".

---

## 6. MISE À JOUR DE LA CONFIGURATION

### 6.1 Modifications de `config/default.yaml`

**Section à supprimer complètement :**
```yaml
# SUPPRIMER TOUT CE BLOC
rag:
  enabled: true
  chunking:
    strategy: "semantic"
    max_chunk_tokens: 800
  embedding_mode: "local"
  embedding_model: "intfloat/multilingual-e5-large"
  top_k: 10
  reranking_enabled: true
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

**Nouvelle section à ajouter :**
```yaml
# NOUVELLE SECTION v4.0
context_strategy:
  # Seuils de déclenchement du mode cache
  input_threshold_tokens: 650000
  output_threshold_tokens: 50000

  # Seuil de déclenchement de la compression sémantique
  compression_threshold_tokens: 900000

  # Compression sémantique
  compression:
    enabled: true
    target_tokens: 800000
    min_fact_preservation_ratio: 0.90
    max_concurrent_compressions: 4  # Parallélisme niveau 2
    compression_model: "gemini-flash"  # Modèle léger pour les compressions
    compression_provider: "gemini"

# MODIFIER la section gemini existante
gemini:
  caching_enabled: true          # Changer false → true
  cache_ttl_seconds: 7200        # Conserver
  heartbeat_enabled: true        # NOUVEAU
  heartbeat_interval_seconds: 1800  # NOUVEAU - 30 minutes
  thinking_level_mode: "auto"

  # Seuil de déclenchement du cache (minimum API)
  # Note: override automatique si context_strategy.input_threshold est atteint
  cache_min_tokens: 2048  # Contrainte technique Gemini Pro

# MODIFIER la section multi_agent
multi_agent:
  enabled: false
  max_parallel_writers: 4
  max_parallel_writers_cache_mode: 8  # NOUVEAU - doublement en mode cache
  max_parallel_verifiers: 4
  quality_threshold: 3.5

  # Rolling Architectural State (activé auto si output > 50K)
  architectural_state:
    enabled: true
    max_section_summary_tokens: 200
    inject_full_plan: true  # Réinjecter le plan global à chaque section

# SUPPRIMER la section plan_corpus_linker (inutile sans RAG)
```

### 6.2 `config/model_pricing.yaml` — Section Google (aucune modification requise)

La section Google est **déjà correctement configurée** dans le codebase existant. Aucune modification nécessaire. Pour référence, les valeurs actives :

```yaml
# Phase 5 — Google (Gemini 3.1) — config/model_pricing.yaml ACTUELLE
google:
  gemini-3.1-pro-preview:          # Modèle principal — rédaction, architecture, vérification
    input: 2.00                    # USD / M tokens (≤ 200K tokens)
    input_cached: 0.20             # USD / M tokens cachés → -90% vs standard
    input_long_context: 4.00       # USD / M tokens (> 200K tokens)
    output: 12.00                  # USD / M tokens output
    output_long_context: 18.00     # USD / M tokens output (long context)
    cache_storage_per_hour: 0.50   # USD / heure / M tokens stockés
    context_window: 1000000        # 1M tokens fenêtre réelle
    max_output_tokens: 65536       # ⚠️ CRITIQUE : plafond output par appel

  gemini-3-flash-preview:          # Modèle compression sémantique et tâches légères
    input: 0.50                    # USD / M tokens
    input_cached: 0.05             # USD / M tokens cachés → -90%
    output: 3.00                   # USD / M tokens output
    cache_storage_per_hour: 1.00   # USD / heure / M tokens (plus cher à stocker que Pro)
    context_window: 1000000        # 1M tokens
    max_output_tokens: 65536       # ⚠️ Même plafond que Pro
    # Pas de long_context_threshold ni repricing pour Flash
```

**Rôle de chaque modèle dans l'architecture v4.0 :**

| Tâche | Modèle recommandé | Justification |
|-------|------------------|---------------|
| Agent Architecte | `gemini-3.1-pro-preview` | Raisonnement complexe sur corpus global |
| Agent Rédacteur | `gemini-3.1-pro-preview` | Qualité rédactionnelle maximale |
| Agent Vérificateur | `gemini-3.1-pro-preview` | Fact-checking rigoureux (accès cache) |
| Agent Évaluateur | `gemini-3.1-pro-preview` ou GPT-4.1 | Scoring qualité (sans accès corpus) |
| Agent Correcteur | `gemini-3.1-pro-preview` ou GPT-4.1 | Correction stylistique |
| Compression Niveau 2 (extractive) | `gemini-3-flash-preview` | Extraction faits — vitesse et coût |
| Compression Niveau 3 (télégraphique) | `gemini-3-flash-preview` | Compression de forme — modèle léger suffit |
| Format corpus XML (pré-cache) | `gemini-3-flash-preview` | Tâche mécanique, faible coût |

---

## 7. PLAN DE MIGRATION

### Phase 1 : Fondations (Semaine 1)

**Objectif** : Créer les nouveaux modules sans casser l'existant.

- [ ] Créer `core/strategy_selector.py` avec `GenerationStrategy` enum et `StrategySelector`
- [ ] Créer `core/context_manager.py` avec mode STANDARD uniquement (pas de cache)
- [ ] Créer `core/cache_heartbeat.py`
- [ ] Modifier `ProjectState` : ajouter les nouveaux champs (rétrocompatible)
- [ ] Mettre à jour `config/default.yaml` : ajouter `context_strategy`, garder `rag` pour l'instant
- [ ] Tests unitaires des nouveaux modules

### Phase 2 : Intégration de la Logique Décisionnelle (Semaine 2)

**Objectif** : Brancher le StrategySelector dans le pipeline principal.

- [ ] Modifier `orchestrator.py` : remplacer `index_corpus_rag()` par `prepare_corpus_context()`
- [ ] Ajouter `finalize_strategy_after_plan()` post-planification
- [ ] Modifier `multi_agent_orchestrator.py` : supprimer références RAG, brancher ContextManager
- [ ] Adapter `BaseAgent.execute()` : remplacer `corpus_chunks` par `AgentContextPayload`
- [ ] Tests d'intégration bout-en-bout (mode STANDARD)

### Phase 3 : Mode HIGH_VOLUME_CACHE (Semaine 3)

**Objectif** : Activer et valider le mode cache avec les seuils 650K/50K.

- [ ] Compléter `ContextManager.prepare()` pour le mode HIGH_VOLUME_CACHE
- [ ] Intégrer `CacheHeartbeat` dans `ContextManager`
- [ ] Valider la gestion `try...finally` pour la suppression du cache
- [ ] Mettre à jour `CostTracker` : tracking storage cache, rapport enrichi
- [ ] Tests d'intégration avec mock Gemini API (éviter les coûts réels en CI)
- [ ] Mettre à jour l'UI (`page_generation.py`) : affichage stratégie + coûts

### Phase 4 : SemanticCompressor (Semaine 4)

**Objectif** : Implémenter le fallback pour les corpus > 900K tokens.

- [ ] Créer `core/semantic_compressor.py` avec les 3 niveaux
- [ ] Intégrer dans `prepare_corpus_context()` du Orchestrator
- [ ] Valider sur un corpus de test > 900K tokens
- [ ] Mettre à jour `CostTracker` : cas compression
- [ ] Tests unitaires des 3 niveaux de compression séparément

### Phase 5 : Nettoyage et Suppression RAG (Semaine 5)

**Objectif** : Supprimer proprement toutes les dépendances RAG.

- [ ] Supprimer `core/rag_engine.py`
- [ ] Supprimer `core/local_embedder.py`
- [ ] Supprimer `core/reranker.py`
- [ ] Supprimer `core/semantic_chunker.py`
- [ ] Supprimer `core/plan_corpus_linker.py`
- [ ] Retirer `chromadb`, `fastembed`, `sentence-transformers` de `requirements.txt`
- [ ] Supprimer la section `rag` de `config/default.yaml`
- [ ] Supprimer les dossiers `chromadb/` des projets existants (migration)
- [ ] Exécuter la suite de tests complète et corriger les régressions

### Phase 6 : Validation Finale et Optimisation (Semaine 6)

- [ ] Tests de charge : corpus de 700K, 900K, 1.2M tokens
- [ ] Validation des coûts réels vs estimés (écart < 10%)
- [ ] Validation de la qualité des documents générés (score EvaluatorAgent ≥ 3.8/5)
- [ ] Documentation de la migration dans README.md
- [ ] Mise à jour des profils YAML (`profiles/default/`)

---

## 8. GESTION DES RISQUES

### 8.1 Tableau des Risques

| # | Risque | Probabilité | Impact | Mitigation |
|---|--------|-------------|--------|------------|
| R01 | Cache Gemini expiré avant fin de génération | Moyenne | Élevé | CacheHeartbeat + try/finally + recréation auto |
| R02 | Coût cache storage > coût RAG pour petits corpus | Faible | Moyen | Seuils 650K/50K calibrés pour éviter ce cas |
| R03 | Compression Niveau 3 perd des faits critiques | Faible | Élevé | Ratio préservation ≥ 0.90, vérification par LLM |
| R04 | Rate-limiting Gemini avec 8 agents parallèles | Moyenne | Moyen | Max 8 agents, exponential backoff dans BaseAgent |
| R05 | Corpus > 1M tokens même après compression | Très faible | Très élevé | Alerte UI + sélection manuelle obligatoire |
| R06 | Provider Gemini indisponible en mode HIGH_VOLUME | Faible | Élevé | Fallback sur mode STANDARD avec corpus tronqué + alerte |
| R07 | Projets existants avec ChromaDB cassés | Certaine | Moyen | Migration gracieuse : ignorer `chromadb_path` si inexistant |
| R08 | TTL cache insuffisant pour génération > 2h | Possible | Élevé | Heartbeat extensible à l'infini |

### 8.2 Plan de Fallback Général

```
Tentative stratégie principale échoue :
│
├─ Si HIGH_VOLUME_CACHE échoue (API Gemini down) :
│   → Alerter l'utilisateur via UI
│   → Proposer : "Continuer en mode STANDARD (injection directe) ?"
│   → Si STANDARD : tronquer le corpus à 500K tokens (avec log des docs exclus)
│
├─ Si SEMANTIC_COMPRESSION échoue (après 3 tentatives) :
│   → Afficher les documents classés par taille
│   → Demander à l'utilisateur de décocher les moins importants
│   → Relancer avec corpus réduit
│
└─ Si les deux échouent :
    → Sauvegarder l'état (ProjectState) et permettre reprise plus tard
    → Email/notification d'erreur si configuré
```

---

## 9. TESTS ET VALIDATION

### 9.1 Tests Unitaires à Créer/Mettre à Jour

**Nouveaux fichiers de tests :**

```
tests/unit/
├── test_strategy_selector.py      # 8 tests
├── test_context_manager.py        # 10 tests (modes STANDARD et CACHE)
├── test_semantic_compressor.py    # 12 tests (1 par niveau × variantes)
├── test_cache_heartbeat.py        # 5 tests (start, stop, renewal, crash)
└── test_architectural_state.py    # 6 tests
```

**Tests existants à mettre à jour :**
```
tests/unit/
├── test_orchestrator.py           # Remplacer index_corpus_rag → prepare_corpus_context
├── test_multi_agent_orchestrator.py  # Remplacer corpus_chunks → AgentContextPayload
├── test_cost_tracker.py           # Ajouter tests cas compression
├── test_project_state.py          # Tester nouveaux champs
└── test_writer_agent.py           # Tester avec AgentContextPayload
```

### 9.2 Tests d'Intégration

```
tests/integration/
├── test_pipeline_standard_mode.py          # Corpus < 650K, output < 50K
├── test_pipeline_high_volume_input.py      # Corpus > 650K
├── test_pipeline_high_volume_output.py     # Output > 50K (corpus petit)
├── test_pipeline_semantic_compression.py   # Corpus > 900K (avec mock LLM)
├── test_cache_lifecycle_full.py            # Création → renouvellement → suppression
└── test_cost_tracking_accuracy.py         # Vérification coûts vs factures
```

### 9.3 Critères de Succès

| Critère | Seuil |
|---------|-------|
| Suite de tests (637 tests) | ≥ 635 passés (les 2 restants = bugs pré-existants) |
| Couverture code nouveaux modules | ≥ 85% |
| Qualité documents générés (EvaluatorAgent) | ≥ 3.8 / 5.0 |
| Précision estimation coûts | Écart ≤ 10% vs coûts réels |
| Temps de création cache (corpus 650K) | ≤ 45 secondes |
| Économies en mode cache vs sans cache | ≥ 80% sur tokens input |
| Taux de réussite compression (niv. 1+2) | ≥ 95% corpus → sous 900K tokens |

---

## 10. ASPECTS NON-FONCTIONNELS

### 10.1 Performance

- **Temps de démarrage** : Sans ChromaDB ni fastembed à charger, le démarrage doit être **< 5 secondes** (vs ~15 secondes actuellement à cause des modèles ONNX)
- **Mémoire RAM** : Réduction de ~2 Go en production (suppression des modèles fastembed/sentence-transformers)
- **Parallélisme** : En mode cache, passer de 4 à 8 agents parallèles (le cache est thread-safe côté Google)

### 10.2 Observabilité

Ajouter dans les logs (fichier `utils/logger.py` / ActivityLog) :
- Événement : `STRATEGY_SELECTED` (avec raison et seuils)
- Événement : `CACHE_CREATED` (avec cache_name, TTL, tokens)
- Événement : `CACHE_RENEWED` (avec nouveau TTL)
- Événement : `CACHE_DELETED` (durée totale de vie)
- Événement : `COMPRESSION_LEVEL_N_APPLIED` (avec ratio)
- Métrique : `tokens_saved_by_cache` par section

### 10.3 Sécurité des Données

- Le corpus est envoyé aux serveurs Google via l'API Gemini → Respecter la politique de données de l'utilisateur
- Ajouter un avertissement UI : "Votre corpus sera stocké temporairement sur les serveurs Google (TTL : 2h)"
- Ajouter option de désactivation du cache Gemini (`gemini.caching_enabled: false`) pour les corpus confidentiels

### 10.4 Rétrocompatibilité

- Les projets existants (avec état JSON contenant `chromadb_path`) doivent se charger sans erreur
- Si un ancien état contient `rag_indexed: true`, l'ignorer silencieusement
- Les profils YAML dans `profiles/default/` n'ont pas besoin d'être modifiés immédiatement

---

## 11. GLOSSAIRE TECHNIQUE

| Terme | Définition dans ce contexte |
|-------|----------------------------|
| **Full Context** | Mode où l'intégralité du corpus est accessible à chaque agent, sans sélection partielle |
| **Context Caching** | Mécanisme Gemini permettant de stocker le corpus une fois et de le référencer par ID |
| **Strategy Selector** | Module décisionnel qui choisit entre STANDARD et HIGH_VOLUME_CACHE |
| **STANDARD** | Mode d'injection directe du corpus dans le prompt (corpus < 650K, output < 50K) |
| **HIGH_VOLUME_CACHE** | Mode cache Gemini (corpus ≥ 650K OU output ≥ 50K tokens) |
| **Compression Sémantique** | Pipeline de réduction de densité informationnelle (corpus > 900K tokens) |
| **Heartbeat** | Tâche asyncio qui renouvelle le TTL du cache toutes les 30 minutes |
| **Architectural State** | Contexte narratif global injecté dans chaque section pour les documents > 50K tokens |
| **TTL** | Time To Live — durée de vie d'un cache avant suppression automatique |
| **RAG** | Retrieval-Augmented Generation — architecture à base de recherche vectorielle (SUPPRIMÉE) |
| **TSC** | Telegraphic Semantic Compression — compression grammaticale préservant les faits |
| **AgentContextPayload** | Objet standard passé à chaque agent contenant soit le corpus texte, soit le cache_name |

---

## ANNEXE A : Comparaison Architecturale

### Avant (v3.x — RAG)

```
Corpus → [ChromaDB] → Embeddings → Vecteurs
                                       ↓
Section N → Recherche vectorielle → Top-K chunks → LLM → Texte
```

**Limites :**
- Vision partielle (seulement K fragments)
- Dépendance aux embeddings locaux (ONNX, 2 Go)
- Qualité du fact-checking limitée (peut manquer des contradictions inter-documents)
- Latence d'indexation ChromaDB (~2min pour 50K chunks)

### Après (v4.0 — Full Context)

```
Corpus → [Format XML] → Cache Gemini (une fois)
                               ↓
Section N → cache_name → LLM (vision totale) → Texte
Section N+1 → cache_name → LLM → Texte
...
Section N+k → Suppression cache
```

**Avantages :**
- Vision totale pour chaque agent
- Zéro dépendance locale ONNX
- Fact-checking inter-documents possible
- Démarrage immédiat (pas d'indexation)
- 85-90% d'économies sur les tokens input répétés

---

## ANNEXE B : Références et Sources

**Documentation Officielle :**
- [Gemini API — Context Caching](https://ai.google.dev/gemini-api/docs/caching)
- [Gemini API — Long Context](https://ai.google.dev/gemini-api/docs/long-context)
- [Gemini API — Pricing 2026](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini — Implicit Caching (blog Google Developers)](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/)

**Recherche Académique :**
- [Semantic Compression With Large Language Models — arXiv:2304.12512](https://arxiv.org/abs/2304.12512)
- [TRIM: Token Reduction and Inference Modeling](https://arxiv.org/html/2412.07682)
- [SemToken: Semantic-Aware Tokenization](https://arxiv.org/html/2508.15190)
- [Telegraphic Semantic Compression (TSC)](https://developer-service.blog/telegraphic-semantic-compression-tsc-a-semantic-compression-method-for-llm-contexts/)

**Bonnes Pratiques Multi-Agents :**
- [LLM Orchestration — orq.ai](https://orq.ai/blog/llm-orchestration)
- [Circuit Breakers for LLM Services](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/)
- [LangGraph — Architecture Guide 2025](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/)

**Outils :**
- [Token Reducer — PyPI](https://pypi.org/project/token-reducer/)
- [Context Caching Guide — Luis Aviles](https://luixaviles.com/2025/12/a-practical-guide-to-use-context-caching-gemini/)
- [Optimizing LLM Costs — Phase2](https://phase2online.com/2025/04/28/optimizing-llm-costs-with-context-caching/)
