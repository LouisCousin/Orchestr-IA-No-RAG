# Cahier des Charges — Correction des Bugs Identifiés

**Projet** : Orchestr'IA (No-RAG)
**Date** : 2026-02-28
**Auteur** : Revue de code approfondie (Claude)
**Périmètre** : Totalité du code source (43 modules core, 4 providers, 9 utilitaires, 8 pages UI, 3 modules API)
**Tests exécutés** : 632 passés, 5 échoués sur 637

---

## 1. Tableau Synthétique des Bugs par Ordre de Gravité

| # | Gravité | Module | Fichier | Ligne(s) | Description Succincte | Avantages du Correctif | Inconvénients / Risques du Correctif |
|---|---------|--------|---------|----------|----------------------|----------------------|-------------------------------------|
| **B01** | **CRITIQUE** | MessageBus | `core/message_bus.py` | 37 | `asyncio.Lock()` instancié hors boucle événementielle | Supprime le crash `RuntimeError` au démarrage quand le bus est créé hors async | Nécessite un refactoring du pattern d'initialisation (lazy init) |
| **B02** | **CRITIQUE** | MessageBus | `core/message_bus.py` | 37-48 | Mélange `asyncio.Lock` et `threading.Lock` : la `_history` est accédée par les deux sans protection commune | Élimine les corruptions de données et les race conditions sur l'historique | Complexifie la logique de verrouillage ; nécessite de choisir un paradigme unique (async ou threading) |
| **B03** | **CRITIQUE** | Orchestrator | `core/orchestrator.py` | 940+ | `generate_all_sections` utilise un `ThreadPoolExecutor` pour le pipelining (post-gen), mais `save_state` et les mutations de `self.state` ne sont protégées que par un `Lock` threading — les futures pouvant s'exécuter après `return`, elles risquent de modifier un état déjà renvoyé | Garantit l'intégrité de l'état du projet sous pipelining concurrent | Peut réduire légèrement les performances du pipelining si le verrouillage est trop large |
| **B04** | **CRITIQUE** | CorpusAcquirer | `core/corpus_acquirer.py` | 570-574 | `_get_seq_lock()` crée un `asyncio.Lock` lazily dans un contexte async sans protection : double initialisation possible sous concurrence | Supprime la possibilité de créer deux verrous distincts pour la même ressource | Minime ; ajout d'un `if` atomique ou initialisation dans `__init__` |
| **B05** | **HAUTE** | GeminiCacheManager | `core/gemini_cache_manager.py` | 52-57 | `_get_client()` initialise `self._client` sans verrou ; accès concurrent crée plusieurs clients API | Évite les doublons de client, économise la mémoire et prévient les problèmes de rate-limiting | Ajout d'un verrou avec légère latence sur le premier appel |
| **B06** | **HAUTE** | CorpusAcquirer | `core/corpus_acquirer.py` | 666-676 | `acquire_urls_sync_or_async()` détecte le loop async et utilise `ThreadPoolExecutor` en fallback ; les exceptions du thread ne se propagent pas correctement | Propagation fiable des erreurs d'acquisition aux appelants synchrones | Refactoring modéré ; tester les deux modes d'appel (sync et async) |
| **B07** | **HAUTE** | API Projects | `api/routes/projects.py` | 359-360 | Accès à `state.agent_architecture` sans vérifier que l'attribut existe sur le `ProjectState` | Supprime le `AttributeError` quand l'API est appelée avant la phase multi-agents | Test de régression nécessaire pour tous les endpoints |
| **B08** | **HAUTE** | ProvidersRegistry | `utils/providers_registry.py` | 51-70 | `create_provider()` retourne `None` silencieusement pour un fournisseur inconnu, alors que `get_provider_info()` lève `ValueError` — incohérence d'API | Comportement cohérent : les appelants détectent immédiatement le problème | Peut casser le code appelant qui traite `None` au lieu de capturer l'exception |
| **B09** | **HAUTE** | ProvidersRegistry | `utils/providers_registry.py` | 45-48 | `get_default_model()` retourne `"gpt-4o"` (modèle OpenAI) comme fallback même pour un fournisseur inconnu | Retourne une erreur claire au lieu d'un modèle inutilisable | Impact faible |
| **B10** | **HAUTE** | Page Plan | `pages/page_plan.py` | 185-187 | `st.session_state.get("orchestrator")` peut être `None` ; accès aux attributs `.state`, `.plan` sans vérification | Supprime le crash `AttributeError` quand l'utilisateur navigue directement vers la page Plan | Ajout d'une condition avec message d'erreur utilisateur |
| **B11** | **HAUTE** | Page Plan | `pages/page_plan.py` | 227-231 | `.lower()` appelé sur `section.title` ou `theme` potentiellement `None` | Supprime le `TypeError` sur les plans avec sections sans titre | Défensif ; ajout de `or ""` sur les valeurs nullable |
| **B12** | **HAUTE** | Page Bibliothèque | `pages/page_bibliotheque.py` | 64, 110 | `st.session_state.selected_template_id` accédé sans initialisation préalable | Supprime le `AttributeError` lors de la navigation vers les templates | Initialisation dans un bloc `if "key" not in st.session_state` en haut de page |
| **B13** | **HAUTE** | Page Dashboard | `pages/page_dashboard.py` | 127 | `criterion.get("name", criterion.get("id", ""))` peut produire une clé `None` dans le dict `row` | Supprime le `KeyError` ou la clé `None` dans les DataFrames Plotly | Ajout d'un fallback `"unknown"` |
| **B14** | **HAUTE** | Page Génération | `pages/page_generation.py` | 198-203 | L'extraction du corpus peut échouer silencieusement ; la génération continue avec un corpus vide | Détecte l'échec d'extraction et affiche un avertissement clair à l'utilisateur | Peut bloquer le workflow si l'extraction échoue toujours (fallback nécessaire) |
| **B15** | **HAUTE** | Token Counter | `utils/token_counter.py` | 20-22 | `_heuristic_count("")` retourne `max(1, 0) = 1` — un texte vide est compté comme 1 token | Comptage correct : texte vide = 0 token | Certains calculs de coût pourraient diviser par zéro si 0 est retourné (vérifier les appelants) |
| **B16** | **HAUTE** | File Utils | `utils/file_utils.py` | 78-92 | `get_next_sequence_number()` n'est pas thread-safe (TOCTOU) et `format_sequence_name()` produit des noms de 4+ chiffres quand `seq >= 1000`, cassant le regex `\d{3}` | Prévient les collisions de noms de fichiers et la perte de numérotation au-delà de 999 fichiers | Changer le regex en `\d+` ou utiliser un format `{:04d}` ; migration des fichiers existants |
| **B17** | **HAUTE** | Page Accueil | `pages/page_accueil.py` | 207-210 | `_restore_message` stocké dans `session_state` persiste indéfiniment sans nettoyage | Messages de restauration périmés affichés à chaque rerun | Pattern `st.session_state.pop()` dans le rendu ou utiliser `st.toast()` |
| **B18** | **MOYENNE** | Page Configuration | `pages/page_configuration.py` | 32 | `st.session_state.pop("_restore_message")` modifie le state pendant le rendu Streamlit | Conforme au modèle réactif de Streamlit | Déplacer en callback ou utiliser un flag de consommation |
| **B19** | **MOYENNE** | TextExtractor | `core/text_extractor.py` | 497-505 | `del converter` / `del result` dans un `finally` peut lever `NameError` si la variable n'a pas été définie | Nettoyage robuste des ressources Docling | Entourer les `del` d'un `try/except NameError` (déjà partiellement fait) |
| **B20** | **MOYENNE** | SemanticChunker | `core/semantic_chunker.py` | 140-143 | La fusion d'un chunk court avec le précédent peut créer un chunk dépassant `max_chunk_tokens` | Garantit le respect de la limite `max_chunk_tokens` | Ajout d'une vérification post-fusion avec resplit si nécessaire |
| **B21** | **MOYENNE** | RAG Engine | `core/rag_engine.py` | 81-86 | `chunk_overlap >= chunk_size` est corrigé silencieusement en `chunk_size // 4` sans avertir clairement l'utilisateur de la config UI | L'utilisateur voit pourquoi son overlap a été réduit | Ajout d'un avertissement plus visible dans l'UI |
| **B22** | **MOYENNE** | WebSocket Route | `api/routes/ws.py` | 123-128 | `new_bus is not bus` utilise la comparaison d'identité : si le bus est recréé avec la même adresse mémoire, le listener ne sera pas réenregistré | Comparaison correcte via un identifiant unique du bus | Faible probabilité d'occurrence ; correctif préventif |
| **B23** | **MOYENNE** | WebSocket Route | `api/routes/ws.py` | 86-90 | `except Exception` trop large dans `ws_listener` : capture des `asyncio.CancelledError` (Python <3.9) et empêche l'arrêt propre | Arrêt propre des WebSockets | Distinguer `CancelledError`, `ConnectionClosed` et les erreurs applicatives |
| **B24** | **MOYENNE** | Page Export | `pages/page_export.py` | 208 | `state.cost_report` peut être `None` ; appel `.get()` sur `None` | Supprime le `TypeError` à l'export si aucun coût n'est enregistré | Vérification `cost_report or {}` |
| **B25** | **MOYENNE** | Page Acquisition | `pages/page_acquisition.py` | 625-626 | Bouton édition de métadonnées ne valide pas que `doc_id` référence un document existant | Empêche l'affichage d'un formulaire pour un document inexistant | Ajout d'un `if doc_id in docs:` |
| **B26** | **MOYENNE** | Page Acquisition | `pages/page_acquisition.py` | 277-279 | `st.session_state.project_state.config` accédé sans vérifier que `project_state` existe | Supprime l'`AttributeError` si la page est accédée hors contexte projet | Guard clause en haut de la fonction |
| **B27** | **MOYENNE** | Page Plan | `pages/page_plan.py` | 343-344 | `glossary.apply_generated_terms(generated)` sans vérifier le format du retour | Évite un crash si l'IA retourne un glossaire malformé | `try/except` avec message utilisateur |
| **B28** | **MOYENNE** | Page Plan | `pages/page_plan.py` | 108-109 | Warning affiché si provider indisponible mais la fonction ne retourne pas — le code continue et crash | Ajout d'un `return` explicite après le warning | Faible |
| **B29** | **MOYENNE** | Cost Tracker | `core/cost_tracker.py` | 258-262 | Message warning hardcodé `"$4/1M input, $18/1M output"` qui ne correspond pas forcément aux tarifs réels du modèle | Message dynamique basé sur les tarifs réels | Modification cosmétique |
| **B30** | **MOYENNE** | ExportEngine | `core/export_engine.py` | 388 | Détection de sous-titres en gras (`**texte**`) : `stripped.count("**") == 2` est incorrect si le texte contient `**` au milieu (ex: `**bold** and **bold**`) | Parsing correct du markdown bold | Utiliser une regex pour détecter le pattern `^\\*\\*[^*]+\\*\\*$` |
| **B31** | **MOYENNE** | Config | `utils/config.py` | 21-27 | `load_yaml()` ne gère pas `FileNotFoundError` ni `yaml.YAMLError` — crash non informatif | Message d'erreur clair pour les fichiers de config manquants ou malformés | Ajouter un `try/except` avec logging |
| **B32** | **MOYENNE** | ActivityLog | `utils/logger.py` | 12-41 | `ActivityLog.entries` n'a pas de limite de taille — croissance mémoire illimitée en session longue | Prévient les fuites mémoire | Ajouter un `maxlen` (deque) ou un pruning périodique |
| **B33** | **MOYENNE** | ActivityLog | `utils/logger.py` | 12-41 | Pas de thread-safety sur `entries` (list mutable sans lock) | Prévient les corruptions de données en multi-thread | Ajouter un `threading.Lock` ou utiliser `collections.deque` (thread-safe pour append/pop) |
| **B34** | **MOYENNE** | CheckpointManager | `core/checkpoint_manager.py` | 47-48 | `is_enabled()` avec `getattr()` silencieux : un type de checkpoint mal orthographié retourne `False` au lieu de signaler l'erreur | Détection immédiate des erreurs de configuration | Valider `checkpoint_type` contre les champs connus |
| **B35** | **MOYENNE** | Page Génération | `pages/page_generation.py` | 356-357 | `state.deferred_sections.append()` — la persistance n'est pas immédiate si plusieurs sections sont reportées dans la même passe | Persistance immédiate après chaque ajout | Performance légèrement réduite par les sauvegardes fréquentes |
| **B36** | **FAIBLE** | Token Counter | `utils/token_counter.py` | 25-29 | `estimate_pages()` accepte un `token_count` négatif et retourne un nombre de pages négatif | Validation d'entrée avec `max(0, token_count)` | Impact nul en pratique |
| **B37** | **FAIBLE** | SemanticChunker | `core/semantic_chunker.py` | 172-173 | Edge case: si `len(sentences) < overlap_sentences`, l'overlap est vide | Comportement explicitement géré | Probabilité d'occurrence très faible |
| **B38** | **FAIBLE** | CorpusDeduplicator | `core/corpus_deduplicator.py` | 160-165 | `tokens_saved` comptabilise le même document plusieurs fois si dupliqué N fois | Comptage correct des économies de tokens | Impact cosmétique uniquement (rapport) |
| **B39** | **FAIBLE** | Page Dashboard | `pages/page_dashboard.py` | 161 | Quadruple accolades `{{{{NEEDS_SOURCE}}}}` dans f-string — affiche `{{NEEDS_SOURCE}}` au lieu de `{NEEDS_SOURCE}` | Affichage correct du marqueur | Correction triviale |
| **B40** | **FAIBLE** | Page Acquisition | `pages/page_acquisition.py` | 311 | Variable `updated = False` définie mais jamais utilisée | Code mort supprimé | Nettoyage sans risque |
| **B41** | **FAIBLE** | Page Génération | `pages/page_generation.py` | 437 | `state.section_summaries.append()` peut lever `AttributeError` si le champ est `None` après chargement disque | Initialisation défensive `state.section_summaries = state.section_summaries or []` | Impact nul si l'état est toujours bien initialisé |
| **B42** | **FAIBLE** | Tests | `tests/` | N/A | 5 tests échouent : 2 à cause de `bs4` absent, 2 à cause de `openpyxl` absent, 1 à cause de `docx` absent — dépendances non installées | Ajout des dépendances manquantes dans `requirements.txt` ou skip conditionnel | Aucun risque |

---

## 2. Analyse Détaillée par Catégorie

### 2.1 Bugs Critiques (4 bugs) — Impact : Crash ou Corruption de Données

#### B01 + B02 : MessageBus — Problème d'initialisation et de verrouillage mixte

**Fichier** : `src/core/message_bus.py`

**Problème** :
- Ligne 37 : `self._lock: asyncio.Lock = asyncio.Lock()` — un `asyncio.Lock` instancié en dehors d'un event loop provoque un `RuntimeError` sur certaines versions de Python (<3.10) ou si le `MessageBus` est créé avant le démarrage de la boucle événementielle.
- Lignes 37-48 : Le `_history` est accédé via `_sync_lock` (threading) dans `publish()` et `store_alert_sync()`, mais aussi via `_lock` (asyncio) dans d'autres méthodes. Il n'y a aucune coordination entre ces deux mécanismes, ce qui crée une race condition.

**Correctif proposé** :
1. Remplacer l'initialisation immédiate de `asyncio.Lock()` par une propriété paresseuse (`@property` avec `_lock = None`).
2. Unifier la stratégie de verrouillage : soit utiliser uniquement `threading.Lock` (compatible sync et async), soit cloisonner strictement les accès sync et async sur des structures séparées.

**Tests de validation** :
- Créer un `MessageBus` hors event loop → pas de crash.
- Accès concurrent depuis 10 coroutines et 5 threads → pas de corruption de `_history`.

---

#### B03 : Orchestrator — Race condition sous pipelining ThreadPool

**Fichier** : `src/core/orchestrator.py`

**Problème** :
Le pipelining asynchrone (Phase 4) lance l'évaluation post-génération de la section N dans un `ThreadPoolExecutor` pendant que la section N+1 est générée. Cependant :
- Les futures soumises au pool continuent de s'exécuter après le `return` de `generate_all_sections()`.
- `self.state.generated_sections` est muté par ces futures après que la méthode a renvoyé l'état.
- Le `_state_lock` protège `save_state()`, mais pas les lectures de `self.state` faites par le code appelant.

**Correctif proposé** :
1. Ajouter un `_executor.shutdown(wait=True)` avant le `return` final.
2. Ou utiliser un pattern producteur/consommateur avec une `Queue` pour collecter les résultats post-gen.

**Tests de validation** :
- Générer 10 sections avec pipelining → vérifier que toutes les évaluations post-gen sont terminées avant le retour.

---

#### B04 : CorpusAcquirer — Double initialisation du verrou async

**Fichier** : `src/core/corpus_acquirer.py`

**Problème** :
`_get_seq_lock()` initialise `self._seq_lock` de manière paresseuse sans aucune protection. Si deux coroutines appellent cette méthode simultanément, deux `asyncio.Lock` distincts sont créés, rendant le verrouillage inefficace.

**Correctif proposé** :
Initialiser `self._seq_lock = asyncio.Lock()` dans `__init__()`, ou utiliser un `threading.Lock` pour protéger la création paresseuse.

---

### 2.2 Bugs de Haute Gravité (13 bugs) — Impact : Comportement Incorrect ou Crash Utilisateur

#### B05 : GeminiCacheManager — Client non singleton

**Fichier** : `src/core/gemini_cache_manager.py`

**Problème** : `_get_client()` initialise `self._client` sans verrou. Sous concurrence, plusieurs instances du client API Google sont créées.

**Correctif** : Ajout d'un `threading.Lock` pour l'initialisation.

---

#### B06 : CorpusAcquirer — Propagation d'erreurs sync/async

**Fichier** : `src/core/corpus_acquirer.py`

**Problème** : `acquire_urls_sync_or_async()` utilise `ThreadPoolExecutor` quand un event loop est déjà actif, mais les exceptions dans le thread ne se propagent pas correctement.

**Correctif** : Appeler `future.result()` avec `timeout` pour capturer les exceptions, ou utiliser `asyncio.run_coroutine_threadsafe()`.

---

#### B07 : API Projects — AttributeError sur `agent_architecture`

**Fichier** : `src/api/routes/projects.py`

**Problème** : L'endpoint accède à `state.agent_architecture` qui peut ne pas exister si le pipeline multi-agents n'a pas encore été exécuté.

**Correctif** : `getattr(state, "agent_architecture", None)`.

---

#### B08 + B09 : ProvidersRegistry — Comportement incohérent

**Fichier** : `src/utils/providers_registry.py`

**Problème** : `create_provider()` retourne `None` pour un fournisseur inconnu (B08), et `get_default_model()` retourne `"gpt-4o"` pour un fournisseur inconnu (B09).

**Correctif** :
- B08 : Lever `ValueError` comme `get_provider_info()`.
- B09 : Lever `ValueError` au lieu de retourner un modèle OpenAI.

---

#### B10 à B14 : Pages Streamlit — Crashes UI

**Problèmes communs** : Accès à des attributs du `session_state` non initialisés, absence de null-checks sur des valeurs optionnelles, poursuite du code après un warning.

**Correctif global** :
- Ajouter des blocs de garde (`guard clause`) en début de chaque fonction de page.
- Initialiser toutes les clés `session_state` nécessaires dans `app.py`.
- Ajouter `return` après les messages d'erreur critiques.

---

#### B15 : Token Counter — Texte vide retourne 1 token

**Fichier** : `src/utils/token_counter.py`

**Problème** : `_heuristic_count("")` retourne `max(1, 0) = 1`. Un texte vide devrait retourner 0 tokens.

**Correctif** : `return len(text) // 4 if text else 0`.

**Attention** : Vérifier que les appelants ne divisent pas par `count_tokens()` (risque de division par zéro).

---

#### B16 : File Utils — TOCTOU et format incohérent

**Fichier** : `src/utils/file_utils.py`

**Problème** :
1. `get_next_sequence_number()` lit le max puis retourne `max + 1` sans atomicité → deux appels concurrents obtiennent le même numéro.
2. Le regex `\d{3}` ne matche pas les numéros >= 1000 produits par `format_sequence_name()`.

**Correctif** :
1. Utiliser un verrou (`threading.Lock`) autour de la lecture/écriture.
2. Changer le regex en `r"^(\d+)_"` et le format en `{:04d}` pour supporter jusqu'à 9999 fichiers.

---

#### B17 : Page Accueil — Message de restauration persistant

**Fichier** : `src/pages/page_accueil.py`

**Problème** : `_restore_message` injecté dans `session_state` n'est jamais nettoyé proprement.

**Correctif** : Utiliser `st.toast()` (éphémère) ou consommer le message immédiatement via un callback.

---

### 2.3 Bugs de Moyenne Gravité (17 bugs) — Impact : Comportement Dégradé ou Erreurs Cosmétiques

Les bugs B18 à B35 concernent principalement :

1. **Parsing JSON fragile** (B30) : La détection des sous-titres en gras dans `ExportEngine` est trop permissive. Le test `count("**") == 2` échoue si le texte contient plusieurs segments bold.

2. **Configuration silencieuse** (B21, B31, B34) : Des erreurs de config (YAML manquant, overlap invalide, checkpoint mal orthographié) sont absorbées sans feedback utilisateur clair.

3. **Fuites mémoire** (B32, B33) : `ActivityLog.entries` croît indéfiniment sans limite ni thread-safety.

4. **Validations manquantes** (B18, B19, B24, B25, B26, B27, B28, B29, B35) : Accès à des valeurs potentiellement `None`, formats de retour non validés, messages d'erreur hardcodés.

---

### 2.4 Bugs de Faible Gravité (7 bugs) — Impact : Cosmétique ou Très Rare

- **B36** : `estimate_pages()` accepte des tokens négatifs.
- **B37** : Edge case overlap chunker avec très peu de phrases.
- **B38** : Double-comptage des tokens économisés dans la déduplication.
- **B39** : Affichage incorrect du marqueur `NEEDS_SOURCE` dans le dashboard.
- **B40** : Variable morte dans la page acquisition.
- **B41** : Initialisation défensive manquante pour `section_summaries`.
- **B42** : Tests en échec par dépendances manquantes.

---

## 3. Plan de Correction Priorisé

### Phase 1 — Correctifs Critiques (Sprint 1 : 3-5 jours)

| Tâche | Bugs | Effort | Risque de Régression |
|-------|------|--------|---------------------|
| Refactoring `MessageBus` : unifier la stratégie de verrouillage | B01, B02 | 2j | MOYEN — tester tous les agents et le WebSocket |
| Ajout `shutdown(wait=True)` dans le pipelining orchestrator | B03 | 0.5j | FAIBLE |
| Initialisation du verrou séquentiel dans `__init__` | B04 | 0.5j | NUL |

### Phase 2 — Correctifs Haute Gravité (Sprint 2 : 5-7 jours)

| Tâche | Bugs | Effort | Risque de Régression |
|-------|------|--------|---------------------|
| Thread-safety `GeminiCacheManager` | B05 | 0.5j | FAIBLE |
| Propagation erreurs sync/async `CorpusAcquirer` | B06 | 1j | MOYEN |
| Guard clauses et null-checks pages Streamlit | B10-B14, B17 | 2j | FAIBLE |
| Cohérence API `ProvidersRegistry` | B08, B09 | 0.5j | MOYEN — vérifier tous les appelants |
| `getattr` défensif API Projects | B07 | 0.25j | NUL |
| Fix `token_counter` texte vide | B15 | 0.25j | FAIBLE — vérifier les divisions |
| Fix `file_utils` regex + TOCTOU | B16 | 1j | MOYEN — migration des fichiers existants |

### Phase 3 — Correctifs Moyenne Gravité (Sprint 3 : 3-5 jours)

| Tâche | Bugs | Effort | Risque de Régression |
|-------|------|--------|---------------------|
| Verrouillage et limite `ActivityLog` | B32, B33 | 0.5j | FAIBLE |
| Error handling `config.py` YAML | B31 | 0.5j | FAIBLE |
| Fix parsing bold `ExportEngine` | B30 | 0.5j | FAIBLE |
| Correctifs WebSocket route | B22, B23 | 1j | MOYEN |
| Validations et null-checks restants | B18-B29, B34, B35 | 2j | FAIBLE |

### Phase 4 — Correctifs Faibles + Tests (Sprint 4 : 1-2 jours)

| Tâche | Bugs | Effort | Risque de Régression |
|-------|------|--------|---------------------|
| Correctifs cosmétiques et edge cases | B36-B41 | 0.5j | NUL |
| Ajout dépendances manquantes + tests | B42 | 0.5j | NUL |
| Tests de régression complets | — | 1j | — |

---

## 4. Recommandations Transversales

### 4.1 Thread-Safety
Le projet mélange `asyncio` (coroutines) et `threading` (ThreadPoolExecutor). Il est recommandé de :
- Documenter clairement quelles structures sont accédées par quel paradigme.
- Utiliser `threading.Lock` partout où une structure est partagée entre les deux mondes.
- Éviter d'instancier `asyncio.Lock()` hors d'un event loop.

### 4.2 Gestion d'Erreurs
De nombreuses fonctions absorbent les erreurs silencieusement (retour `None`, `{}`, valeurs par défaut). Recommandation :
- Les erreurs de configuration doivent remonter immédiatement.
- Les erreurs runtime (API, réseau) peuvent être absorbées avec logging, mais un compteur d'erreurs doit être maintenu.

### 4.3 Session State Streamlit
Les pages Streamlit accèdent à des clés de session potentiellement non initialisées. Recommandation :
- Centraliser l'initialisation de toutes les clés dans `app.py` ou un module `state_init.py`.
- Utiliser `st.session_state.setdefault()` plutôt que `.get()` pour garantir l'initialisation.

### 4.4 Tests
- Ajouter les dépendances manquantes (`beautifulsoup4`, `openpyxl`) dans `requirements.txt` ou ajouter des `pytest.importorskip()`.
- Ajouter des tests spécifiques pour les race conditions identifiées (B01-B04, B05, B16).
- Viser un coverage > 80% sur les modules critiques (`orchestrator`, `multi_agent_orchestrator`, `message_bus`).

---

## 5. Métriques de Suivi

| Métrique | Valeur Actuelle | Cible Post-Correction |
|----------|-----------------|----------------------|
| Tests passés | 632/637 (99.2%) | 637/637 (100%) |
| Bugs critiques | 4 | 0 |
| Bugs haute gravité | 13 | 0 |
| Bugs moyenne gravité | 17 | 0 |
| Bugs faible gravité | 7 | 0 |
| Total bugs identifiés | **42** | **0** |

---

*Document généré le 2026-02-28 — Revue de code exhaustive du projet Orchestr'IA (No-RAG).*
