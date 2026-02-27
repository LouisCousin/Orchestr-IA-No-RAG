"""Acquisition de dépôts GitHub via l'API REST (sans git clone).

Phase 6 No-RAG : récupération des fichiers via /git/trees et /contents,
décodage base64, assemblage en corpus texte structuré injecté directement
dans la fenêtre de contexte LLM.
"""

import base64
import fnmatch
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger("orchestria")

# Budget tokens par défaut (400 000 tokens → réserve output)
DEFAULT_TOKEN_BUDGET = 400_000

# Taux approximatif : 1 token ≈ 4 octets de texte source
BYTES_PER_TOKEN = 4


class GitHubAuthError(Exception):
    """Dépôt privé ou token invalide."""


class GitHubNotFoundError(Exception):
    """Dépôt ou branche introuvable."""


class GitHubBranchError(Exception):
    """Branche inexistante — liste des branches disponibles fournie."""

    def __init__(self, message: str, available_branches: list[str] | None = None):
        super().__init__(message)
        self.available_branches = available_branches or []


class GitHubRateLimitError(Exception):
    """Rate limit dépassé."""


class GitHubAcquirer:
    """Acquisition de dépôts GitHub via l'API REST (sans git clone)."""

    BASE_URL = "https://api.github.com"

    def __init__(self, github_token: str | None = None):
        """Initialise le client avec un token optionnel (dépôts privés).

        Lit GITHUB_TOKEN depuis l'environnement si non fourni.
        Configure le rate limiting automatique (core: 5000 req/h authentifié).
        """
        token = github_token or os.environ.get("GITHUB_TOKEN")
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
        self._rate_limit_remaining: int = 60
        self._rate_limit_reset: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Méthodes publiques
    # ──────────────────────────────────────────────────────────────────────

    def fetch_repo_tree(
        self,
        owner: str,
        repo: str,
        branch: str = "main",
    ) -> list[dict]:
        """Récupère l'arbre complet du dépôt via /git/trees?recursive=1.

        Retourne une liste de dict : {path, type, size, sha, url}.
        Gère la pagination automatique pour les grands dépôts (>100 000 fichiers).
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        try:
            resp = self._get(url)
        except GitHubNotFoundError:
            # Branche peut-être invalide — récupérer les branches disponibles
            branches = self._list_branches(owner, repo)
            raise GitHubBranchError(
                f"Branche '{branch}' introuvable dans {owner}/{repo}.",
                available_branches=branches,
            )

        data = resp.json()
        tree = [
            {
                "path": item["path"],
                "type": item["type"],   # "blob" ou "tree"
                "size": item.get("size", 0),
                "sha": item["sha"],
                "url": item.get("url", ""),
            }
            for item in data.get("tree", [])
            if item["type"] == "blob"
        ]

        # GitHub indique truncated=true si le dépôt est très grand
        if data.get("truncated"):
            logger.warning(
                "L'arbre du dépôt est tronqué (>100 000 objets). "
                "Certains fichiers peuvent être manquants."
            )
        return tree

    def filter_files(
        self,
        tree: list[dict],
        include_patterns: list[str],
        exclude_patterns: list[str],
        max_file_size_kb: int = 500,
    ) -> list[dict]:
        """Filtre l'arbre selon les patterns (fnmatch) et la taille max.

        Exclut automatiquement les fichiers binaires (pas de décodage UTF-8 possible).
        Retourne la liste filtrée avec estimation des tokens.
        """
        max_bytes = max_file_size_kb * 1024
        result = []

        for item in tree:
            path = item["path"]
            size = item.get("size", 0)

            # Exclusion par taille
            if size > max_bytes:
                continue

            # Exclusion par pattern
            if self._matches_any(path, exclude_patterns):
                continue

            # Inclusion par pattern (si liste non vide)
            if include_patterns and not self._matches_any(path, include_patterns):
                continue

            # Estimation tokens de ce fichier
            estimated_tokens = max(1, size // BYTES_PER_TOKEN)
            result.append({**item, "estimated_tokens": estimated_tokens})

        return result

    def fetch_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        branch: str = "main",
    ) -> str | None:
        """Télécharge le contenu d'un fichier via /contents/{path}.

        Décode le base64 retourné par l'API GitHub.
        Retourne None si le fichier est binaire ou illisible (UTF-8/latin-1 fallback).
        Respecte le rate limit : attente automatique si X-RateLimit-Remaining < 10.
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}?ref={branch}"
        try:
            resp = self._get(url)
        except (GitHubNotFoundError, GitHubAuthError):
            return None

        data = resp.json()
        content_b64 = data.get("content", "")
        if not content_b64:
            return None

        # Décoder le base64 (GitHub insère des \n)
        try:
            raw_bytes = base64.b64decode(content_b64.replace("\n", ""))
        except Exception:
            logger.warning("Impossible de décoder base64 pour %s", path)
            return None

        # Tenter le décodage texte
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                return raw_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        logger.warning("Fichier binaire ignoré : %s", path)
        return None

    def fetch_repo_metadata(
        self,
        owner: str,
        repo: str,
    ) -> dict:
        """Récupère les métadonnées du dépôt via /repos/{owner}/{repo}.

        Retourne : description, topics, language, stargazers_count,
                   default_branch, license, updated_at, open_issues_count.
        Effectue un second appel /languages pour la répartition détaillée.
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        resp = self._get(url)
        data = resp.json()

        # Langages détaillés
        languages: dict = {}
        try:
            lang_resp = self._get(f"{self.BASE_URL}/repos/{owner}/{repo}/languages")
            languages = lang_resp.json()
        except Exception:
            pass

        # Calcul des pourcentages
        total_bytes = sum(languages.values()) or 1
        lang_pct = {
            lang: round(100 * count / total_bytes, 1)
            for lang, count in sorted(languages.items(), key=lambda x: -x[1])
        }

        license_info = data.get("license") or {}
        return {
            "full_name": data.get("full_name", f"{owner}/{repo}"),
            "description": data.get("description") or "",
            "topics": data.get("topics", []),
            "language": data.get("language") or "",
            "languages": lang_pct,
            "stargazers_count": data.get("stargazers_count", 0),
            "default_branch": data.get("default_branch", "main"),
            "license": license_info.get("spdx_id") or license_info.get("name") or "",
            "updated_at": data.get("updated_at") or "",
            "open_issues_count": data.get("open_issues_count", 0),
        }

    def fetch_readme(
        self,
        owner: str,
        repo: str,
        branch: str = "main",
    ) -> str | None:
        """Récupère le README via /readme (endpoint dédié, insensible à la casse).

        Fallback : cherche README.md, README.rst, README.txt à la racine.
        Retourne le contenu décodé ou None si absent.
        """
        # Endpoint dédié GitHub
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/readme?ref={branch}"
        try:
            resp = self._get(url)
            data = resp.json()
            content_b64 = data.get("content", "")
            if content_b64:
                try:
                    return base64.b64decode(content_b64.replace("\n", "")).decode("utf-8")
                except Exception:
                    pass
        except (GitHubNotFoundError, GitHubAuthError):
            pass

        # Fallback : noms courants
        for name in ("README.md", "README.rst", "README.txt", "readme.md"):
            content = self.fetch_file_content(owner, repo, name, branch)
            if content:
                return content

        return None

    def build_corpus(
        self,
        owner: str,
        repo: str,
        branch: str,
        filtered_files: list[dict],
        include_tree: bool = True,
        include_metadata: bool = True,
    ) -> str:
        """Assemble le corpus texte complet prêt pour injection dans le contexte LLM.

        Structure : [REPO_HEADER] + [TREE] + [README] + [FILE_1] + ... + [FILE_N].
        Retourne une chaîne UTF-8 unique.
        """
        parts: list[str] = []

        # ── En-tête du dépôt ──
        if include_metadata:
            try:
                meta = self.fetch_repo_metadata(owner, repo)
            except Exception:
                meta = {"full_name": f"{owner}/{repo}", "description": "", "languages": {}}

            lang_str = ", ".join(
                f"{lang} {pct}%" for lang, pct in list(meta.get("languages", {}).items())[:3]
            ) or meta.get("language", "")

            header_lines = [
                "=" * 40,
                f"DÉPÔT : {meta.get('full_name', f'{owner}/{repo}')}",
                f"Branche : {branch}" + (f" | Langages : {lang_str}" if lang_str else ""),
            ]
            if meta.get("description"):
                header_lines.append(f"Description : {meta['description']}")
            if meta.get("topics"):
                header_lines.append(f"Topics : {', '.join(meta['topics'])}")
            header_lines.append("=" * 40)
            parts.append("\n".join(header_lines))
        else:
            parts.append(f"{'=' * 40}\nDÉPÔT : {owner}/{repo}\nBranche : {branch}\n{'=' * 40}")

        # ── Arbre du dépôt ──
        if include_tree and filtered_files:
            tree_text = self._build_tree_text(filtered_files)
            parts.append(f"\n=== STRUCTURE DU DÉPÔT ===\n{tree_text}")

        # ── README en tête ──
        readme_paths = {f["path"].lower() for f in filtered_files}
        readme_content = None
        readme_path = None
        for candidate in ("readme.md", "readme.rst", "readme.txt"):
            if candidate in readme_paths:
                readme_path = next(
                    f["path"] for f in filtered_files if f["path"].lower() == candidate
                )
                readme_content = self.fetch_file_content(owner, repo, readme_path, branch)
                break
        if readme_content is None:
            readme_content = self.fetch_readme(owner, repo, branch)

        if readme_content:
            lines = readme_content.splitlines()
            parts.append(
                f"\n=== FICHIER : README ===\n"
                f"Langage : Markdown | Type : documentation | Lignes : {len(lines)}\n"
                f"{'-' * 40}\n{readme_content}"
            )

        # ── Fichiers (téléchargement parallèle, 5 workers max) ──
        # Exclure le README déjà traité
        other_files = [
            f for f in filtered_files
            if readme_path is None or f["path"] != readme_path
        ]

        file_contents: dict[str, str | None] = {}
        if other_files:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_path = {
                    executor.submit(
                        self.fetch_file_content, owner, repo, f["path"], branch
                    ): f["path"]
                    for f in other_files
                }
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        file_contents[path] = future.result()
                    except Exception as exc:
                        logger.warning("Erreur téléchargement %s : %s", path, exc)
                        file_contents[path] = None

        for f in other_files:
            path = f["path"]
            content = file_contents.get(path)
            if content is None:
                continue
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            lang = self._ext_to_language(ext)
            lines = content.splitlines()
            doc_type = "documentation" if ext in ("md", "rst", "txt") else "code"
            parts.append(
                f"\n=== FICHIER : {path} ===\n"
                f"Langage : {lang} | Type : {doc_type} | Lignes : {len(lines)}\n"
                f"{'-' * 40}\n{content}"
            )

        return "\n".join(parts)

    def estimate_tokens(
        self,
        filtered_files: list[dict],
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> dict:
        """Estime le nombre de tokens du corpus avant téléchargement complet.

        Utilise les tailles en octets (size) de l'arbre pour une estimation rapide.
        Retourne : {estimated_tokens, file_count, total_size_kb, within_budget: bool}.
        """
        total_bytes = sum(f.get("size", 0) for f in filtered_files)
        estimated_tokens = max(1, total_bytes // BYTES_PER_TOKEN)
        return {
            "estimated_tokens": estimated_tokens,
            "file_count": len(filtered_files),
            "total_size_kb": round(total_bytes / 1024, 1),
            "within_budget": estimated_tokens <= token_budget,
            "budget_pct": round(100 * estimated_tokens / token_budget, 1) if token_budget else 0,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Méthodes privées
    # ──────────────────────────────────────────────────────────────────────

    def _get(self, url: str) -> requests.Response:
        """Effectue un GET avec gestion rate limit et erreurs HTTP."""
        self._wait_if_rate_limited()
        try:
            resp = self._session.get(url, timeout=10)
        except requests.Timeout:
            raise requests.Timeout(f"Timeout lors de la requête : {url}")

        # Mise à jour du rate limit
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining is not None:
            self._rate_limit_remaining = int(remaining)
        if reset is not None:
            self._rate_limit_reset = int(reset)

        if resp.status_code == 401:
            raise GitHubAuthError(
                "Dépôt privé ou token invalide — configurez GITHUB_TOKEN dans .env"
            )
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            wait_sec = max(0, self._rate_limit_reset - int(time.time()))
            raise GitHubRateLimitError(
                f"Limite GitHub atteinte, reprise dans {wait_sec}s"
            )
        if resp.status_code == 404:
            raise GitHubNotFoundError(f"Ressource introuvable : {url}")
        if resp.status_code >= 400:
            raise requests.HTTPError(f"HTTP {resp.status_code} pour {url}", response=resp)

        return resp

    def _wait_if_rate_limited(self) -> None:
        """Attend si le quota restant passe sous 10 requêtes."""
        if self._rate_limit_remaining < 10 and self._rate_limit_reset > 0:
            wait_sec = max(0, self._rate_limit_reset - int(time.time())) + 1
            logger.warning(
                "Rate limit GitHub : %d requêtes restantes. Attente de %ds.",
                self._rate_limit_remaining,
                wait_sec,
            )
            time.sleep(wait_sec)
            self._rate_limit_remaining = 60  # reset optimiste

    def _list_branches(self, owner: str, repo: str) -> list[str]:
        """Retourne la liste des branches disponibles pour un dépôt."""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/branches"
        try:
            resp = self._get(url)
            return [b["name"] for b in resp.json()]
        except Exception:
            return []

    @staticmethod
    def _matches_any(path: str, patterns: list[str]) -> bool:
        """Vérifie si le chemin correspond à l'un des patterns fnmatch."""
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
            # Aussi tester le nom de fichier seul (sans chemin)
            filename = path.split("/")[-1]
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    @staticmethod
    def _build_tree_text(files: list[dict]) -> str:
        """Génère une représentation textuelle de l'arbre du dépôt."""
        paths = sorted(f["path"] for f in files)
        lines = []
        for path in paths:
            depth = path.count("/")
            name = path.split("/")[-1]
            indent = "│   " * depth
            lines.append(f"{indent}├── {name}")
        return "\n".join(lines)

    @staticmethod
    def _ext_to_language(ext: str) -> str:
        """Mappe une extension de fichier vers un nom de langage lisible."""
        mapping = {
            "py": "Python", "js": "JavaScript", "ts": "TypeScript",
            "tsx": "TypeScript/React", "jsx": "JavaScript/React",
            "java": "Java", "go": "Go", "rs": "Rust",
            "c": "C", "cpp": "C++", "h": "C/C++ Header",
            "rb": "Ruby", "php": "PHP", "swift": "Swift", "kt": "Kotlin",
            "md": "Markdown", "rst": "reStructuredText", "txt": "Text",
            "yaml": "YAML", "yml": "YAML", "json": "JSON", "toml": "TOML",
            "sh": "Shell", "dockerfile": "Dockerfile",
        }
        return mapping.get(ext.lower(), ext.upper() if ext else "Unknown")
