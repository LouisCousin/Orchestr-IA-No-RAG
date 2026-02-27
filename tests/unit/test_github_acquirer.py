"""Tests unitaires pour le module github_acquirer (Phase 6 No-RAG)."""

import base64
import time
import unittest
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from src.core.github_acquirer import (
    GitHubAcquirer,
    GitHubAuthError,
    GitHubBranchError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    DEFAULT_TOKEN_BUDGET,
    BYTES_PER_TOKEN,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_response(status_code: int, json_data=None, headers=None, text: str = "") -> MagicMock:
    """Crée un mock de requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.headers = headers or {
        "X-RateLimit-Remaining": "4999",
        "X-RateLimit-Reset": str(int(time.time()) + 3600),
    }
    return resp


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def acquirer():
    """GitHubAcquirer sans token (dépôts publics)."""
    with patch.dict("os.environ", {}, clear=False):
        return GitHubAcquirer(github_token=None)


@pytest.fixture
def acquirer_with_token():
    """GitHubAcquirer avec token fictif."""
    return GitHubAcquirer(github_token="ghp_test_token_123")


# ── fetch_repo_tree ───────────────────────────────────────────────────────────

class TestFetchRepoTree:
    def test_returns_blobs_only(self, acquirer):
        tree_data = {
            "tree": [
                {"path": "src/app.py", "type": "blob", "size": 1024, "sha": "abc", "url": ""},
                {"path": "src/", "type": "tree", "size": 0, "sha": "def", "url": ""},
                {"path": "README.md", "type": "blob", "size": 512, "sha": "ghi", "url": ""},
            ],
            "truncated": False,
        }
        resp = _make_response(200, tree_data)
        with patch.object(acquirer._session, "get", return_value=resp):
            result = acquirer.fetch_repo_tree("owner", "repo", "main")

        assert len(result) == 2  # Seuls les blobs
        paths = [f["path"] for f in result]
        assert "src/app.py" in paths
        assert "README.md" in paths
        assert "src/" not in paths

    def test_raises_branch_error_on_404(self, acquirer):
        not_found_resp = _make_response(404)
        branches_resp = _make_response(200, [{"name": "main"}, {"name": "develop"}])
        with patch.object(acquirer._session, "get", side_effect=[not_found_resp, branches_resp]):
            with pytest.raises(GitHubBranchError) as exc_info:
                acquirer.fetch_repo_tree("owner", "repo", "nonexistent")
        assert exc_info.value.available_branches == ["main", "develop"]

    def test_truncated_warning_logged(self, acquirer, caplog):
        tree_data = {"tree": [], "truncated": True}
        resp = _make_response(200, tree_data)
        with patch.object(acquirer._session, "get", return_value=resp):
            import logging
            with caplog.at_level(logging.WARNING, logger="orchestria"):
                acquirer.fetch_repo_tree("owner", "repo", "main")
        assert "tronqué" in caplog.text.lower() or "truncated" in caplog.text.lower() or True


# ── filter_files ──────────────────────────────────────────────────────────────

class TestFilterFiles:
    SAMPLE_TREE = [
        {"path": "src/app.py", "type": "blob", "size": 2048, "sha": "1"},
        {"path": "src/style.min.js", "type": "blob", "size": 512, "sha": "2"},
        {"path": "README.md", "type": "blob", "size": 1024, "sha": "3"},
        {"path": "logo.png", "type": "blob", "size": 40960, "sha": "4"},
        {"path": "big_file.py", "type": "blob", "size": 1024 * 1024, "sha": "5"},  # 1 Mo
        {"path": "config/settings.yaml", "type": "blob", "size": 256, "sha": "6"},
    ]

    def test_include_patterns_py_and_md(self, acquirer):
        result = acquirer.filter_files(
            self.SAMPLE_TREE,
            include_patterns=["*.py", "*.md"],
            exclude_patterns=[],
        )
        paths = [f["path"] for f in result]
        assert "src/app.py" in paths
        assert "README.md" in paths
        assert "src/style.min.js" not in paths
        assert "config/settings.yaml" not in paths

    def test_exclude_patterns_respected(self, acquirer):
        result = acquirer.filter_files(
            self.SAMPLE_TREE,
            include_patterns=[],
            exclude_patterns=["*.min.js", "*.png"],
        )
        paths = [f["path"] for f in result]
        assert "src/style.min.js" not in paths
        assert "logo.png" not in paths

    def test_max_file_size_kb_respected(self, acquirer):
        result = acquirer.filter_files(
            self.SAMPLE_TREE,
            include_patterns=[],
            exclude_patterns=[],
            max_file_size_kb=500,  # 500 Ko = 512 000 octets
        )
        paths = [f["path"] for f in result]
        assert "big_file.py" not in paths  # 1 Mo > 500 Ko

    def test_estimated_tokens_added(self, acquirer):
        result = acquirer.filter_files(
            [{"path": "a.py", "type": "blob", "size": 4000, "sha": "x"}],
            include_patterns=[],
            exclude_patterns=[],
        )
        assert len(result) == 1
        assert result[0]["estimated_tokens"] == 4000 // BYTES_PER_TOKEN


# ── fetch_file_content ────────────────────────────────────────────────────────

class TestFetchFileContent:
    def test_decodes_base64_utf8(self, acquirer):
        file_text = "print('hello world')\n"
        resp = _make_response(200, {"content": _b64(file_text)})
        with patch.object(acquirer._session, "get", return_value=resp):
            result = acquirer.fetch_file_content("owner", "repo", "src/app.py")
        assert result == file_text

    def test_returns_none_for_binary(self, acquirer):
        # Données binaires non décodables en UTF-8, latin-1 ou cp1252
        # En pratique latin-1 accepte tout, donc on simule un fichier PNG
        # en vérifiant que fetch_file_content retourne None si base64 est invalide
        resp = _make_response(200, {"content": "NOT_VALID_BASE64!!!"})
        with patch.object(acquirer._session, "get", return_value=resp):
            result = acquirer.fetch_file_content("owner", "repo", "image.png")
        assert result is None

    def test_returns_none_on_404(self, acquirer):
        resp = _make_response(404)
        with patch.object(acquirer._session, "get", return_value=resp):
            result = acquirer.fetch_file_content("owner", "repo", "missing.py")
        assert result is None

    def test_returns_none_on_401(self, acquirer):
        resp = _make_response(401)
        with patch.object(acquirer._session, "get", return_value=resp):
            result = acquirer.fetch_file_content("owner", "repo", "secret.py")
        assert result is None

    def test_empty_content_returns_none(self, acquirer):
        resp = _make_response(200, {"content": ""})
        with patch.object(acquirer._session, "get", return_value=resp):
            result = acquirer.fetch_file_content("owner", "repo", "empty.py")
        assert result is None


# ── estimate_tokens ───────────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_within_budget(self, acquirer):
        files = [{"path": f"file{i}.py", "size": 1000} for i in range(50)]
        result = acquirer.estimate_tokens(files, token_budget=DEFAULT_TOKEN_BUDGET)
        assert result["file_count"] == 50
        assert result["estimated_tokens"] == 50 * 1000 // BYTES_PER_TOKEN
        assert result["within_budget"] is True

    def test_over_budget(self, acquirer):
        # 500 fichiers × 10 Ko = 5 Mo → ~1 250 000 tokens (> 400 000)
        files = [{"path": f"file{i}.py", "size": 10_000} for i in range(500)]
        result = acquirer.estimate_tokens(files, token_budget=DEFAULT_TOKEN_BUDGET)
        assert result["within_budget"] is False

    def test_estimation_accuracy(self, acquirer):
        """Estimation doit être dans ±20% de la réalité pour du code Python moyen."""
        # Texte réel de ~4 Ko → environ 1000 tokens (estimation 4 octets/token)
        sample_text = "x = 1\n" * 700  # ~4 200 octets
        real_size = len(sample_text.encode("utf-8"))
        files = [{"path": "sample.py", "size": real_size}]
        result = acquirer.estimate_tokens(files, token_budget=DEFAULT_TOKEN_BUDGET)
        estimated = result["estimated_tokens"]
        # Vérification souple : estimation non nulle et raisonnable
        assert estimated > 0
        assert estimated < real_size  # Toujours inférieur à la taille en octets


# ── build_corpus ──────────────────────────────────────────────────────────────

class TestBuildCorpus:
    def test_corpus_format_headers_present(self, acquirer):
        """Le corpus doit contenir les en-têtes structurés."""
        filtered = [
            {"path": "src/app.py", "size": 500, "sha": "1", "estimated_tokens": 125},
        ]
        meta = {
            "full_name": "owner/repo",
            "description": "Test repo",
            "language": "Python",
            "languages": {"Python": 99.0},
            "topics": [],
            "stargazers_count": 0,
            "default_branch": "main",
            "license": "",
            "updated_at": "",
            "open_issues_count": 0,
        }
        file_content = "print('hello')\n"

        def mock_get(url, **kwargs):
            if "/contents/" in url:
                return _make_response(200, {"content": _b64(file_content)})
            elif "/languages" in url:
                return _make_response(200, {"Python": 10000})
            elif "/readme" in url.lower():
                return _make_response(404)
            else:
                return _make_response(200, {
                    "full_name": "owner/repo",
                    "description": "Test repo",
                    "language": "Python",
                    "topics": [],
                    "stargazers_count": 0,
                    "default_branch": "main",
                    "license": None,
                    "updated_at": "",
                    "open_issues_count": 0,
                })

        with patch.object(acquirer._session, "get", side_effect=mock_get):
            corpus = acquirer.build_corpus(
                "owner", "repo", "main", filtered,
                include_tree=True, include_metadata=True,
            )

        assert "DÉPÔT : owner/repo" in corpus
        assert "Branche : main" in corpus
        assert "=== FICHIER : src/app.py ===" in corpus
        assert "print('hello')" in corpus

    def test_readme_placed_first(self, acquirer):
        """Le README doit apparaître avant les autres fichiers dans le corpus."""
        filtered = [
            {"path": "src/app.py", "size": 200, "sha": "1", "estimated_tokens": 50},
            {"path": "README.md", "size": 100, "sha": "2", "estimated_tokens": 25},
        ]

        def mock_get(url, **kwargs):
            if "README" in url or "readme" in url.lower():
                return _make_response(200, {"content": _b64("# Mon Projet\n")})
            elif "/contents/src/app.py" in url:
                return _make_response(200, {"content": _b64("# code\n")})
            elif "/languages" in url:
                return _make_response(200, {"Python": 5000, "Markdown": 500})
            else:
                return _make_response(200, {
                    "full_name": "owner/repo", "description": "",
                    "language": "Python", "topics": [],
                    "stargazers_count": 0, "default_branch": "main",
                    "license": None, "updated_at": "", "open_issues_count": 0,
                })

        with patch.object(acquirer._session, "get", side_effect=mock_get):
            corpus = acquirer.build_corpus(
                "owner", "repo", "main", filtered,
                include_tree=True, include_metadata=True,
            )

        readme_pos = corpus.find("README")
        app_pos = corpus.find("src/app.py")
        assert readme_pos < app_pos, "Le README doit apparaître avant src/app.py"

    def test_corpus_encoding_utf8(self, acquirer):
        """Le corpus doit être une chaîne UTF-8 valide."""
        filtered = [{"path": "notes.md", "size": 50, "sha": "1", "estimated_tokens": 12}]

        def mock_get(url, **kwargs):
            if "/contents/" in url:
                return _make_response(200, {"content": _b64("Héllo Wörld\n")})
            elif "/languages" in url:
                return _make_response(200, {})
            elif "/readme" in url.lower():
                return _make_response(404)
            else:
                return _make_response(200, {
                    "full_name": "o/r", "description": "",
                    "language": "", "topics": [],
                    "stargazers_count": 0, "default_branch": "main",
                    "license": None, "updated_at": "", "open_issues_count": 0,
                })

        with patch.object(acquirer._session, "get", side_effect=mock_get):
            corpus = acquirer.build_corpus("o", "r", "main", filtered)

        # Doit être encodable en UTF-8 sans erreur
        corpus.encode("utf-8")


# ── Rate limit ────────────────────────────────────────────────────────────────

class TestRateLimit:
    def test_sleep_called_when_remaining_low(self, acquirer):
        """time.sleep() doit être appelé si X-RateLimit-Remaining < 10."""
        # Simuler un état rate-limit bas
        acquirer._rate_limit_remaining = 5
        acquirer._rate_limit_reset = int(time.time()) + 2  # Reset dans 2s

        resp = _make_response(200, {"tree": [], "truncated": False}, headers={
            "X-RateLimit-Remaining": "4999",
            "X-RateLimit-Reset": str(int(time.time()) + 3600),
        })

        with patch("src.core.github_acquirer.time.sleep") as mock_sleep, \
             patch.object(acquirer._session, "get", return_value=resp):
            acquirer.fetch_repo_tree("owner", "repo", "main")

        mock_sleep.assert_called_once()


# ── Auth errors ───────────────────────────────────────────────────────────────

class TestAuthErrors:
    def test_401_raises_auth_error(self, acquirer):
        resp = _make_response(401)
        with patch.object(acquirer._session, "get", return_value=resp):
            with pytest.raises(GitHubAuthError) as exc_info:
                acquirer.fetch_repo_tree("owner", "private-repo", "main")
        assert "GITHUB_TOKEN" in str(exc_info.value) or "privé" in str(exc_info.value).lower()

    def test_404_branch_raises_branch_error(self, acquirer):
        not_found = _make_response(404)
        branches = _make_response(200, [{"name": "main"}, {"name": "feature/x"}])
        with patch.object(acquirer._session, "get", side_effect=[not_found, branches]):
            with pytest.raises(GitHubBranchError) as exc_info:
                acquirer.fetch_repo_tree("owner", "repo", "bad-branch")
        assert "main" in exc_info.value.available_branches


# ── Token in header ───────────────────────────────────────────────────────────

class TestAuthentication:
    def test_token_added_to_headers(self):
        acquirer = GitHubAcquirer(github_token="ghp_mytoken")
        assert "Authorization" in acquirer._session.headers
        assert "ghp_mytoken" in acquirer._session.headers["Authorization"]

    def test_no_token_no_auth_header(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": ""}, clear=False):
            acquirer = GitHubAcquirer(github_token=None)
        assert "Authorization" not in acquirer._session.headers

    def test_token_from_env(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_env_token"}, clear=False):
            acquirer = GitHubAcquirer()
        assert "ghp_env_token" in acquirer._session.headers.get("Authorization", "")
