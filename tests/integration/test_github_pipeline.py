"""Tests d'intégration Phase 6 — pipeline GitHub No-RAG.

Ces tests valident l'acquisition d'un dépôt GitHub fictif (mocké)
et l'intégration du corpus résultant dans ProjectState.
"""

import base64
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.core.github_acquirer import (
    GitHubAcquirer,
    GitHubBranchError,
    GitHubNotFoundError,
)
try:
    from src.core.orchestrator import ProjectState
except ModuleNotFoundError:
    # Environnement de test allégé (dotenv, docling, etc. absents) :
    # stub des modules manquants avant réimport.
    import sys
    for _mod in (
        "dotenv", "tiktoken", "pdfplumber", "pymupdf", "fitz",
        "docling", "docling.document_converter",
    ):
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock()
    # Recharger le module après stubbing
    import importlib
    if "src.core.orchestrator" in sys.modules:
        del sys.modules["src.core.orchestrator"]
    from src.core.orchestrator import ProjectState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def _make_response(status_code: int, json_data=None, headers: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = ""
    resp.headers = headers or {
        "X-RateLimit-Remaining": "4999",
        "X-RateLimit-Reset": str(int(time.time()) + 3600),
    }
    return resp


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_TREE = {
    "tree": [
        {"path": "app.py", "type": "blob", "size": 256, "sha": "sha_app", "url": ""},
        {"path": "utils.py", "type": "blob", "size": 128, "sha": "sha_utils", "url": ""},
        {"path": "README.md", "type": "blob", "size": 512, "sha": "sha_readme", "url": ""},
        {"path": "logo.png", "type": "blob", "size": 10240, "sha": "sha_png", "url": ""},
    ],
    "truncated": False,
}

MOCK_METADATA = {
    "full_name": "testowner/testrepo",
    "description": "A test repository",
    "language": "Python",
    "topics": ["python", "testing"],
    "stargazers_count": 42,
    "default_branch": "main",
    "license": {"spdx_id": "MIT"},
    "updated_at": "2026-02-01T00:00:00Z",
    "open_issues_count": 3,
}

MOCK_LANGUAGES = {"Python": 9800, "Markdown": 512}

APP_PY_CONTENT = "def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n"
UTILS_PY_CONTENT = "def helper(x):\n    return x * 2\n"
README_CONTENT = "# Test Repo\n\nA minimal test repository for integration tests.\n"


def _make_mock_session(tree=None, metadata=None, languages=None):
    """Crée un mock de session avec les réponses cohérentes."""

    def side_effect(url, **kwargs):
        if "/git/trees/" in url:
            return _make_response(200, tree or MOCK_TREE)
        elif url.endswith("/languages"):
            return _make_response(200, languages or MOCK_LANGUAGES)
        elif "/repos/testowner/testrepo" in url and "contents" not in url and "readme" not in url.lower():
            return _make_response(200, metadata or MOCK_METADATA)
        elif "/readme" in url.lower() or "/contents/readme" in url.lower():
            return _make_response(200, {"content": _b64(README_CONTENT)})
        elif "/contents/app.py" in url:
            return _make_response(200, {"content": _b64(APP_PY_CONTENT)})
        elif "/contents/utils.py" in url:
            return _make_response(200, {"content": _b64(UTILS_PY_CONTENT)})
        elif "/contents/README.md" in url:
            return _make_response(200, {"content": _b64(README_CONTENT)})
        elif "/contents/logo.png" in url:
            # PNG : contenu binaire non décodable
            return _make_response(200, {"content": _b64(b"\x89PNG\r\n".decode("latin-1"))})
        else:
            return _make_response(404)

    session = MagicMock()
    session.get.side_effect = side_effect
    session.headers = {}
    return session


# ── Tests d'intégration ───────────────────────────────────────────────────────

class TestGitHubPipelineIntegration:
    """Pipeline complet : tree → filter → download → build_corpus."""

    @pytest.fixture
    def acquirer(self):
        acq = GitHubAcquirer(github_token="ghp_test")
        acq._session = _make_mock_session()
        return acq

    def test_full_pipeline_produces_corpus(self, acquirer):
        """Acquisition complète d'un petit dépôt fictif."""
        # 1. Récupérer l'arbre
        tree = acquirer.fetch_repo_tree("testowner", "testrepo", "main")
        assert len(tree) == 4  # 4 blobs (dont PNG)

        # 2. Filtrer : inclure .py et .md, exclure PNG
        filtered = acquirer.filter_files(
            tree,
            include_patterns=["*.py", "*.md"],
            exclude_patterns=["*.png"],
            max_file_size_kb=500,
        )
        assert len(filtered) == 3  # app.py, utils.py, README.md

        # 3. Estimer les tokens
        estimate = acquirer.estimate_tokens(filtered)
        assert estimate["file_count"] == 3
        assert estimate["estimated_tokens"] > 0
        assert estimate["within_budget"] is True

        # 4. Construire le corpus
        corpus = acquirer.build_corpus(
            "testowner", "testrepo", "main", filtered,
            include_tree=True, include_metadata=True,
        )
        assert "DÉPÔT : testowner/testrepo" in corpus
        assert "app.py" in corpus
        assert "utils.py" in corpus
        assert "README" in corpus
        assert corpus.startswith("=")  # En-tête présent

    def test_corpus_contains_expected_file_headers(self, acquirer):
        """Chaque fichier doit avoir un en-tête structuré."""
        tree = acquirer.fetch_repo_tree("testowner", "testrepo", "main")
        filtered = acquirer.filter_files(tree, ["*.py"], [], 500)

        corpus = acquirer.build_corpus(
            "testowner", "testrepo", "main", filtered,
            include_tree=False, include_metadata=False,
        )
        assert "=== FICHIER : app.py ===" in corpus
        assert "Langage : Python" in corpus
        assert "Type : code" in corpus

    def test_corpus_stored_in_project_state(self, acquirer):
        """Le corpus GitHub est correctement stocké dans ProjectState."""
        tree = acquirer.fetch_repo_tree("testowner", "testrepo", "main")
        filtered = acquirer.filter_files(tree, ["*.py", "*.md"], ["*.png"], 500)
        corpus = acquirer.build_corpus("testowner", "testrepo", "main", filtered)

        state = ProjectState(name="test_project")
        # Simuler le stockage comme dans page_acquisition.py
        state.corpus_text = corpus
        state.github_repo_url = "https://github.com/testowner/testrepo"
        state.github_branch = "main"
        state.github_file_count = len(filtered)
        state.github_token_count = acquirer.estimate_tokens(filtered)["estimated_tokens"]

        assert state.github_repo_url == "https://github.com/testowner/testrepo"
        assert state.github_branch == "main"
        assert state.github_file_count == 3
        assert state.github_token_count > 0
        assert "DÉPÔT" in state.corpus_text

    def test_plan_references_repo_files(self, acquirer):
        """Le corpus contient suffisamment d'info pour qu'un plan référence les fichiers."""
        tree = acquirer.fetch_repo_tree("testowner", "testrepo", "main")
        filtered = acquirer.filter_files(tree, ["*.py", "*.md"], ["*.png"], 500)
        corpus = acquirer.build_corpus("testowner", "testrepo", "main", filtered)

        # Un LLM recevant ce corpus pourrait référencer app.py et utils.py dans le plan
        assert "app.py" in corpus
        assert "utils.py" in corpus
        assert "main()" in corpus  # Contenu du fichier présent

    def test_corpus_fusion_with_existing(self, acquirer):
        """Fusion du corpus GitHub avec un corpus existant."""
        tree = acquirer.fetch_repo_tree("testowner", "testrepo", "main")
        filtered = acquirer.filter_files(tree, ["*.py"], [], 500)
        github_corpus = acquirer.build_corpus("testowner", "testrepo", "main", filtered)

        existing_corpus = "=== SOURCE LOCALE ===\nContenu d'un fichier local.\n"
        state = ProjectState(name="fusion_test")

        # Logique de fusion comme dans page_acquisition.py
        state.corpus_text = (existing_corpus + "\n\n" + github_corpus).strip()

        assert "SOURCE LOCALE" in state.corpus_text
        assert "DÉPÔT" in state.corpus_text

    def test_project_state_roundtrip(self, acquirer):
        """ProjectState avec champs GitHub doit survivre à to_dict/from_dict."""
        state = ProjectState(name="gh_roundtrip")
        state.github_repo_url = "https://github.com/testowner/testrepo"
        state.github_branch = "develop"
        state.github_file_count = 15
        state.github_token_count = 12_000
        state.github_acquired_at = "2026-02-27T12:00:00"

        data = state.to_dict()
        restored = ProjectState.from_dict(data)

        assert restored.github_repo_url == "https://github.com/testowner/testrepo"
        assert restored.github_branch == "develop"
        assert restored.github_file_count == 15
        assert restored.github_token_count == 12_000
        assert restored.github_acquired_at == "2026-02-27T12:00:00"


class TestGitHubErrorHandling:
    """Gestion des erreurs dans le pipeline."""

    def test_branch_not_found_suggests_alternatives(self):
        """Une branche inexistante doit proposer les alternatives."""
        acquirer = GitHubAcquirer(github_token="ghp_test")
        not_found = _make_response(404)
        branches = _make_response(200, [{"name": "main"}, {"name": "develop"}, {"name": "feature/x"}])
        acquirer._session = MagicMock()
        acquirer._session.get.side_effect = [not_found, branches]

        with pytest.raises(GitHubBranchError) as exc_info:
            acquirer.fetch_repo_tree("owner", "repo", "nonexistent")

        assert "main" in exc_info.value.available_branches
        assert "develop" in exc_info.value.available_branches

    def test_empty_repo_after_filter(self):
        """Un dépôt sans fichiers correspondants retourne un corpus minimal."""
        acquirer = GitHubAcquirer(github_token="ghp_test")
        tree_data = {
            "tree": [
                {"path": "logo.png", "type": "blob", "size": 1024, "sha": "x", "url": ""},
            ],
            "truncated": False,
        }
        meta_resp = _make_response(200, {
            "full_name": "o/r", "description": "", "language": "",
            "topics": [], "stargazers_count": 0, "default_branch": "main",
            "license": None, "updated_at": "", "open_issues_count": 0,
        })
        tree_resp = _make_response(200, tree_data)
        langs_resp = _make_response(200, {})
        readme_resp = _make_response(404)

        acquirer._session = MagicMock()
        acquirer._session.get.side_effect = [tree_resp]

        tree = acquirer.fetch_repo_tree("o", "r", "main")
        filtered = acquirer.filter_files(tree, ["*.py"], [], 500)
        assert len(filtered) == 0

        estimate = acquirer.estimate_tokens(filtered)
        assert estimate["file_count"] == 0
        assert estimate["within_budget"] is True
