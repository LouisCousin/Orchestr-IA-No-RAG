"""Tests unitaires pour le DAG de dépendances (Phase 7 + Phase 8 Circuit Breaker)."""

import pytest

from src.core.multi_agent_orchestrator import build_dag, get_descendants, get_ready_sections


class TestBuildDag:
    def test_no_dependencies(self):
        """Toutes les sections sont indépendantes."""
        deps = {"s01": [], "s02": [], "s03": []}
        dag = build_dag(deps)
        assert dag == deps

    def test_linear_dependencies(self):
        """Dépendances linéaires : s01 → s02 → s03."""
        deps = {"s01": [], "s02": ["s01"], "s03": ["s02"]}
        dag = build_dag(deps)
        assert dag == deps

    def test_complex_dependencies(self):
        """DAG complexe avec branches parallèles."""
        deps = {
            "s01": [],
            "s02": [],
            "s03": ["s01"],
            "s04": ["s01", "s02"],
            "s05": ["s03", "s04"],
        }
        dag = build_dag(deps)
        assert dag == deps

    def test_cycle_detection(self):
        """Cycle direct entre deux sections."""
        deps = {"s01": ["s02"], "s02": ["s01"]}
        with pytest.raises(ValueError, match="dépendances cycliques"):
            build_dag(deps)

    def test_cycle_detection_indirect(self):
        """Cycle indirect : s01 → s02 → s03 → s01."""
        deps = {"s01": ["s03"], "s02": ["s01"], "s03": ["s02"]}
        with pytest.raises(ValueError, match="dépendances cycliques"):
            build_dag(deps)

    def test_empty_dag(self):
        """DAG vide."""
        dag = build_dag({})
        assert dag == {}

    def test_single_section(self):
        """Une seule section."""
        deps = {"s01": []}
        dag = build_dag(deps)
        assert dag == {"s01": []}

    def test_missing_dependency_reference(self):
        """Dépendance vers une section non déclarée — pas de cycle."""
        deps = {"s01": [], "s02": ["s99"]}  # s99 n'existe pas comme clé
        dag = build_dag(deps)  # Ne doit pas lever d'erreur
        assert dag == deps


class TestGetReadySections:
    def test_all_ready(self):
        """Toutes les sections sont prêtes (aucune dépendance)."""
        dag = {"s01": [], "s02": [], "s03": []}
        ready = get_ready_sections(dag, completed=set(), in_progress=set())
        assert set(ready) == {"s01", "s02", "s03"}

    def test_none_ready(self):
        """Aucune section n'est prête (dépendances non résolues)."""
        dag = {"s01": ["s02"], "s02": ["s01"]}
        ready = get_ready_sections(dag, completed=set(), in_progress=set())
        assert ready == []  # Cycle — rien n'est prêt

    def test_partial_ready(self):
        """Certaines sections débloquées après complétion."""
        dag = {
            "s01": [],
            "s02": [],
            "s03": ["s01"],
            "s04": ["s01", "s02"],
        }
        # Initialement, s01 et s02 sont prêtes
        ready = get_ready_sections(dag, completed=set(), in_progress=set())
        assert set(ready) == {"s01", "s02"}

        # Après complétion de s01
        ready = get_ready_sections(dag, completed={"s01"}, in_progress=set())
        assert "s02" in ready
        assert "s03" in ready  # s03 dépend seulement de s01

        # s04 n'est pas prête (attend s02)
        assert "s04" not in ready

    def test_excludes_in_progress(self):
        """Les sections en cours ne sont pas relancées."""
        dag = {"s01": [], "s02": []}
        ready = get_ready_sections(dag, completed=set(), in_progress={"s01"})
        assert ready == ["s02"]

    def test_excludes_completed(self):
        """Les sections complétées ne sont pas relancées."""
        dag = {"s01": [], "s02": []}
        ready = get_ready_sections(dag, completed={"s01"}, in_progress=set())
        assert ready == ["s02"]

    def test_all_completed(self):
        """Toutes les sections sont complétées."""
        dag = {"s01": [], "s02": []}
        ready = get_ready_sections(dag, completed={"s01", "s02"}, in_progress=set())
        assert ready == []

    def test_sequential_unlock(self):
        """Déblocage séquentiel : s01 → s02 → s03."""
        dag = {"s01": [], "s02": ["s01"], "s03": ["s02"]}

        # Étape 1 : seul s01 est prêt
        ready = get_ready_sections(dag, completed=set(), in_progress=set())
        assert ready == ["s01"]

        # Étape 2 : s01 terminé → s02 débloqué
        ready = get_ready_sections(dag, completed={"s01"}, in_progress=set())
        assert ready == ["s02"]

        # Étape 3 : s02 terminé → s03 débloqué
        ready = get_ready_sections(dag, completed={"s01", "s02"}, in_progress=set())
        assert ready == ["s03"]


class TestGetDescendants:
    """Phase 8 — Circuit Breaker : parcours du graphe pour annulation en cascade."""

    def test_no_descendants(self):
        """Un nœud feuille n'a aucun descendant."""
        dag = {"s01": [], "s02": ["s01"], "s03": ["s02"]}
        descendants = get_descendants(dag, "s03")
        assert descendants == set()

    def test_direct_descendants(self):
        """Descendants directs d'un nœud racine."""
        dag = {"s01": [], "s02": ["s01"], "s03": ["s01"]}
        descendants = get_descendants(dag, "s01")
        assert descendants == {"s02", "s03"}

    def test_transitive_descendants(self):
        """Descendants transitifs : s01 → s02 → s03."""
        dag = {"s01": [], "s02": ["s01"], "s03": ["s02"]}
        descendants = get_descendants(dag, "s01")
        assert descendants == {"s02", "s03"}

    def test_complex_dag(self):
        """DAG complexe avec branches parallèles.

        s01 ──┬── s03 ──┐
              │          ├── s05
        s02 ──┴── s04 ──┘
        """
        dag = {
            "s01": [],
            "s02": [],
            "s03": ["s01"],
            "s04": ["s01", "s02"],
            "s05": ["s03", "s04"],
        }
        # Si s01 échoue, s03, s04 et s05 sont des descendants
        descendants = get_descendants(dag, "s01")
        assert descendants == {"s03", "s04", "s05"}

        # Si s02 échoue, s04 et s05 sont des descendants
        descendants = get_descendants(dag, "s02")
        assert descendants == {"s04", "s05"}

        # Si s03 échoue, seul s05 est descendant
        descendants = get_descendants(dag, "s03")
        assert descendants == {"s05"}

    def test_empty_dag(self):
        """DAG vide — aucun descendant."""
        descendants = get_descendants({}, "s01")
        assert descendants == set()

    def test_single_node(self):
        """Un seul nœud sans dépendances."""
        dag = {"s01": []}
        descendants = get_descendants(dag, "s01")
        assert descendants == set()

    def test_unknown_node(self):
        """Nœud inconnu — aucun descendant (pas d'erreur)."""
        dag = {"s01": [], "s02": ["s01"]}
        descendants = get_descendants(dag, "s99")
        assert descendants == set()

    def test_diamond_dag(self):
        """DAG en diamant : s01 → s02, s03 → s04.

        s01 ──┬── s02 ──┐
              │          ├── s04
              └── s03 ──┘
        """
        dag = {
            "s01": [],
            "s02": ["s01"],
            "s03": ["s01"],
            "s04": ["s02", "s03"],
        }
        # Si s01 échoue, tout le reste est touché
        descendants = get_descendants(dag, "s01")
        assert descendants == {"s02", "s03", "s04"}

        # Si s02 échoue, seul s04 est descendant
        descendants = get_descendants(dag, "s02")
        assert descendants == {"s04"}

    def test_deep_chain(self):
        """Chaîne profonde : s01 → s02 → s03 → s04 → s05."""
        dag = {
            "s01": [],
            "s02": ["s01"],
            "s03": ["s02"],
            "s04": ["s03"],
            "s05": ["s04"],
        }
        descendants = get_descendants(dag, "s01")
        assert descendants == {"s02", "s03", "s04", "s05"}

        descendants = get_descendants(dag, "s03")
        assert descendants == {"s04", "s05"}

    def test_no_api_calls_for_cancelled_sections(self):
        """Critère d'acceptation : les sections filles ne doivent pas être lancées.

        Vérifie que get_descendants identifie correctement toutes les sections
        qui doivent être annulées, empêchant tout appel API.
        """
        # Scénario : section 1.0 échoue, 1.1 et 1.2 en dépendent
        dag = {
            "1.0": [],
            "1.1": ["1.0"],
            "1.2": ["1.0"],
            "2.0": [],
            "2.1": ["2.0"],
        }
        cancelled = get_descendants(dag, "1.0")
        # Seules 1.1 et 1.2 doivent être annulées, pas 2.0 ni 2.1
        assert cancelled == {"1.1", "1.2"}
        assert "2.0" not in cancelled
        assert "2.1" not in cancelled
