"""Configuration globale des tests pytest."""

import sys
from pathlib import Path

# Ajouter la racine du projet au path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
