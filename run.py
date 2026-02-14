#!/usr/bin/env python3
"""Script de lancement d'Orchestr'IA."""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parent / "src" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        cwd=Path(__file__).parent,
    )


if __name__ == "__main__":
    main()
