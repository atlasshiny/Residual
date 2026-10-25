"""Thin wrapper around run.py using synthetic demo data."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .run import main
except ImportError:  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run import main


if __name__ == "__main__":
    main([])
