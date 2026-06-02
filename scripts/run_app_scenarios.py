#!/usr/bin/env python3
"""Run Zone Z2 APP charging scenarios."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.deterministic import run_app_scenarios

if __name__ == "__main__":
    run_app_scenarios()
