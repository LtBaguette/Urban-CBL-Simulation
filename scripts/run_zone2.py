#!/usr/bin/env python3
"""Run Zone Z2 intervention simulation (10/15/20% peak shift)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.deterministic import run_zone2_interventions

if __name__ == "__main__":
    run_zone2_interventions()
