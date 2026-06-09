#!/usr/bin/env python3
"""Run 15-min AR(1) stochastic simulation (Simulation V5)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SImulation_V5 import main

if __name__ == "__main__":
    main()
