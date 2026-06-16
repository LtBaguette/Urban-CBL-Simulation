"""Hourly AR(1) + partial smart-charging prototype (aligned with main fleet config)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.residuals import main  # noqa: E402

if __name__ == "__main__":
    main()
