"""Generate all chart PNGs into Graphs/ from existing sim_outputs CSVs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.plots_stakeholder import generate_all_graphs  # noqa: E402


def main() -> None:
    paths = generate_all_graphs()
    print(f"Wrote {len(paths)} chart(s) to {paths[0].parent if paths else 'Graphs/'}:")
    for p in paths:
        print(" ", p.name)


if __name__ == "__main__":
    main()
