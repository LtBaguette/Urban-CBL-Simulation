"""Generate stakeholder / full chart pack into Graphs/ (see config plots.mode)."""

from sim.config import load_config
from sim.plots_stakeholder import generate_all_graphs, generate_full_pack, generate_stakeholder_pack


def main() -> None:
    cfg = load_config()
    if cfg.is_stakeholder_plots:
        paths = generate_stakeholder_pack(cfg)
    else:
        paths = generate_full_pack(cfg)
    for p in paths:
        print("Saved:", p)


if __name__ == "__main__":
    main()
