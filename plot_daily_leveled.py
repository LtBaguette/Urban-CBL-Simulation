"""Daily load leveling chart (writes to Graphs/)."""

from sim.plots_stakeholder import plot_daily_load_leveled


def main() -> None:
    out = plot_daily_load_leveled()
    print("Saved:", out)


if __name__ == "__main__":
    main()
