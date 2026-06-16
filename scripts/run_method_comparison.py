#!/usr/bin/env python3
"""Compare all EV charging methods on identical KPI fields."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.method_comparison import generate_method_comparison  # noqa: E402


def main() -> None:
    df, chart = generate_method_comparison()
    print("=== Method comparison (reference: immediate plug-in) ===\n")
    show = [
        "display_name",
        "rank_by_total_savings",
        "annual_customer_savings_eur",
        "annual_dso_savings_eur",
        "annual_total_savings_eur",
        "monthly_savings_per_ev_eur",
        "peak_total_load_mw",
        "peak_reduction_mw_vs_reference",
    ]
    print(df[show].to_string(index=False))
    best = df.loc[df["rank_by_total_savings"] == 1, "display_name"].iloc[0]
    print(f"\nBest overall (customer + DSO): {best}")
    print(f"\nCSV: {chart.parent.parent / 'sim_outputs' / 'method_comparison.csv'}")
    print(f"Chart: {chart}")


if __name__ == "__main__":
    main()
