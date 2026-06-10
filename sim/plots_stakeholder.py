"""Chart generation; all PNG outputs go to config graphs_dir (default: Graphs/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from sim.config import SimConfig, load_config
from sim.customer_savings import SMART_SCENARIOS, load_monthly_customer_savings
from sim.metrics import days_per_month

ORDER_STORY = [
    "immediate_plug_in",
    "unmanaged_evening",
    "smart_flat_spread",
    "smart_price_aware",
    "smart_grid_aware",
]

LABELS = {
    "immediate_plug_in": "Immediate\n(no app)",
    "unmanaged_evening": "Evening peak\n(no app)",
    "smart_flat_spread": "Smart app:\nflat spread",
    "smart_price_aware": "Smart app:\nprice-optimized",
    "smart_grid_aware": "Smart app:\ngrid + price",
}

PROFILE_SUBSET_STAKEHOLDER = [
    "immediate_plug_in",
    "unmanaged_evening",
    "smart_price_aware",
]


def _graphs(cfg: SimConfig) -> Path:
    return cfg.ensure_graphs_dir()


def load_app_kpi(cfg: SimConfig | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = cfg.app_scenarios_dir / "app_scenarios_kpi.csv"
    df = pd.read_csv(path)
    if "annual_customer_savings_eur" not in df.columns:
        df["annual_customer_savings_eur"] = df["annual_savings_eur"]
        df["annual_dso_savings_eur"] = 0.0
    return df.set_index("scenario").loc[ORDER_STORY].reset_index()


def load_timeseries(scenario: str, cfg: SimConfig | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = cfg.app_scenarios_dir / f"{scenario}_timeseries.csv"
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")


COLOR_CUSTOMER = "#22c55e"
COLOR_DSO_GAIN = "#6366f1"
COLOR_DSO_COST = "#f87171"


def plot_all_savings(kpi_df: pd.DataFrame, out_path: Path, cfg: SimConfig) -> None:
    """Stacked annual savings: customers (green) and DSOs (indigo / red if negative)."""
    x = np.arange(len(kpi_df))
    x_labels = [LABELS[s] for s in kpi_df["scenario"]]
    dso_k = kpi_df["annual_dso_savings_eur"] / 1000.0
    customer_k = kpi_df["annual_customer_savings_eur"] / 1000.0

    fig, ax = plt.subplots(figsize=(11, 6.5))
    width = 0.65

    for i in range(len(x)):
        dso = float(dso_k.iloc[i])
        customer = float(customer_k.iloc[i])
        if dso < 0:
            ax.bar(
                x[i],
                dso,
                width=width,
                color=COLOR_DSO_COST,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            ax.bar(
                x[i],
                customer,
                width=width,
                bottom=dso,
                color=COLOR_CUSTOMER,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
            total_top = customer + dso
        else:
            if dso > 0:
                ax.bar(
                    x[i],
                    dso,
                    width=width,
                    color=COLOR_DSO_GAIN,
                    edgecolor="white",
                    linewidth=0.8,
                )
            ax.bar(
                x[i],
                customer,
                width=width,
                bottom=max(dso, 0),
                color=COLOR_CUSTOMER,
                edgecolor="white",
                linewidth=0.8,
            )
            total_top = dso + customer

        cust_eur = float(kpi_df["annual_customer_savings_eur"].iloc[i])
        dso_eur = float(kpi_df["annual_dso_savings_eur"].iloc[i])
        if cust_eur == 0 and dso_eur == 0:
            ax.text(x[i], 12, "Reference", ha="center", va="bottom", fontsize=9)
        elif dso < 0:
            mid_dso = dso / 2
            mid_cust = dso + customer / 2
            if abs(dso) >= 1:
                ax.text(
                    x[i],
                    mid_dso,
                    f"DSO\nEUR {dso_eur / 1000:.0f}k",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            if customer >= 1:
                ax.text(
                    x[i],
                    mid_cust,
                    f"Customer\nEUR {cust_eur / 1000:.0f}k",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            ax.text(
                x[i],
                total_top + 20,
                f"Net EUR {(cust_eur + dso_eur) / 1000:.0f}k",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        elif dso > 0 or customer > 0:
            mid_dso = dso / 2 if dso > 0 else 0
            mid_cust = max(dso, 0) + customer / 2
            if dso > 0:
                ax.text(
                    x[i],
                    mid_dso,
                    f"DSO\nEUR {dso_eur / 1000:.0f}k",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            if customer > 0:
                label_y = mid_cust if dso > 0 else total_top / 2
                ax.text(
                    x[i],
                    label_y,
                    f"EUR {cust_eur / 1000:.0f}k" if dso <= 0 else f"Customer\nEUR {cust_eur / 1000:.0f}k",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

    ax.axhline(0, color="#374151", linewidth=0.9)
    ax.set_ylabel("Annual savings (thousand EUR)")
    ax.set_xlabel("Charging behaviour")
    ax.set_title(
        "Annual cost savings by party (vs immediate plug-in)\nZone Z2, 5,000 EVs",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    y_bottom = min(dso_k.min() * 1.08, -50) if dso_k.min() < 0 else 0
    y_top = max((dso_k.clip(lower=0) + customer_k).max() * 1.15, 50)
    if dso_k.min() < 0:
        y_top = max(y_top, customer_k.max() * 1.15)
    ax.set_ylim(y_bottom, y_top)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_CUSTOMER, edgecolor="white", label="Customers (EV bills)"),
        mpatches.Patch(facecolor=COLOR_DSO_GAIN, edgecolor="white", label="DSOs (grid value)"),
        mpatches.Patch(
            facecolor=COLOR_DSO_COST, edgecolor="white", label="DSOs (extra grid cost)"
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.text(
        0.5,
        -0.12,
        "Stacked bars show who benefits; negative DSO = higher grid stress cost vs reference",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_customer_monthly_savings(
    monthly_df: pd.DataFrame, out_path: Path, cfg: SimConfig
) -> None:
    """Per-EV monthly bill savings for smart app scenarios vs immediate plug-in."""
    smart = monthly_df[monthly_df["scenario"].isin(SMART_SCENARIOS)].copy()
    if smart.empty:
        smart = monthly_df[monthly_df["monthly_savings_per_ev_eur"] > 0].copy()
    x = np.arange(len(smart))
    x_labels = [LABELS[s] for s in smart["scenario"]]
    per_ev = smart["monthly_savings_per_ev_eur"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(x, per_ev, width=0.55, color="#22c55e", edgecolor="white")
    ax.set_ylabel("Average savings per EV (EUR / month)")
    ax.set_xlabel("Smart charging app")
    ax.set_title(
        "What drivers save each month with the smart charging app\n"
        f"(vs charging at full power when plugged in, {cfg.fleet.n_evs:,} EV fleet)",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_ylim(0, max(per_ev.max() * 1.18, 5))

    for bar, eur in zip(bars, per_ev):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"EUR {eur:.0f}/mo",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0.5,
        -0.11,
        f"Based on simulated day-ahead prices × {cfg.fleet.kwh_per_day:.0f} kWh/day per vehicle "
        f"({days_per_month(cfg):.1f} days/month)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_stakeholder_pack(cfg: SimConfig | None = None) -> list[Path]:
    cfg = cfg or load_config()
    out = _graphs(cfg)
    kpi_df = load_app_kpi(cfg)

    monthly_df = load_monthly_customer_savings(cfg)
    paths = [
        out / "graph_customer_monthly_savings.png",
        out / "graph_all_savings.png",
    ]
    plot_customer_monthly_savings(monthly_df, paths[0], cfg)
    plot_all_savings(kpi_df, paths[1], cfg)
    return paths


def generate_full_pack(cfg: SimConfig | None = None) -> list[Path]:
    return generate_stakeholder_pack(cfg)


def generate_all_graphs(cfg: SimConfig | None = None) -> list[Path]:
    """Generate every chart PNG into graphs_dir (reads CSVs from sim_outputs)."""
    from sim.deterministic import _plot_charging_profiles

    cfg = cfg or load_config()
    graphs = cfg.ensure_graphs_dir()
    paths: list[Path] = []

    if cfg.is_stakeholder_plots:
        paths.extend(generate_stakeholder_pack(cfg))
    else:
        paths.extend(generate_full_pack(cfg))

    frames = {name: load_timeseries(name, cfg) for name in ORDER_STORY}
    app_profiles = graphs / "app_charging_profiles.png"
    _plot_charging_profiles(frames, app_profiles, cfg)
    paths.append(app_profiles)

    try:
        from sim.method_comparison import (
            generate_all_methods_profiles,
            generate_method_comparison,
        )

        _, method_chart = generate_method_comparison(cfg)
        paths.append(method_chart)
        paths.append(generate_all_methods_profiles(cfg))
    except FileNotFoundError:
        pass

    return paths
