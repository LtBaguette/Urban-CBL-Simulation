"""Chart generation; all PNG outputs go to config graphs_dir (default: Graphs/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from sim.config import SimConfig, load_config

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

PEAK_COMPARE_SCENARIOS = [
    "immediate_plug_in",
    "unmanaged_evening",
    "smart_price_aware",
]

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


def plot_customer_savings(kpi_df: pd.DataFrame, out_path: Path, cfg: SimConfig) -> None:
    x = np.arange(len(kpi_df))
    x_labels = [LABELS[s] for s in kpi_df["scenario"]]
    savings_k = kpi_df["annual_customer_savings_eur"] / 1000.0
    colors = ["#9ca3af" if s == "immediate_plug_in" else "#22c55e" for s in kpi_df["scenario"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, savings_k, color=colors, width=0.65, edgecolor="white")
    ax.set_ylabel("Annual savings (thousand EUR)")
    ax.set_xlabel("Charging behaviour")
    ax.set_title(
        "Annual EV charging cost savings (Zone Z2, 5,000 EVs)",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_ylim(0, max(savings_k.max() * 1.15, 50))

    for bar, eur in zip(bars, kpi_df["annual_customer_savings_eur"]):
        label = "Reference" if eur <= 0 else f"EUR {eur / 1000:.0f}k\nper year"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if eur > 0 else "normal",
        )

    ax.text(
        0.5,
        -0.14,
        "Compared to charging at full power when plugged in",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        style="italic",
    )
    ax.text(
        0.98,
        0.97,
        f"Fleet: {cfg.fleet.n_evs:,} EVs | {cfg.fleet.kwh_per_day:.0f} kWh/day per vehicle\n"
        f"Daily fleet energy: {cfg.fleet.daily_mwh:.0f} MWh",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#ecfdf5", edgecolor="#a7f3d0"),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_daily_comparison(
    reference_df: pd.DataFrame,
    smart_df: pd.DataFrame,
    out_path: Path,
    cfg: SimConfig,
) -> None:
    cap = float(reference_df["Zone_Capacity_MW"].iloc[0])
    peak_ref = float(reference_df["Total_Load_MW"].max())
    peak_smart = float(smart_df["Total_Load_MW"].max())

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        reference_df.index,
        reference_df["Total_Load_MW"],
        color="#6b7280",
        lw=2.2,
        label="Immediate plug-in (reference)",
    )
    ax.plot(
        smart_df.index,
        smart_df["Total_Load_MW"],
        color="#059669",
        lw=2.4,
        label="Smart app (price-optimized)",
    )
    ax.axhline(cap, color="#dc2626", ls="--", lw=1.5, label=f"Zone capacity ({cap:.0f} MW)")
    ax.set_ylabel("Total grid load (MW)")
    ax.set_xlabel("Time of day")
    ax.set_title(
        "Daily total load: smart charging shapes demand through the day",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)

    if abs(peak_ref - peak_smart) < 5:
        note = (
            f"Similar daily peak ({peak_ref:.0f} vs {peak_smart:.0f} MW); "
            "main savings come from cheaper charging hours."
        )
    else:
        note = f"Peak load: {peak_ref:.0f} MW (reference) vs {peak_smart:.0f} MW (smart)."

    ax.text(
        0.99,
        0.03,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#d1d5db"),
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ev_shift(
    reference_df: pd.DataFrame,
    smart_df: pd.DataFrame,
    out_path: Path,
    cfg: SimConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        reference_df.index,
        reference_df["EV_Load_MW"],
        color="#6b7280",
        lw=2.2,
        label="Immediate plug-in (reference)",
    )
    ax.plot(
        smart_df.index,
        smart_df["EV_Load_MW"],
        color="#059669",
        lw=2.4,
        label="Smart app (price-optimized)",
    )
    plug_start = reference_df.index[reference_df.index.hour >= cfg.plug_in_hour][0]
    plug_end = reference_df.index[reference_df.index.hour < cfg.ready_by_hour][-1]
    ax.axvspan(plug_start, reference_df.index[-1], alpha=0.06, color="#3b82f6")
    ax.axvspan(reference_df.index[0], plug_end, alpha=0.06, color="#3b82f6")
    window_patch = mpatches.Patch(
        facecolor="#3b82f6", alpha=0.15, label="Plug-in window (18:00-07:00)"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [window_patch], loc="upper right", fontsize=9)
    ax.set_ylabel("EV charging load (MW)")
    ax.set_xlabel("Time of day")
    ax.set_title("Smart app moves charging to lower-price hours", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    ax.text(
        0.02,
        0.97,
        f"Same daily energy: {cfg.fleet.daily_mwh:.0f} MWh",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="#f0fdf4", edgecolor="#bbf7d0"),
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_peak_compare_subset(kpi_df: pd.DataFrame, out_path: Path, cfg: SimConfig) -> None:
    subset = kpi_df[kpi_df["scenario"].isin(PEAK_COMPARE_SCENARIOS)].copy()
    subset["_order"] = subset["scenario"].map(
        {s: i for i, s in enumerate(PEAK_COMPARE_SCENARIOS)}
    )
    subset = subset.sort_values("_order")
    x = np.arange(len(subset))
    x_labels = [LABELS[s] for s in subset["scenario"]]
    cap = float(subset["zone_capacity_mw"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    bars = ax.bar(x, subset["peak_total_load_mw"], width=0.5, color="#60a5fa", edgecolor="white")
    ax.axhline(cap, color="#6b7280", ls="--", lw=1.2)
    ax.set_ylabel("Peak total load (MW)")
    ax.set_title("Peak load", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    ax2 = axes[1]
    stress_colors = ["#9ca3af", "#ea580c", "#059669"]
    ax2.bar(
        x, subset["peak_stress_ratio"], width=0.5, color=stress_colors, edgecolor="white"
    )
    ax2.axhline(1.0, color="#6b7280", ls="--", lw=1.2)
    ax2.set_ylabel("Peak stress ratio")
    ax2.set_title("Peak stress", fontweight="bold", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)

    fig.suptitle("Grid peak reference (3 behaviours)", fontweight="bold", fontsize=12)
    fig.text(
        0.5,
        0.02,
        "Smart price-optimized matches plug-in peak; main benefit is electricity cost.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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


def generate_stakeholder_pack(cfg: SimConfig | None = None) -> list[Path]:
    cfg = cfg or load_config()
    out = _graphs(cfg)
    kpi_df = load_app_kpi(cfg)
    ref_ts = load_timeseries("immediate_plug_in", cfg)
    smart_ts = load_timeseries("smart_price_aware", cfg)

    paths = [
        out / "graph_customer_savings.png",
        out / "graph_all_savings.png",
        out / "graph_ev_shift.png",
        out / "graph_daily_comparison.png",
        out / "graph_grid_peak_compare.png",
    ]
    plot_customer_savings(kpi_df, paths[0], cfg)
    plot_all_savings(kpi_df, paths[1], cfg)
    plot_ev_shift(ref_ts, smart_ts, paths[2], cfg)
    plot_daily_comparison(ref_ts, smart_ts, paths[3], cfg)
    plot_peak_compare_subset(kpi_df, paths[4], cfg)
    return paths


def generate_full_pack(cfg: SimConfig | None = None) -> list[Path]:
    return generate_stakeholder_pack(cfg)


def plot_daily_load_leveled(cfg: SimConfig | None = None) -> Path:
    """Two-panel daily load + stress (evening peak vs price-optimized)."""
    cfg = cfg or load_config()
    no_app = load_timeseries("unmanaged_evening", cfg)
    smart = load_timeseries("smart_price_aware", cfg)
    cap = float(no_app["Zone_Capacity_MW"].iloc[0])

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax = axes[0]
    ax.fill_between(no_app.index, no_app["Total_Load_MW"], alpha=0.18, color="#dc2626")
    ax.plot(
        no_app.index,
        no_app["Total_Load_MW"],
        color="#dc2626",
        lw=2.4,
        label="Without smart charging (evening peak)",
    )
    ax.fill_between(smart.index, smart["Total_Load_MW"], alpha=0.15, color="#059669")
    ax.plot(
        smart.index,
        smart["Total_Load_MW"],
        color="#059669",
        lw=2.4,
        label="With smart charging app (price-optimized)",
    )
    ax.axhline(cap, color="#6b7280", ls="--", lw=1.5, label=f"Zone capacity ({cap:.0f} MW)")
    peak_no = no_app["Total_Load_MW"].max()
    peak_yes = smart["Total_Load_MW"].max()
    ax.annotate(
        f"Peak {peak_no:.0f} MW",
        xy=(no_app["Total_Load_MW"].idxmax(), peak_no),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=9,
        color="#dc2626",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
    )
    ax.annotate(
        f"Peak {peak_yes:.0f} MW",
        xy=(smart["Total_Load_MW"].idxmax(), peak_yes),
        xytext=(-60, 18),
        textcoords="offset points",
        fontsize=9,
        color="#059669",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
    )
    ax.set_ylabel("Total grid load (MW)")
    ax.set_title(
        "Smart charging levels daily load — sharper peak vs smoother profile",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    std_no = no_app["Total_Load_MW"].std()
    std_yes = smart["Total_Load_MW"].std()
    ax.text(
        0.99,
        0.03,
        f"Load variability (std): {std_no:.1f} MW without app  ->  {std_yes:.1f} MW with app",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#d1d5db"),
    )

    ax2 = axes[1]
    ax2.fill_between(no_app.index, no_app["Stress_Ratio"], alpha=0.18, color="#dc2626")
    ax2.plot(
        no_app.index, no_app["Stress_Ratio"], color="#dc2626", lw=2.2, label="Without smart charging"
    )
    ax2.plot(
        smart.index, smart["Stress_Ratio"], color="#059669", lw=2.2, label="With smart charging"
    )
    ax2.axhline(1.0, color="#6b7280", ls="--", lw=1.5, label="Grid limit (stress = 1.0)")
    overload_no = int(no_app["Bottleneck"].sum())
    overload_yes = int(smart["Bottleneck"].sum())
    ax2.set_ylabel("Grid stress ratio")
    ax2.set_xlabel("Time of day")
    ax2.set_title("Congestion stress is spread out instead of spiking above the limit", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.text(
        0.99,
        0.97,
        f"Overload intervals: {overload_no} without  vs  {overload_yes} with",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            fc="#fef2f2" if overload_no > overload_yes else "#ecfdf5",
            ec="#fecaca",
        ),
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = _graphs(cfg) / "graph_daily_load_leveled.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_graphs(cfg: SimConfig | None = None) -> list[Path]:
    """Generate every chart PNG into graphs_dir (reads CSVs from sim_outputs)."""
    from sim.deterministic import (
        _plot_baseline_vs_intervention,
        _plot_charging_profiles,
    )

    cfg = cfg or load_config()
    graphs = cfg.ensure_graphs_dir()
    paths: list[Path] = []

    if cfg.is_stakeholder_plots:
        paths.extend(generate_stakeholder_pack(cfg))
    else:
        paths.extend(generate_full_pack(cfg))

    paths.append(plot_daily_load_leveled(cfg))

    frames = {name: load_timeseries(name, cfg) for name in ORDER_STORY}
    app_profiles = graphs / "app_charging_profiles.png"
    _plot_charging_profiles(frames, app_profiles, cfg)
    paths.append(app_profiles)

    baseline_csv = cfg.output_dir / "zone2_timeseries_baseline.csv"
    if baseline_csv.exists():
        baseline = pd.read_csv(baseline_csv, parse_dates=["timestamp"]).set_index("timestamp")
        interventions: dict[int, pd.DataFrame] = {}
        for pct in cfg.intervention_pcts:
            p = cfg.output_dir / f"zone2_timeseries_intervention_{pct}pct.csv"
            if p.exists():
                interventions[pct] = pd.read_csv(p, parse_dates=["timestamp"]).set_index(
                    "timestamp"
                )
        if interventions:
            zone2_path = graphs / "zone2_baseline_vs_intervention.png"
            _plot_baseline_vs_intervention(baseline, interventions, zone2_path, cfg)
            paths.append(zone2_path)

    return paths
