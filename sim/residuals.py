"""
Hourly AR(1) demand + behaviour-based EV fleet + partial smart-charging optimizer.

Aligned with the main Zone Z2 pipeline (fleet size, capacity, data paths from config).
Outputs: sim_outputs/residuals/*.csv and Graphs/residuals_ar1_managed.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

from sim.capacity import resolve_zone_capacity_mw
from sim.config import DEFAULT_CONFIG_PATH, SimConfig, load_config
from sim.data_loaders import load_zone2_15min_demand
from sim.paths import ZONAL_LOAD_FILE

CHARGER_KEYS = ("Home", "Fast", "Super")


@dataclass(frozen=True)
class ResidualsConfig:
    app_adoption_rate: float
    simulation_repeats: int
    random_seed: int
    confidence_interval: float
    morning_max_window_h: int
    evening_deadline_h: int
    show_plot: bool
    save_outputs: bool
    charger_kw: dict[str, float]
    morning_charger_mix: dict[str, float]
    evening_charger_mix: dict[str, float]
    v5_price_weight: float
    v5_grid_weight: float
    v5_grid_load_penalty_eur_per_mw: float


def load_residuals_config(
    sim_cfg: SimConfig | None = None,
    path: Path | None = None,
) -> tuple[SimConfig, ResidualsConfig]:
    sim_cfg = sim_cfg or load_config()
    cfg_path = path or DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    res = raw.get("residuals", {})
    timing = raw.get("timing", {})

    def _mix(section: str) -> dict[str, float]:
        m = res.get(section, {})
        return {k: float(m[k]) for k in CHARGER_KEYS}

    evening_h = int(res.get("evening_deadline_h", timing.get("ready_by_hour", 7)))
    v5_obj = res.get("v5_objective", {})
    default_v5_penalty = float(
        v5_obj.get("grid_load_penalty_eur_per_mw", sim_cfg.grid_load_penalty_eur_per_mw)
    )

    return sim_cfg, ResidualsConfig(
        app_adoption_rate=float(res.get("app_adoption_rate", 0.60)),
        simulation_repeats=int(res.get("simulation_repeats", 5)),
        random_seed=int(res.get("random_seed", 67)),
        confidence_interval=float(res.get("confidence_interval", 0.95)),
        morning_max_window_h=int(res.get("morning_max_window_h", 4)),
        evening_deadline_h=evening_h,
        show_plot=bool(res.get("show_plot", False)),
        save_outputs=bool(res.get("save_outputs", True)),
        charger_kw={
            "Home": float(res.get("charger_kw", {}).get("home", sim_cfg.fleet.charger_kw)),
            "Fast": float(res.get("charger_kw", {}).get("fast", 50)),
            "Super": float(res.get("charger_kw", {}).get("super", 150)),
        },
        morning_charger_mix=_mix("morning_charger_mix")
        if "morning_charger_mix" in res
        else {"Home": 0.40, "Fast": 0.50, "Super": 0.10},
        evening_charger_mix=_mix("evening_charger_mix")
        if "evening_charger_mix" in res
        else {"Home": 0.75, "Fast": 0.20, "Super": 0.05},
        v5_price_weight=float(v5_obj.get("price_weight", 1.0)),
        v5_grid_weight=float(v5_obj.get("grid_weight", 1.0)),
        v5_grid_load_penalty_eur_per_mw=default_v5_penalty,
    )


def _validate_mix(mix: dict[str, float], name: str) -> None:
    if set(mix.keys()) != set(CHARGER_KEYS):
        raise ValueError(f"{name} must have keys {CHARGER_KEYS}")
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1.0")


def build_arrival_weights() -> pd.Series:
    hours = np.arange(24)
    components = [(9.0, 1.5, 0.30), (18.5, 1.8, 0.55), (2.0, 2.0, 0.15)]
    profile = np.zeros(24)
    for mu, sigma, weight in components:
        profile += weight * np.exp(-0.5 * ((hours - mu) / sigma) ** 2)
    profile /= profile.sum()
    return pd.Series(profile, index=range(24), name="arrival_weight")


def _session_hours(charger_key: str, kwh_per_day: float, charger_kw: dict[str, float]) -> int:
    return max(1, math.ceil(kwh_per_day / charger_kw[charger_key]))


def _peak_label(hour: int) -> str:
    return "morning" if 6 <= hour < 14 else "evening"


def _charger_mix_for_hour(
    hour: int, morning_mix: dict[str, float], evening_mix: dict[str, float]
) -> dict[str, float]:
    return morning_mix if _peak_label(hour) == "morning" else evening_mix


def _allowed_hours(
    arrival_hour: int,
    charger_key: str,
    res_cfg: ResidualsConfig,
    kwh_per_day: float,
) -> list[int]:
    session_h = _session_hours(charger_key, kwh_per_day, res_cfg.charger_kw)
    if _peak_label(arrival_hour) == "morning":
        window_h = max(session_h, res_cfg.morning_max_window_h)
        return [(arrival_hour + i) % 24 for i in range(window_h)]
    hours_until_deadline = (res_cfg.evening_deadline_h - arrival_hour) % 24
    window_h = max(session_h, hours_until_deadline)
    return [(arrival_hour + i) % 24 for i in range(window_h)]


def build_ev_load_profile(
    sim_cfg: SimConfig,
    res_cfg: ResidualsConfig,
    arrival_weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    n_evs = sim_cfg.fleet.n_evs
    daily_mwh = sim_cfg.fleet.daily_mwh
    kwh = sim_cfg.fleet.kwh_per_day
    ev_load_mw = np.zeros(24)
    ev_active = np.zeros(24, dtype=float)

    for arrival_hour, arr_weight in arrival_weights.items():
        n_arriving = arr_weight * n_evs
        mix = _charger_mix_for_hour(
            int(arrival_hour), res_cfg.morning_charger_mix, res_cfg.evening_charger_mix
        )
        for charger_key, mix_fraction in mix.items():
            n_evs_type = n_arriving * mix_fraction
            kw = res_cfg.charger_kw[charger_key]
            session_h = _session_hours(charger_key, kwh, res_cfg.charger_kw)
            mw_per_h = kw / 1_000
            for offset in range(session_h):
                h = (int(arrival_hour) + offset) % 24
                ev_load_mw[h] += n_evs_type * mw_per_h
                ev_active[h] += n_evs_type

    raw = ev_load_mw.sum()
    if raw > 0:
        ev_load_mw *= daily_mwh / raw

    return (
        pd.Series(ev_load_mw, index=range(24), name="EV_Load_MW"),
        pd.Series(ev_active.round().astype(int), index=range(24), name="EVs_Active"),
    )


def load_zone2_hourly_statistics(sim_cfg: SimConfig) -> tuple[pd.Series, pd.Series, float, int]:
    load_df = pd.read_csv(ZONAL_LOAD_FILE)
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"], errors="coerce")
    load_df = load_df.dropna(subset=["timestamp"]).copy()
    zone2 = load_df.loc[load_df["zone_id"] == sim_cfg.focus_zone].copy()
    zone2["hour"] = zone2["timestamp"].dt.hour
    grouped = zone2.groupby("hour")["demand_MW"]
    hourly_mean = grouped.mean().reindex(range(24)).interpolate()
    hourly_std = grouped.std().reindex(range(24)).interpolate()
    zone2["date"] = zone2["timestamp"].dt.date
    pivot = zone2.pivot(index="date", columns="hour", values="demand_MW")
    residuals = pivot - hourly_mean.values
    flat = residuals.values.flatten()
    phi = float(np.corrcoef(flat[:-1], flat[1:])[0, 1])
    dof = len(pivot) - 1
    return hourly_mean, hourly_std, phi, dof


def simulate_day_ar1(
    hourly_mean: pd.Series,
    hourly_std: pd.Series,
    phi: float,
    dof: int,
    res_cfg: ResidualsConfig,
    seed: int | None = None,
) -> pd.Series:
    alpha = 1 - res_cfg.confidence_interval
    t_crit = t.ppf(1 - alpha / 2, df=dof)
    rng = np.random.RandomState(seed)
    simulated: list[float] = []
    prev_eps = 0.0
    for hour in range(24):
        mu = hourly_mean[hour]
        sigma = hourly_std[hour]
        while True:
            innovation = t.rvs(df=dof, random_state=rng) * sigma * np.sqrt(1 - phi**2)
            eps = phi * prev_eps + innovation
            if abs(eps) <= t_crit * sigma:
                break
        value = max(0.0, mu + eps)
        simulated.append(value)
        prev_eps = eps
    return pd.Series(simulated, index=range(24), name="demand_MW")


def decompose_ev_from_demand(
    total_demand: pd.Series,
    ev_load: pd.Series,
    ev_active: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    base_load = (total_demand - ev_load).clip(lower=0).rename("Base_Load_MW")
    return ev_load, base_load, ev_active


def optimize_ev_load_constrained(
    base_load: pd.Series,
    sim_cfg: SimConfig,
    res_cfg: ResidualsConfig,
    arrival_weights: pd.Series,
    app_seed: int,
) -> tuple[pd.Series, pd.Series, float]:
    rng = np.random.default_rng(app_seed)
    n_evs = sim_cfg.fleet.n_evs
    total_ev_mwh = sim_cfg.fleet.daily_mwh
    kwh = sim_cfg.fleet.kwh_per_day

    non_app_load = np.zeros(24)
    non_app_active = np.zeros(24, dtype=float)
    app_groups: list[dict[str, Any]] = []
    total_app_evs = 0.0
    total_non_app_evs = 0.0

    for arrival_hour, arr_weight in arrival_weights.items():
        n_arriving = arr_weight * n_evs
        mix = _charger_mix_for_hour(
            int(arrival_hour), res_cfg.morning_charger_mix, res_cfg.evening_charger_mix
        )
        for charger_key, mix_fraction in mix.items():
            n_total = n_arriving * mix_fraction
            n_app = float(rng.binomial(int(round(n_total)), res_cfg.app_adoption_rate))
            n_no_app = n_total - n_app
            session_h = _session_hours(charger_key, kwh, res_cfg.charger_kw)
            mw_per_h = res_cfg.charger_kw[charger_key] / 1_000
            allowed = _allowed_hours(int(arrival_hour), charger_key, res_cfg, kwh)

            for offset in range(session_h):
                h = (int(arrival_hour) + offset) % 24
                non_app_load[h] += n_no_app * mw_per_h
                non_app_active[h] += n_no_app

            if n_app > 0:
                app_groups.append(
                    {
                        "allowed": allowed,
                        "session_h": session_h,
                        "mw_per_h": mw_per_h * n_app,
                        "n_evs": n_app,
                        "energy_mwh": n_app * kwh / 1_000,
                    }
                )
                total_app_evs += n_app
            total_non_app_evs += n_no_app

    current_total = base_load.values.copy() + non_app_load
    app_schedule = np.zeros(24)
    app_active = np.zeros(24, dtype=float)

    for group in app_groups:
        allowed = group["allowed"]
        session_h = group["session_h"]
        mw_per_h = group["mw_per_h"]
        if len(allowed) < session_h:
            continue
        best_start = 0
        best_peak = float("inf")
        for start_idx in range(len(allowed) - session_h + 1):
            block = allowed[start_idx : start_idx + session_h]
            peak = max(current_total[h] + mw_per_h for h in block)
            if peak < best_peak:
                best_peak = peak
                best_start = start_idx
        for h in allowed[best_start : best_start + session_h]:
            current_total[h] += mw_per_h
            app_schedule[h] += mw_per_h
            app_active[h] += group["n_evs"]

    opt_ev = non_app_load + app_schedule
    raw = opt_ev.sum()
    if raw > 0:
        opt_ev *= total_ev_mwh / raw

    opt_active = (non_app_active + app_active).round().astype(int)
    actual_rate = total_app_evs / n_evs if n_evs > 0 else 0.0
    return (
        pd.Series(opt_ev, index=range(24), name="Opt_EV_Load_MW"),
        pd.Series(opt_active, index=range(24), name="Opt_EVs_Active"),
        float(actual_rate),
    )


def run_single(
    run_id: int,
    sim_cfg: SimConfig,
    res_cfg: ResidualsConfig,
    hourly_mean: pd.Series,
    hourly_std: pd.Series,
    phi: float,
    dof: int,
    zone_capacity_mw: float,
    ev_load: pd.Series,
    ev_active: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    seed = res_cfg.random_seed + run_id
    app_seed = res_cfg.random_seed * 100 + run_id

    stochastic_total = simulate_day_ar1(
        hourly_mean, hourly_std, phi, dof, res_cfg, seed=seed
    )
    _, base_load, _ = decompose_ev_from_demand(stochastic_total, ev_load, ev_active)
    unmanaged_total = base_load + ev_load

    opt_ev, opt_active, actual_rate = optimize_ev_load_constrained(
        base_load, sim_cfg, res_cfg, build_arrival_weights(), app_seed
    )
    managed_total = base_load + opt_ev

    rows = []
    for h in range(24):
        rows.append(
            {
                "hour": h,
                "stochastic_grid_mw": float(stochastic_total[h]),
                "base_load_mw": float(base_load[h]),
                "unmanaged_ev_mw": float(ev_load[h]),
                "unmanaged_total_mw": float(unmanaged_total[h]),
                "managed_ev_mw": float(opt_ev[h]),
                "managed_total_mw": float(managed_total[h]),
                "unmanaged_stress": float(unmanaged_total[h]) / zone_capacity_mw,
                "managed_stress": float(managed_total[h]) / zone_capacity_mw,
                "unmanaged_evs_active": int(ev_active[h]),
                "managed_evs_active": int(opt_active[h]),
            }
        )
    frame = pd.DataFrame(rows)
    kpi = {
        "run_id": run_id,
        "app_adoption_target": res_cfg.app_adoption_rate,
        "app_adoption_actual": actual_rate,
        "zone_capacity_mw": zone_capacity_mw,
        "peak_unmanaged_mw": float(unmanaged_total.max()),
        "peak_managed_mw": float(managed_total.max()),
        "peak_reduction_mw": float(unmanaged_total.max() - managed_total.max()),
        "peak_stress_unmanaged": float(frame["unmanaged_stress"].max()),
        "peak_stress_managed": float(frame["managed_stress"].max()),
        "ev_energy_managed_mwh": float(opt_ev.sum()),
        "ev_energy_target_mwh": sim_cfg.fleet.daily_mwh,
    }
    return frame, kpi


def run_residuals(
    sim_cfg: SimConfig | None = None,
    res_cfg: ResidualsConfig | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    if sim_cfg is None:
        sim_cfg, res_cfg = load_residuals_config()
    elif res_cfg is None:
        _, res_cfg = load_residuals_config(sim_cfg)

    _validate_mix(res_cfg.morning_charger_mix, "morning_charger_mix")
    _validate_mix(res_cfg.evening_charger_mix, "evening_charger_mix")

    arrival_weights = build_arrival_weights()
    ev_load, ev_active = build_ev_load_profile(sim_cfg, res_cfg, arrival_weights)
    hourly_mean, hourly_std, phi, dof = load_zone2_hourly_statistics(sim_cfg)
    grid_15 = load_zone2_15min_demand(sim_cfg)
    zone_capacity_mw, cap_meta = resolve_zone_capacity_mw(grid_15, sim_cfg)

    out_dir = sim_cfg.output_dir / "residuals"
    if res_cfg.save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Zone {sim_cfg.focus_zone} residuals (hourly AR1 + smart charging) ===")
    print(
        f"Fleet: {sim_cfg.fleet.n_evs:,} EVs × {sim_cfg.fleet.kwh_per_day} kWh/day "
        f"= {sim_cfg.fleet.daily_mwh:.1f} MWh/day (matches config/default.yaml)"
    )
    print(f"Fixed zone capacity: {zone_capacity_mw:.2f} MW ({cap_meta['capacity_source']})")
    print(f"App adoption target: {res_cfg.app_adoption_rate * 100:.0f}%")
    print(f"Runs: {res_cfg.simulation_repeats}  |  AR(1) phi = {phi:.4f}\n")

    kpi_rows: list[dict[str, float]] = []
    last_frame: pd.DataFrame | None = None

    for run_id in range(1, res_cfg.simulation_repeats + 1):
        frame, kpi = run_single(
            run_id,
            sim_cfg,
            res_cfg,
            hourly_mean,
            hourly_std,
            phi,
            dof,
            zone_capacity_mw,
            ev_load,
            ev_active,
        )
        kpi_rows.append(kpi)
        last_frame = frame
        if res_cfg.save_outputs:
            frame.to_csv(out_dir / f"run_{run_id:03d}_hourly.csv", index=False)
        print(
            f"Run {run_id}: adoption {kpi['app_adoption_actual'] * 100:.1f}% | "
            f"peak {kpi['peak_unmanaged_mw']:.1f} -> {kpi['peak_managed_mw']:.1f} MW "
            f"(delta {kpi['peak_reduction_mw']:.1f} MW)"
        )

    kpi_df = pd.DataFrame(kpi_rows)
    if res_cfg.save_outputs:
        kpi_df.to_csv(out_dir / "residuals_kpi.csv", index=False)

    fig_path = _save_figure(
        sim_cfg,
        res_cfg,
        hourly_mean,
        zone_capacity_mw,
        ev_load,
        ev_active,
        hourly_std,
        phi,
        dof,
        kpi_df,
    )

    if res_cfg.show_plot:
        plt.show()
    else:
        plt.close("all")

    print(f"\nMean peak reduction: {kpi_df['peak_reduction_mw'].mean():.2f} MW")
    if res_cfg.save_outputs:
        print(f"CSVs: {out_dir}")
    if fig_path:
        print(f"Chart: {fig_path}")

    return kpi_df, fig_path


def _save_figure(
    sim_cfg: SimConfig,
    res_cfg: ResidualsConfig,
    hourly_mean: pd.Series,
    zone_capacity_mw: float,
    ev_load: pd.Series,
    ev_active: pd.Series,
    hourly_std: pd.Series,
    phi: float,
    dof: int,
    kpi_df: pd.DataFrame,
) -> Path | None:
    hours = range(24)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    colours = ["#378ADD", "#1D9E75", "#D85A30", "#9B59B6", "#E67E22"]
    n_plot = min(len(kpi_df), len(colours))

    for idx in range(n_plot):
        run_id = int(kpi_df.iloc[idx]["run_id"])
        colour = colours[idx]
        seed = res_cfg.random_seed + run_id
        app_seed = res_cfg.random_seed * 100 + run_id
        stochastic = simulate_day_ar1(
            hourly_mean, hourly_std, phi, dof, res_cfg, seed=seed
        )
        _, base_load, _ = decompose_ev_from_demand(stochastic, ev_load, ev_active)
        opt_ev, opt_active, rate = optimize_ev_load_constrained(
            base_load, sim_cfg, res_cfg, build_arrival_weights(), app_seed
        )
        unmanaged = base_load + ev_load
        managed = base_load + opt_ev

        axes[0].plot(
            hours, unmanaged, color=colour, lw=2, ls="-", label=f"Run {run_id} unmanaged"
        )
        axes[0].plot(
            hours,
            managed,
            color=colour,
            lw=2,
            ls="--",
            label=f"Run {run_id} managed ({rate * 100:.0f}% app)",
        )
        axes[1].plot(hours, ev_active, color=colour, lw=1.8, ls="-", alpha=0.85)
        axes[1].plot(hours, opt_active, color=colour, lw=1.8, ls="--", alpha=0.85)

    axes[0].axhline(
        zone_capacity_mw,
        color="#dc2626",
        ls=":",
        lw=1.5,
        label=f"Zone capacity ({zone_capacity_mw:.0f} MW)",
    )
    axes[0].plot(
        hours, hourly_mean, color="black", lw=1.5, ls=":", label="Historical mean (Z2)"
    )
    axes[1].axhline(
        sim_cfg.fleet.n_evs,
        color="#6b7280",
        ls=":",
        lw=1.2,
        label="Fleet size",
    )
    axes[0].set_ylabel("Demand (MW)")
    axes[0].set_title(
        f"Zone {sim_cfg.focus_zone}: stochastic demand vs partial smart charging "
        f"({res_cfg.app_adoption_rate * 100:.0f}% adoption target)",
        fontweight="bold",
    )
    axes[1].set_ylabel("EVs charging")
    axes[1].set_xlabel("Hour of day")
    axes[0].legend(loc="lower left", fontsize=7, ncol=2)
    axes[1].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[1].grid(True, ls=":", alpha=0.5)
    plt.xticks(range(24))
    fig.tight_layout()

    sim_cfg.ensure_graphs_dir()
    path = sim_cfg.graphs_dir / "residuals_ar1_managed.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def main() -> None:
    run_residuals()


if __name__ == "__main__":
    main()
