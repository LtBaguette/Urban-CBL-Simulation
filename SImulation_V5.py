from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.capacity import resolve_zone_capacity_mw
from sim.config import SimConfig, load_config
from sim.data_loaders import load_mean_hourly_price_profile, load_zone2_15min_demand
from sim.dso_value import annual_dso_savings_eur
from sim.metrics import (
    congestion_stress_integral_saved,
    daily_ev_energy_cost_slots,
    grid_stress_minutes,
)
from sim.paths import ZONAL_LOAD_FILE
from sim.residuals import ResidualsConfig, load_residuals_config

CONFIDENCE_INTERVAL = 0.95
SIMULATION_REPEATS = 1
RANDOM_SEED = 60
N_EVS = 5000
EV_KWH_PER_DAY = 25.0
FOCUS_ZONE = "Z2"
SLOTS_PER_HOUR = 4
SLOT_MINUTES = 60 // SLOTS_PER_HOUR
N_SLOTS = 24 * SLOTS_PER_HOUR
APP_ADOPTION_RATE = 0.60
MORNING_MAX_WINDOW_H = 4
EVENING_DEADLINE_H = 7
AR1_MAX_ITER = 200
SAVE_OUTPUTS = True
SHOW_PLOT = False
EV_DAILY_MWH = N_EVS * EV_KWH_PER_DAY / 1_000
CHARGER_KW = {"Home": 11.0, "Fast": 50.0, "Super": 150.0}
MORNING_CHARGER_MIX = {"Home": 0.40, "Fast": 0.50, "Super": 0.10}
EVENING_CHARGER_MIX = {"Home": 0.75, "Fast": 0.20, "Super": 0.05}


def _validate_charger_config() -> None:
    assert abs(sum(MORNING_CHARGER_MIX.values()) - 1.0) < 1e-6
    assert abs(sum(EVENING_CHARGER_MIX.values()) - 1.0) < 1e-6
    assert MORNING_CHARGER_MIX.keys() == CHARGER_KW.keys()
    assert EVENING_CHARGER_MIX.keys() == CHARGER_KW.keys()
    assert 0.0 <= APP_ADOPTION_RATE <= 1.0


def _apply_runtime_config(sim_cfg: SimConfig, res_cfg: ResidualsConfig) -> None:
    global CONFIDENCE_INTERVAL, SIMULATION_REPEATS, RANDOM_SEED
    global N_EVS, EV_KWH_PER_DAY, EV_DAILY_MWH, FOCUS_ZONE
    global APP_ADOPTION_RATE, MORNING_MAX_WINDOW_H, EVENING_DEADLINE_H
    global CHARGER_KW, MORNING_CHARGER_MIX, EVENING_CHARGER_MIX
    global SAVE_OUTPUTS, SHOW_PLOT

    CONFIDENCE_INTERVAL = res_cfg.confidence_interval
    SIMULATION_REPEATS = res_cfg.simulation_repeats
    RANDOM_SEED = res_cfg.random_seed
    N_EVS = sim_cfg.fleet.n_evs
    EV_KWH_PER_DAY = sim_cfg.fleet.kwh_per_day
    EV_DAILY_MWH = sim_cfg.fleet.daily_mwh
    FOCUS_ZONE = sim_cfg.focus_zone
    APP_ADOPTION_RATE = res_cfg.app_adoption_rate
    MORNING_MAX_WINDOW_H = res_cfg.morning_max_window_h
    EVENING_DEADLINE_H = res_cfg.evening_deadline_h
    CHARGER_KW = dict(res_cfg.charger_kw)
    MORNING_CHARGER_MIX = dict(res_cfg.morning_charger_mix)
    EVENING_CHARGER_MIX = dict(res_cfg.evening_charger_mix)
    SAVE_OUTPUTS = res_cfg.save_outputs
    SHOW_PLOT = res_cfg.show_plot
    _validate_charger_config()


def slot_to_hour(slot: int) -> int:
    return (slot % N_SLOTS) // SLOTS_PER_HOUR


def slot_label(slot: int) -> str:
    s = slot % N_SLOTS
    h = s // SLOTS_PER_HOUR
    m = (s % SLOTS_PER_HOUR) * SLOT_MINUTES
    return f"{h:02d}:{m:02d}"


def build_arrival_weights() -> pd.Series:
    slots = np.arange(N_SLOTS)
    components = [
        (9.0 * SLOTS_PER_HOUR, 1.5 * SLOTS_PER_HOUR, 0.30),
        (18.5 * SLOTS_PER_HOUR, 2.3 * SLOTS_PER_HOUR, 0.70),
    ]
    profile = np.zeros(N_SLOTS)
    for mu, sigma, weight in components:
        profile += weight * np.exp(-0.5 * ((slots - mu) / sigma) ** 2)
    profile /= profile.sum()
    return pd.Series(profile, index=range(N_SLOTS), name="arrival_weight")


def _session_slots(charger_key: str) -> int:
    slots = EV_KWH_PER_DAY / CHARGER_KW[charger_key] * SLOTS_PER_HOUR
    return max(1, math.ceil(slots))


def _peak_label(slot: int) -> str:
    hour = slot_to_hour(slot)
    return "morning" if 6 <= hour < 17 else "evening"


def _charger_mix_for_slot(slot: int) -> dict[str, float]:
    return MORNING_CHARGER_MIX if _peak_label(slot) == "morning" else EVENING_CHARGER_MIX


def _allowed_slots(arrival_slot: int, charger_key: str) -> list[int]:
    session_s = _session_slots(charger_key)
    deadline_slot = EVENING_DEADLINE_H * SLOTS_PER_HOUR
    if _peak_label(arrival_slot) == "morning":
        window_s = max(session_s, MORNING_MAX_WINDOW_H * SLOTS_PER_HOUR)
        return [(arrival_slot + i) % N_SLOTS for i in range(window_s)]
    if arrival_slot < deadline_slot:
        slots_until = deadline_slot - arrival_slot
    else:
        slots_until = (N_SLOTS - arrival_slot) + deadline_slot
    window_s = max(session_s, slots_until)
    return [(arrival_slot + i) % N_SLOTS for i in range(window_s)]


def build_ev_load_profile(arrival_weights: pd.Series) -> tuple[pd.Series, pd.Series]:
    ev_load_mw = np.zeros(N_SLOTS)
    ev_active_float = np.zeros(N_SLOTS, dtype=float)
    for arrival_slot, arr_weight in arrival_weights.items():
        n_arriving = arr_weight * N_EVS
        mix = _charger_mix_for_slot(int(arrival_slot))
        for charger_key, mix_fraction in mix.items():
            n_evs_this_type = n_arriving * mix_fraction
            session_s = _session_slots(charger_key)
            mw_per_ev = CHARGER_KW[charger_key] / 1_000
            for offset in range(session_s):
                target_slot = (int(arrival_slot) + offset) % N_SLOTS
                ev_load_mw[target_slot] += n_evs_this_type * mw_per_ev
                ev_active_float[target_slot] += n_evs_this_type
    slot_duration_h = 1.0 / SLOTS_PER_HOUR
    raw_energy_mwh = ev_load_mw.sum() * slot_duration_h
    scale = EV_DAILY_MWH / raw_energy_mwh
    ev_load_mw *= scale
    ev_active_float *= scale
    return (
        pd.Series(ev_load_mw, index=range(N_SLOTS), name="EV_Load_MW"),
        pd.Series(ev_active_float.round().astype(int), index=range(N_SLOTS), name="EVs_Active"),
    )


def load_zone2_statistics() -> tuple[pd.Series, pd.Series, float, int]:
    load_df = pd.read_csv(ZONAL_LOAD_FILE)
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"], errors="coerce")
    load_df = load_df.dropna(subset=["timestamp"]).copy()
    zone2 = load_df.loc[load_df["zone_id"] == FOCUS_ZONE].copy()
    zone2["hour"] = zone2["timestamp"].dt.hour
    grouped = zone2.groupby("hour")["demand_MW"]
    hourly_mean = grouped.mean().reindex(range(24)).interpolate()
    hourly_std = grouped.std().reindex(range(24)).interpolate()
    zone2 = zone2.sort_values("timestamp")
    zone2["date"] = zone2["timestamp"].dt.date
    zone2 = zone2.drop_duplicates(subset=["date", "hour"])
    pivot = zone2.pivot(index="date", columns="hour", values="demand_MW")
    residuals = (pivot - hourly_mean.values).values
    r0 = residuals[:, :-1].flatten()
    r1 = residuals[:, 1:].flatten()
    valid = np.isfinite(r0) & np.isfinite(r1)
    phi_hourly = float(np.corrcoef(r0[valid], r1[valid])[0, 1])
    dof = int(valid.sum()) - 1
    phi_15min = phi_hourly ** (1.0 / SLOTS_PER_HOUR)
    print(
        f"Estimated AR(1) phi (hourly) = {phi_hourly:.4f} "
        f"-> phi (15-min) = {phi_15min:.4f} | DoF = {dof}"
    )
    hour_centres = np.arange(24)
    slot_centres = np.arange(N_SLOTS) / SLOTS_PER_HOUR
    ext_hours = np.concatenate([hour_centres - 24, hour_centres, hour_centres + 24])
    ext_mean = np.tile(hourly_mean.values, 3)
    ext_std = np.tile(hourly_std.values, 3)
    mean_15min = CubicSpline(ext_hours, ext_mean)(slot_centres)
    std_15min = CubicSpline(ext_hours, ext_std)(slot_centres)
    std_15min = np.clip(std_15min, 1e-3, None)
    return (
        pd.Series(mean_15min, index=range(N_SLOTS), name="mean_MW"),
        pd.Series(std_15min, index=range(N_SLOTS), name="std_MW"),
        phi_15min,
        dof,
    )


def simulate_day_ar1(
    slot_mean: pd.Series,
    slot_std: pd.Series,
    phi: float,
    dof: int,
    confidence: float = CONFIDENCE_INTERVAL,
    seed: int | None = None,
) -> pd.Series:
    alpha = 1 - confidence
    t_crit = t.ppf(1 - alpha / 2, df=dof)
    rng = np.random.RandomState(seed)
    simulated: list[float] = []
    prev_eps = 0.0
    for s in range(N_SLOTS):
        mu = slot_mean[s]
        sigma = slot_std[s]
        band = t_crit * sigma
        eps = None
        for _ in range(AR1_MAX_ITER):
            innovation = t.rvs(df=dof, random_state=rng) * sigma * np.sqrt(max(0.0, 1 - phi**2))
            candidate = phi * prev_eps + innovation
            if abs(candidate) <= band:
                eps = candidate
                break
        if eps is None:
            eps = float(np.clip(phi * prev_eps, -band, band))
            warnings.warn(
                f"AR(1) sampler clamped at slot {s} (phi={phi:.3f}).",
                RuntimeWarning,
                stacklevel=2,
            )
        value = max(0.0, mu + eps)
        simulated.append(value)
        prev_eps = eps
    return pd.Series(simulated, index=range(N_SLOTS), name="demand_MW")


def decompose_ev_from_demand(
    total_demand: pd.Series,
    ev_load_profile: pd.Series,
    ev_active_profile: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    base_load = (total_demand - ev_load_profile).clip(lower=0).rename("Base_Load_MW")
    return ev_load_profile, base_load, ev_active_profile


def load_slot_price_profile(sim_cfg: SimConfig) -> pd.Series:
    hourly_price = load_mean_hourly_price_profile(sim_cfg)
    return pd.Series(hourly_price.values, index=range(N_SLOTS), name="price_eur_mwh")


def _block_effective_cost(
    block: list[int],
    current_total: np.ndarray,
    slot_prices: np.ndarray,
    mw_fleet: float,
    grid_load_penalty: float,
    slot_duration_h: float,
) -> float:
    return sum(
        (slot_prices[s] + grid_load_penalty * (current_total[s] + mw_fleet))
        * mw_fleet
        * slot_duration_h
        for s in block
    )


def _scale_app_load(app_mw: np.ndarray, target_app_mwh: float) -> np.ndarray:
    raw_mwh = app_mw.sum() / SLOTS_PER_HOUR
    if raw_mwh <= 0 or target_app_mwh <= 0:
        return app_mw
    return app_mw * (target_app_mwh / raw_mwh)


def optimize_ev_load_constrained(
    base_load: pd.Series,
    slot_prices: pd.Series,
    grid_load_penalty: float,
    arrival_weights: pd.Series,
    app_seed: int,
) -> tuple[pd.Series, pd.Series, float, float, float]:
    rng = np.random.default_rng(app_seed)
    non_app_load_mw = np.zeros(N_SLOTS)
    non_app_active = np.zeros(N_SLOTS, dtype=float)
    app_groups: list[dict] = []
    total_app_evs = 0.0
    total_non_app_evs = 0.0

    for arrival_slot, arr_weight in arrival_weights.items():
        n_arriving = arr_weight * N_EVS
        mix = _charger_mix_for_slot(int(arrival_slot))
        for charger_key, mix_fraction in mix.items():
            n_total = n_arriving * mix_fraction
            n_app = float(rng.binomial(round(n_total), APP_ADOPTION_RATE))
            n_no_app = n_total - n_app
            session_s = _session_slots(charger_key)
            mw_per_ev = CHARGER_KW[charger_key] / 1_000
            allowed = _allowed_slots(int(arrival_slot), charger_key)
            for offset in range(session_s):
                s = (int(arrival_slot) + offset) % N_SLOTS
                non_app_load_mw[s] += n_no_app * mw_per_ev
                non_app_active[s] += n_no_app
            if n_app > 0:
                app_groups.append({
                    "arrival_slot": int(arrival_slot),
                    "allowed": allowed,
                    "session_s": session_s,
                    "mw_fleet": mw_per_ev * n_app,
                    "n_evs": n_app,
                })
                total_app_evs += n_app
            total_non_app_evs += n_no_app

    current_total = base_load.values.copy() + non_app_load_mw
    price_arr = slot_prices.reindex(range(N_SLOTS)).values
    slot_duration_h = 1.0 / SLOTS_PER_HOUR
    app_schedule = np.zeros(N_SLOTS)
    app_active_arr = np.zeros(N_SLOTS, dtype=float)

    for group in app_groups:
        allowed = group["allowed"]
        session_s = group["session_s"]
        mw_fleet = group["mw_fleet"]
        if len(allowed) < session_s:
            session_s = len(allowed)
        best_start_idx = 0
        best_score = float("inf")
        for start_idx in range(len(allowed) - session_s + 1):
            block = allowed[start_idx : start_idx + session_s]
            score = _block_effective_cost(
                block,
                current_total,
                price_arr,
                mw_fleet,
                grid_load_penalty,
                slot_duration_h,
            )
            if score < best_score:
                best_score = score
                best_start_idx = start_idx
        for s in allowed[best_start_idx : best_start_idx + session_s]:
            current_total[s] += mw_fleet
            app_schedule[s] += mw_fleet
            app_active_arr[s] += group["n_evs"]

    non_app_mwh = float(non_app_load_mw.sum() / SLOTS_PER_HOUR)
    app_target_mwh = max(0.0, EV_DAILY_MWH - non_app_mwh)
    app_schedule = _scale_app_load(app_schedule, app_target_mwh)

    app_counterfactual = np.zeros(N_SLOTS)
    for group in app_groups:
        arr = group["arrival_slot"]
        session_s = group["session_s"]
        for offset in range(session_s):
            s = (arr + offset) % N_SLOTS
            app_counterfactual[s] += group["mw_fleet"]
    app_counterfactual = _scale_app_load(app_counterfactual, app_target_mwh)

    slot_duration_h = 1.0 / SLOTS_PER_HOUR
    app_cf_cost = float((app_counterfactual * price_arr * slot_duration_h).sum())
    app_opt_cost = float((app_schedule * price_arr * slot_duration_h).sum())
    daily_app_user_savings = app_cf_cost - app_opt_cost

    opt_ev_load_arr = non_app_load_mw + app_schedule
    opt_ev_active_arr = (non_app_active + app_active_arr).round().astype(int)
    actual_rate = (
        total_app_evs / (total_app_evs + total_non_app_evs)
        if (total_app_evs + total_non_app_evs) > 0
        else 0.0
    )
    return (
        pd.Series(opt_ev_load_arr, index=range(N_SLOTS), name="Opt_EV_Load_MW"),
        pd.Series(opt_ev_active_arr, index=range(N_SLOTS), name="Opt_EVs_Active"),
        float(actual_rate),
        float(daily_app_user_savings),
        float(total_app_evs),
    )


def main() -> None:
    sim_cfg, res_cfg = load_residuals_config()
    _apply_runtime_config(sim_cfg, res_cfg)
    out_dir = sim_cfg.output_dir / "simulation_v5"
    if SAVE_OUTPUTS:
        out_dir.mkdir(parents=True, exist_ok=True)

    grid_15 = load_zone2_15min_demand(sim_cfg)
    zone_capacity_mw, cap_meta = resolve_zone_capacity_mw(grid_15, sim_cfg)
    slot_prices = load_slot_price_profile(sim_cfg)

    arrival_weights = build_arrival_weights()
    slot_mean, slot_std, phi, dof = load_zone2_statistics()
    ev_load_profile, ev_active_profile = build_ev_load_profile(arrival_weights)

    print(f"\n=== Zone {FOCUS_ZONE} simulation V5 (15-min AR1 + smart charging) ===")
    print(
        f"Fleet: {N_EVS:,} EVs x {EV_KWH_PER_DAY} kWh/day = {EV_DAILY_MWH:.1f} MWh/day"
    )
    print(f"Zone capacity: {zone_capacity_mw:.2f} MW ({cap_meta['capacity_source']})")
    print(
        f"Price + grid penalty: EUR {slot_prices.min():.1f}-{slot_prices.max():.1f}/MWh, "
        f"load_penalty={sim_cfg.grid_load_penalty_eur_per_mw:.2f} EUR/MW"
    )
    print(f"App adoption target: {APP_ADOPTION_RATE * 100:.0f}% | Runs: {SIMULATION_REPEATS}")

    slots = range(N_SLOTS)
    xticks = range(0, N_SLOTS, SLOTS_PER_HOUR)
    xlabels = [f"{h:02d}:00" for h in range(24)]
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    colours = ["#378ADD", "#1D9E75", "#D85A30", "#9B59B6", "#E67E22"]
    historical_peak_mw = float(slot_mean.max() + 3 * slot_std.max())
    kpi_rows: list[dict[str, float]] = []

    for sim in range(1, SIMULATION_REPEATS + 1):
        run_seed = RANDOM_SEED + sim
        app_seed = RANDOM_SEED * 100 + sim
        total_demand = simulate_day_ar1(slot_mean, slot_std, phi, dof, seed=run_seed)
        ev_load, base_load, ev_active = decompose_ev_from_demand(
            total_demand, ev_load_profile, ev_active_profile
        )
        opt_ev_load, opt_ev_active, actual_rate, daily_app_savings, n_app_evs = (
            optimize_ev_load_constrained(
                base_load,
                slot_prices,
                sim_cfg.grid_load_penalty_eur_per_mw,
                arrival_weights,
                app_seed,
            )
        )
        opt_total_demand = base_load + opt_ev_load
        peak_unmanaged = float(total_demand.max())
        peak_managed = float(opt_total_demand.max())
        price_arr = slot_prices.reindex(range(N_SLOTS)).values
        unmanaged_ev_cost = daily_ev_energy_cost_slots(
            ev_load.values, price_arr, SLOTS_PER_HOUR
        )
        managed_ev_cost = daily_ev_energy_cost_slots(
            opt_ev_load.values, price_arr, SLOTS_PER_HOUR
        )
        daily_fleet_savings = unmanaged_ev_cost - managed_ev_cost
        annual_fleet_savings = daily_fleet_savings * 365

        daily_per_app_user = (
            daily_app_savings / n_app_evs if n_app_evs > 0 else 0.0
        )
        annual_per_app_user = daily_per_app_user * 365
        monthly_per_app_user = annual_per_app_user / 12
        annual_at_full_adoption = daily_per_app_user * N_EVS * 365
        monthly_per_ev_fleet_avg = annual_fleet_savings / 12 / N_EVS

        unmanaged_stress = total_demand / zone_capacity_mw
        managed_stress = opt_total_demand / zone_capacity_mw
        ref_frame = pd.DataFrame(
            {
                "Stress_Ratio": unmanaged_stress,
                "Bottleneck": unmanaged_stress >= 1.0,
            }
        )
        scen_frame = pd.DataFrame(
            {
                "Stress_Ratio": managed_stress,
                "Bottleneck": managed_stress >= 1.0,
            }
        )
        ref_overload, ref_high = grid_stress_minutes(ref_frame, sim_cfg)
        m_overload, m_high = grid_stress_minutes(scen_frame, sim_cfg)
        stress_integral = 0.0
        if float(managed_stress.max()) <= float(unmanaged_stress.max()):
            stress_integral = congestion_stress_integral_saved(
                scen_frame, ref_frame, sim_cfg
            )
        annual_dso_savings = annual_dso_savings_eur(
            m_overload,
            m_high,
            float(managed_stress.max()),
            ref_overload,
            ref_high,
            float(unmanaged_stress.max()),
            sim_cfg,
            congestion_stress_integral_saved=stress_integral,
        )

        kpi_rows.append({
            "run_id": sim,
            "app_adoption_target": APP_ADOPTION_RATE,
            "app_adoption_actual": actual_rate,
            "zone_capacity_mw": zone_capacity_mw,
            "peak_unmanaged_mw": peak_unmanaged,
            "peak_managed_mw": peak_managed,
            "peak_reduction_mw": peak_unmanaged - peak_managed,
            "peak_stress_unmanaged": peak_unmanaged / zone_capacity_mw,
            "peak_stress_managed": peak_managed / zone_capacity_mw,
            "ev_energy_managed_mwh": float(opt_ev_load.sum() / SLOTS_PER_HOUR),
            "ev_energy_target_mwh": EV_DAILY_MWH,
            "daily_unmanaged_ev_cost_eur": unmanaged_ev_cost,
            "daily_managed_ev_cost_eur": managed_ev_cost,
            "daily_fleet_savings_eur": daily_fleet_savings,
            "daily_app_user_savings_eur": daily_app_savings,
            "daily_savings_per_app_user_eur": daily_per_app_user,
            "annual_fleet_savings_eur": annual_fleet_savings,
            "annual_customer_savings_eur": annual_at_full_adoption,
            "annual_savings_per_app_user_eur": annual_per_app_user,
            "monthly_savings_per_app_user_eur": monthly_per_app_user,
            "monthly_savings_per_ev_eur": monthly_per_ev_fleet_avg,
            "app_users_count": n_app_evs,
            "annual_dso_savings_eur": annual_dso_savings,
        })
        if SAVE_OUTPUTS:
            rows = []
            for s in slots:
                rows.append({
                    "slot": s,
                    "time": slot_label(s),
                    "stochastic_grid_mw": float(total_demand[s]),
                    "base_load_mw": float(base_load[s]),
                    "unmanaged_ev_mw": float(ev_load[s]),
                    "unmanaged_total_mw": float(total_demand[s]),
                    "managed_ev_mw": float(opt_ev_load[s]),
                    "managed_total_mw": float(opt_total_demand[s]),
                    "unmanaged_stress": float(total_demand[s]) / zone_capacity_mw,
                    "managed_stress": float(opt_total_demand[s]) / zone_capacity_mw,
                    "unmanaged_evs_active": int(ev_active[s]),
                    "managed_evs_active": int(opt_ev_active[s]),
                    "price_eur_mwh": float(slot_prices[s]),
                })
            pd.DataFrame(rows).to_csv(out_dir / f"run_{sim:03d}_15min.csv", index=False)

        colour = colours[(sim - 1) % len(colours)]
        axes[0].plot(slots, total_demand, color=colour, lw=1.5, ls="-", label=f"Run {sim} unmanaged")
        axes[0].plot(slots, opt_total_demand, color=colour, lw=1.5, ls="--", label=f"Run {sim} managed ({actual_rate*100:.0f}% app)")
        axes[1].plot(slots, ev_active, color=colour, lw=1.5, ls="-")
        axes[1].plot(slots, opt_ev_active, color=colour, lw=1.5, ls="--")
        print(
            f"Run {sim}: adoption {actual_rate*100:.1f}% | peak {peak_unmanaged:.1f} -> {peak_managed:.1f} MW"
        )

    axes[0].axhline(zone_capacity_mw, color="#dc2626", ls=":", lw=1.5, label=f"Zone capacity ({zone_capacity_mw:.0f} MW)")
    axes[0].plot(slots, slot_mean, color="black", lw=1.5, ls=":", label="Historical mean")
    axes[1].axhline(N_EVS, color="#6b7280", ls=":", lw=1.2, label="Fleet size")
    axes[0].set_ylabel("Demand (MW)")
    axes[0].set_title(f"Zone {FOCUS_ZONE}: 15-min AR(1) unmanaged vs managed ({APP_ADOPTION_RATE*100:.0f}% app target)")
    axes[1].set_ylabel("EVs charging")
    axes[1].set_xlabel("Time of day")
    axes[1].set_xticks(list(xticks))
    axes[1].set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    axes[0].legend(loc="lower left", fontsize=7, ncol=2)
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[1].grid(True, ls=":", alpha=0.5)
    fig.tight_layout()

    graphs_dir = sim_cfg.ensure_graphs_dir()
    chart_path = graphs_dir / "simulation_v5_15min.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart: {chart_path}")

    if SAVE_OUTPUTS:
        kpi_df = pd.DataFrame(kpi_rows)
        kpi_df.to_csv(out_dir / "simulation_v5_kpi.csv", index=False)
        print(f"CSVs: {out_dir}")
        print(f"Mean peak reduction: {kpi_df['peak_reduction_mw'].mean():.2f} MW")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
