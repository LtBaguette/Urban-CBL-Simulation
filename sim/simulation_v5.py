from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import t

from .capacity import resolve_zone_capacity_mw
from .config import SimConfig, load_config
from .data_loaders import load_mean_hourly_price_profile, load_zone2_15min_demand
from .dso_value import annual_dso_savings_eur
from .metrics import (
    congestion_stress_integral_saved,
    daily_ev_energy_cost_slots,
    grid_stress_minutes,
)
from .paths import ZONAL_LOAD_FILE
from .residuals import ResidualsConfig, load_residuals_config

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
V5_PRICE_WEIGHT = 1.0
V5_GRID_WEIGHT = 1.0
V5_GRID_LOAD_PENALTY = 0.75
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
    global V5_PRICE_WEIGHT, V5_GRID_WEIGHT, V5_GRID_LOAD_PENALTY

    CONFIDENCE_INTERVAL = res_cfg.confidence_interval
    SIMULATION_REPEATS = res_cfg.v5_simulation_repeats
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
    V5_PRICE_WEIGHT = res_cfg.v5_price_weight
    V5_GRID_WEIGHT = res_cfg.v5_grid_weight
    V5_GRID_LOAD_PENALTY = res_cfg.v5_grid_load_penalty_eur_per_mw
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


@dataclass(frozen=True)
class PriceSpike:
    """Temporary wholesale price spike on one or more 15-min slots."""

    start_hour: int
    start_minute: int
    duration_minutes: int
    price_eur_mwh: float

    def slot_indices(self) -> list[int]:
        start_slot = self.start_hour * SLOTS_PER_HOUR + self.start_minute // SLOT_MINUTES
        n_slots = max(1, self.duration_minutes // SLOT_MINUTES)
        return [(start_slot + i) % N_SLOTS for i in range(n_slots)]

    def label(self) -> str:
        end_total = self.start_hour * 60 + self.start_minute + self.duration_minutes
        end_h, end_m = divmod(end_total, 60)
        return (
            f"{self.start_hour:02d}:{self.start_minute:02d}"
            f"-{end_h % 24:02d}:{end_m:02d}"
        )


def apply_price_spike(slot_prices: pd.Series, spike: PriceSpike) -> pd.Series:
    out = slot_prices.copy()
    for slot in spike.slot_indices():
        out[slot] = spike.price_eur_mwh
    return out


DEFAULT_PRICE_SPIKE = PriceSpike(
    start_hour=3,
    start_minute=0,
    duration_minutes=15,
    price_eur_mwh=500.0,
)


def _block_effective_cost(
    block: list[int],
    current_total: np.ndarray,
    slot_prices: np.ndarray,
    mw_fleet: float,
    grid_load_penalty: float,
    slot_duration_h: float,
    price_weight: float = 1.0,
    grid_weight: float = 1.0,
) -> float:
    return sum(
        (
            price_weight * slot_prices[s]
            + grid_weight * grid_load_penalty * (current_total[s] + mw_fleet)
        )
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
    *,
    price_weight: float | None = None,
    grid_weight: float | None = None,
) -> tuple[pd.Series, pd.Series, float, float, float]:
    price_weight = V5_PRICE_WEIGHT if price_weight is None else price_weight
    grid_weight = V5_GRID_WEIGHT if grid_weight is None else grid_weight
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
                price_weight,
                grid_weight,
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


def run_simulation_v5(
    price_spike: PriceSpike | None = None,
    *,
    output_subdir: str = "simulation_v5",
    chart_filename: str = "simulation_v5_15min.png",
) -> Path:
    sim_cfg, res_cfg = load_residuals_config()
    _apply_runtime_config(sim_cfg, res_cfg)
    out_dir = sim_cfg.output_dir / output_subdir
    if SAVE_OUTPUTS:
        out_dir.mkdir(parents=True, exist_ok=True)

    grid_15 = load_zone2_15min_demand(sim_cfg)
    zone_capacity_mw, cap_meta = resolve_zone_capacity_mw(grid_15, sim_cfg)
    baseline_slot_prices = load_slot_price_profile(sim_cfg)
    slot_prices = (
        apply_price_spike(baseline_slot_prices, price_spike)
        if price_spike is not None
        else baseline_slot_prices
    )
    spike_slots = price_spike.slot_indices() if price_spike else []

    arrival_weights = build_arrival_weights()
    slot_mean, slot_std, phi, dof = load_zone2_statistics()
    ev_load_profile, ev_active_profile = build_ev_load_profile(arrival_weights)

    experiment_label = (
        f"price spike {price_spike.label()} @ EUR {price_spike.price_eur_mwh:.0f}/MWh"
        if price_spike
        else "15-min AR1 + smart charging"
    )
    print(f"\n=== Zone {FOCUS_ZONE} el diablo ({experiment_label}) ===")
    print(
        f"Fleet: {N_EVS:,} EVs x {EV_KWH_PER_DAY} kWh/day = {EV_DAILY_MWH:.1f} MWh/day"
    )
    print(f"Zone capacity: {zone_capacity_mw:.2f} MW ({cap_meta['capacity_source']})")
    print(
        f"Price + grid objective: EUR {slot_prices.min():.1f}-{slot_prices.max():.1f}/MWh, "
        f"weights price={res_cfg.v5_price_weight:.2f} grid={res_cfg.v5_grid_weight:.2f}, "
        f"grid_penalty={res_cfg.v5_grid_load_penalty_eur_per_mw:.2f} EUR/MW"
    )
    if price_spike:
        for slot in spike_slots:
            print(
                f"  Spike slot {slot_label(slot)}: "
                f"EUR {baseline_slot_prices[slot]:.1f} -> EUR {slot_prices[slot]:.1f}/MWh"
            )
    print(f"App adoption target: {APP_ADOPTION_RATE * 100:.0f}% | Runs: {SIMULATION_REPEATS}")

    slots = range(N_SLOTS)
    xticks = range(0, N_SLOTS, SLOTS_PER_HOUR)
    xlabels = [f"{h:02d}:00" for h in range(24)]
    n_rows = 3 if price_spike else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 10 if price_spike else 9), sharex=True)
    if n_rows == 2:
        axes = list(axes)
    colours = ["#378ADD", "#1D9E75", "#D85A30", "#9B59B6", "#E67E22"]
    kpi_rows: list[dict[str, float]] = []
    plot_individual_runs = SIMULATION_REPEATS <= 5
    unmanaged_totals: list[np.ndarray] = []
    managed_totals: list[np.ndarray] = []
    unmanaged_active: list[np.ndarray] = []
    managed_active: list[np.ndarray] = []

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
                res_cfg.v5_grid_load_penalty_eur_per_mw,
                arrival_weights,
                app_seed,
                price_weight=res_cfg.v5_price_weight,
                grid_weight=res_cfg.v5_grid_weight,
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
            peak_total_load_mw=peak_managed,
            reference_peak_total_load_mw=peak_unmanaged,
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
        unmanaged_totals.append(total_demand.values.copy())
        managed_totals.append(opt_total_demand.values.copy())
        unmanaged_active.append(ev_active.values.copy())
        managed_active.append(opt_ev_active.values.copy())
        if plot_individual_runs:
            axes[0].plot(slots, total_demand, color=colour, lw=1.5, ls="-", label=f"Run {sim} unmanaged")
            axes[0].plot(
                slots,
                opt_total_demand,
                color=colour,
                lw=1.5,
                ls="--",
                label=f"Run {sim} managed ({actual_rate*100:.0f}% app)",
            )
            axes[1].plot(slots, ev_active, color=colour, lw=1.5, ls="-")
            axes[1].plot(slots, opt_ev_active, color=colour, lw=1.5, ls="--")
        if price_spike and spike_slots:
            spike_slot = spike_slots[0]
            managed_at_spike = int(opt_ev_active[spike_slot])
            unmanaged_at_spike = int(ev_active[spike_slot])
            print(
                f"Run {sim}: adoption {actual_rate*100:.1f}% | peak {peak_unmanaged:.1f} -> "
                f"{peak_managed:.1f} MW | EVs at spike {slot_label(spike_slot)}: "
                f"{unmanaged_at_spike} unmg / {managed_at_spike} mgd"
            )
        else:
            print(
                f"Run {sim}: adoption {actual_rate*100:.1f}% | peak {peak_unmanaged:.1f} -> "
                f"{peak_managed:.1f} MW"
            )

    if not plot_individual_runs:
        mean_unmanaged = np.mean(unmanaged_totals, axis=0)
        mean_managed = np.mean(managed_totals, axis=0)
        mean_ev_unmanaged = np.mean(unmanaged_active, axis=0)
        mean_ev_managed = np.mean(managed_active, axis=0)
        axes[0].plot(
            slots,
            mean_unmanaged,
            color="#378ADD",
            lw=2,
            ls="-",
            label=f"Mean unmanaged ({SIMULATION_REPEATS} runs)",
        )
        axes[0].plot(
            slots,
            mean_managed,
            color="#378ADD",
            lw=2,
            ls="--",
            label=f"Mean managed ({SIMULATION_REPEATS} runs)",
        )
        axes[1].plot(slots, mean_ev_unmanaged, color="#378ADD", lw=2, ls="-")
        axes[1].plot(slots, mean_ev_managed, color="#378ADD", lw=2, ls="--")

    if price_spike and spike_slots:
        for slot in spike_slots:
            axes[0].axvspan(
                slot - 0.5,
                slot + 0.5,
                color="#fecaca",
                alpha=0.35,
                zorder=0,
            )
            axes[1].axvspan(
                slot - 0.5,
                slot + 0.5,
                color="#fecaca",
                alpha=0.35,
                zorder=0,
            )

    axes[0].axhline(zone_capacity_mw, color="#dc2626", ls=":", lw=1.5, label=f"Zone capacity ({zone_capacity_mw:.0f} MW)")
    axes[0].plot(slots, slot_mean, color="black", lw=1.5, ls=":", label="Historical mean")
    axes[1].axhline(N_EVS, color="#6b7280", ls=":", lw=1.2, label="Fleet size")
    axes[0].set_ylabel("Demand (MW)")
    title = (
        f"Zone {FOCUS_ZONE}: el diablo — 15-min AR(1) unmanaged vs managed "
        f"({APP_ADOPTION_RATE*100:.0f}% app target, {SIMULATION_REPEATS} seeds)"
    )
    if price_spike:
        title += f"\nPrice spike test: {price_spike.label()} @ EUR {price_spike.price_eur_mwh:.0f}/MWh (shaded)"
    axes[0].set_title(title)
    axes[1].set_ylabel("EVs charging")
    if price_spike:
        price_ax = axes[2]
        price_ax.plot(
            slots,
            baseline_slot_prices,
            color="#9ca3af",
            lw=1.2,
            ls="--",
            label="Baseline wholesale",
        )
        price_ax.plot(
            slots,
            slot_prices,
            color="#dc2626",
            lw=1.8,
            label="Spiked price profile",
        )
        for slot in spike_slots:
            price_ax.axvspan(
                slot - 0.5,
                slot + 0.5,
                color="#fecaca",
                alpha=0.35,
                zorder=0,
            )
        price_ax.set_ylabel("Price (EUR/MWh)")
        price_ax.legend(loc="upper right", fontsize=7)
        price_ax.grid(True, ls=":", alpha=0.5)
        price_ax.set_xlabel("Time of day")
        price_ax.set_xticks(list(xticks))
        price_ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    else:
        axes[1].set_xlabel("Time of day")
        axes[1].set_xticks(list(xticks))
        axes[1].set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    axes[0].legend(loc="lower left", fontsize=7, ncol=2)
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[1].grid(True, ls=":", alpha=0.5)
    fig.tight_layout()

    graphs_dir = sim_cfg.ensure_graphs_dir()
    chart_path = graphs_dir / chart_filename
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart: {chart_path}")

    if SAVE_OUTPUTS:
        kpi_df = pd.DataFrame(kpi_rows)
        kpi_name = (
            "simulation_v5_price_spike_kpi.csv"
            if price_spike
            else "simulation_v5_kpi.csv"
        )
        kpi_df.to_csv(out_dir / kpi_name, index=False)
        print(f"CSVs: {out_dir}")
        print(f"Mean peak reduction: {kpi_df['peak_reduction_mw'].mean():.2f} MW")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
    return chart_path


def main() -> None:
    run_simulation_v5()


def main_price_spike(
    spike: PriceSpike | None = None,
) -> Path:
    return run_simulation_v5(
        price_spike=spike or DEFAULT_PRICE_SPIKE,
        output_subdir="simulation_v5_price_spike",
        chart_filename="simulation_v5_15min_price_spike.png",
    )


if __name__ == "__main__":
    main()
