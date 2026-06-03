from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

# ─────────────────────────────────────────────
#  CONFIG  — edit these to change the scenario
# ─────────────────────────────────────────────
CONFIDENCE_INTERVAL = 0.95
SIMULATION_REPEATS  = 1
RANDOM_SEED         = 67

N_EVS           = 15000   # number of EVs in the system
EV_KWH_PER_DAY  = 25      # average daily energy consumption per EV (kWh)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_Set" / "Data_Set"
DS6_DIR  = next(DATA_DIR.glob("Dataset 6*"))

ZONAL_LOAD_FILE = DS6_DIR / "eindhoven_zonal_load.csv"
FOCUS_ZONE      = "Z2"

# ── Smart-charging app adoption ───────────────────────────────────────────────
# Fraction of EVs that have the smart-charging app installed.
# Only this share participates in the water-filling optimisation;
# the rest follows the unmanaged behaviour-based profile.
# These users are drawn randomly each simulation run (not the first N%).
APP_ADOPTION_RATE = 0.60   # e.g. 0.30 = 30 % of the fleet uses the app

# ── Optimiser charging-window constraints ─────────────────────────────────────
# Morning arrivals  (06:00–13:00): must finish charging within this many hours.
MORNING_MAX_WINDOW_H = 4

# Evening / overnight arrivals (14:00–05:59): must finish by this hour next day.
EVENING_DEADLINE_H   = 6   # 06:00

# derived — no need to touch
EV_DAILY_MWH = N_EVS * EV_KWH_PER_DAY / 1_000


# ─────────────────────────────────────────────
#  CHARGER TYPES
# ─────────────────────────────────────────────
CHARGER_KW = {
    "Home":  11,
    "Fast":  50,
    "Super": 150,
}

# ─────────────────────────────────────────────
#  CHARGER MIX PER PEAK  (must sum to 1.0)
# ─────────────────────────────────────────────
MORNING_CHARGER_MIX = {
    "Home":  0.40,
    "Fast":  0.50,
    "Super": 0.10,
}

EVENING_CHARGER_MIX = {
    "Home":  0.75,
    "Fast":  0.20,
    "Super": 0.05,
}

assert abs(sum(MORNING_CHARGER_MIX.values()) - 1.0) < 1e-6, "MORNING_CHARGER_MIX must sum to 1.0"
assert abs(sum(EVENING_CHARGER_MIX.values()) - 1.0) < 1e-6, "EVENING_CHARGER_MIX must sum to 1.0"
assert MORNING_CHARGER_MIX.keys() == CHARGER_KW.keys()
assert EVENING_CHARGER_MIX.keys() == CHARGER_KW.keys()
assert 0.0 <= APP_ADOPTION_RATE <= 1.0, "APP_ADOPTION_RATE must be between 0 and 1"


# ─────────────────────────────────────────────
#  ARRIVAL DISTRIBUTION
# ─────────────────────────────────────────────

def build_arrival_weights() -> pd.Series:
    hours = np.arange(24)
    components = [
        (9.0,  1.5, 0.30),
        (18.5, 1.8, 0.55),
        (2.0,  2.0, 0.15),
    ]
    profile = np.zeros(24)
    for mu, sigma, weight in components:
        profile += weight * np.exp(-0.5 * ((hours - mu) / sigma) ** 2)
    profile /= profile.sum()
    return pd.Series(profile, index=range(24), name="arrival_weight")


ARRIVAL_WEIGHTS = build_arrival_weights()


def _session_hours(charger_key: str) -> int:
    return max(1, math.ceil(EV_KWH_PER_DAY / CHARGER_KW[charger_key]))


def _peak_label(hour: int) -> str:
    if 6 <= hour < 14:
        return "morning"
    return "evening"   # covers 14–23 and 0–5


def _charger_mix_for_hour(hour: int) -> dict:
    return MORNING_CHARGER_MIX if _peak_label(hour) == "morning" else EVENING_CHARGER_MIX


def _allowed_hours(arrival_hour: int, charger_key: str) -> list[int]:
    """
    Returns the list of whole hours (mod 24) during which this EV *may* charge,
    respecting the window constraint for its peak type.

    Morning arrivals : window = [arrival, arrival + MORNING_MAX_WINDOW_H)
    Evening arrivals : window = [arrival, EVENING_DEADLINE_H next day)
                       i.e. the car must be done by 06:00.
    In both cases the window is capped to be at least session_hours wide so the
    car can always complete its charge.
    """
    session_h = _session_hours(charger_key)

    if _peak_label(arrival_hour) == "morning":
        window_h = max(session_h, MORNING_MAX_WINDOW_H)
        return [(arrival_hour + i) % 24 for i in range(window_h)]
    else:
        # hours from arrival until 06:00 next day (wrapping)
        hours_until_deadline = (EVENING_DEADLINE_H - arrival_hour) % 24
        # guarantee at least session_h hours
        window_h = max(session_h, hours_until_deadline)
        return [(arrival_hour + i) % 24 for i in range(window_h)]


# ─────────────────────────────────────────────
#  UNMANAGED EV LOAD PROFILE
# ─────────────────────────────────────────────

def build_ev_load_profile() -> tuple[pd.Series, pd.Series]:
    """Behaviour-based profile: EVs start charging immediately on arrival."""
    ev_load_mw = np.zeros(24)
    ev_active  = np.zeros(24, dtype=float)

    for arrival_hour, arr_weight in ARRIVAL_WEIGHTS.items():
        n_arriving = arr_weight * N_EVS
        mix        = _charger_mix_for_hour(arrival_hour)

        for charger_key, mix_fraction in mix.items():
            n_evs_this_type = n_arriving * mix_fraction
            kw_per_ev       = CHARGER_KW[charger_key]
            session_h       = _session_hours(charger_key)
            mw_per_ev_per_h = kw_per_ev / 1_000

            for offset in range(session_h):
                target_hour = (arrival_hour + offset) % 24
                ev_load_mw[target_hour] += n_evs_this_type * mw_per_ev_per_h
                ev_active[target_hour]  += n_evs_this_type

    raw_energy = ev_load_mw.sum()
    scale      = EV_DAILY_MWH / raw_energy
    ev_load_mw *= scale

    return (
        pd.Series(ev_load_mw, index=range(24), name="EV_Load_MW"),
        pd.Series(ev_active.round().astype(int), index=range(24), name="EVs_Active"),
    )


EV_LOAD_PROFILE, EV_ACTIVE_PROFILE = build_ev_load_profile()


# ─────────────────────────────────────────────
#  DATA LOADING & AR(1) PARAMETER ESTIMATION
# ─────────────────────────────────────────────

def load_zone2_statistics() -> tuple[pd.Series, pd.Series, float, int]:
    load_df = pd.read_csv(ZONAL_LOAD_FILE)
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"], errors="coerce")
    load_df = load_df.dropna(subset=["timestamp"]).copy()

    zone2       = load_df.loc[load_df["zone_id"] == FOCUS_ZONE].copy()
    zone2["hour"] = zone2["timestamp"].dt.hour

    grouped     = zone2.groupby("hour")["demand_MW"]
    hourly_mean = grouped.mean().reindex(range(24)).interpolate()
    hourly_std  = grouped.std().reindex(range(24)).interpolate()

    zone2["date"] = zone2["timestamp"].dt.date
    pivot         = zone2.pivot(index="date", columns="hour", values="demand_MW")
    residuals     = pivot - hourly_mean.values
    flat          = residuals.values.flatten()
    phi           = np.corrcoef(flat[:-1], flat[1:])[0, 1]
    dof           = len(pivot) - 1

    print(f"Estimated AR(1) phi = {phi:.4f}  |  DoF = {dof}")
    return hourly_mean, hourly_std, phi, dof


# ─────────────────────────────────────────────
#  AR(1) DEMAND SIMULATION
# ─────────────────────────────────────────────

def simulate_day_ar1(
    hourly_mean: pd.Series,
    hourly_std:  pd.Series,
    phi:         float,
    dof:         int,
    confidence:  float = CONFIDENCE_INTERVAL,
    seed:        int | None = None,
) -> pd.Series:
    alpha  = 1 - confidence
    t_crit = t.ppf(1 - alpha / 2, df=dof)
    rng    = np.random.RandomState(seed)

    simulated = []
    prev_eps  = 0.0
    for hour in range(24):
        mu    = hourly_mean[hour]
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


# ─────────────────────────────────────────────
#  EV DECOMPOSITION
# ─────────────────────────────────────────────

def decompose_ev_from_demand(total_demand: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ev_load   = EV_LOAD_PROFILE
    base_load = (total_demand - ev_load).clip(lower=0).rename("Base_Load_MW")
    ev_active = EV_ACTIVE_PROFILE
    return ev_load, base_load, ev_active


# ─────────────────────────────────────────────
#  CONSTRAINED WATER-FILLING OPTIMISER
# ─────────────────────────────────────────────

def optimize_ev_load_constrained(
    base_load:    pd.Series,
    total_ev_mwh: float,
    n_evs:        int,
    app_seed:     int,
) -> tuple[pd.Series, pd.Series, float]:
    """
    Distributes EV energy to flatten total demand, subject to:
      - Only APP_ADOPTION_RATE of the fleet participates (random draw).
      - Morning arrivals must finish within MORNING_MAX_WINDOW_H hours.
      - Evening arrivals must finish by EVENING_DEADLINE_H.

    The non-app users retain their unmanaged (behaviour-based) load.

    Returns:
        opt_ev_load   — optimised EV load profile (MW) for the whole fleet
        opt_ev_active — active EV count for the whole fleet
        actual_rate   — realised app adoption fraction (informational)
    """
    rng = np.random.default_rng(app_seed)

    # ── Step 1: Split fleet into app / non-app users ──────────────────────
    # For each (arrival_hour, charger_type) group, randomly assign app fraction.
    # We track MW and counts separately so the non-app part is frozen.

    non_app_load_mw = np.zeros(24)
    non_app_active  = np.zeros(24, dtype=float)

    # app_groups: list of (allowed_hours, mw_per_ev_per_h, session_h, n_app_evs)
    # We will water-fill across these groups.
    app_groups = []

    total_app_evs    = 0.0
    total_non_app_evs = 0.0

    for arrival_hour, arr_weight in ARRIVAL_WEIGHTS.items():
        n_arriving = arr_weight * N_EVS
        mix        = _charger_mix_for_hour(arrival_hour)

        for charger_key, mix_fraction in mix.items():
            n_total         = n_arriving * mix_fraction
            # Random split: each EV independently has APP_ADOPTION_RATE chance
            # We model this as a binomial draw for each group.
            n_app    = float(rng.binomial(int(round(n_total)), APP_ADOPTION_RATE))
            n_no_app = n_total - n_app

            kw_per_ev       = CHARGER_KW[charger_key]
            session_h       = _session_hours(charger_key)
            mw_per_ev_per_h = kw_per_ev / 1_000
            allowed         = _allowed_hours(arrival_hour, charger_key)

            # Non-app: immediate charging (same as unmanaged profile for this group)
            for offset in range(session_h):
                h = (arrival_hour + offset) % 24
                non_app_load_mw[h] += n_no_app * mw_per_ev_per_h
                non_app_active[h]  += n_no_app

            # App: register as a flexible group to be scheduled
            if n_app > 0:
                energy_needed_mwh = n_app * EV_KWH_PER_DAY / 1_000
                app_groups.append({
                    "allowed":         allowed,      # hours this group MAY charge
                    "session_h":       session_h,    # minimum consecutive hours needed
                    "mw_per_ev_per_h": mw_per_ev_per_h,
                    "n_evs":           n_app,
                    "energy_mwh":      energy_needed_mwh,
                })
                total_app_evs += n_app

            total_non_app_evs += n_no_app

    # ── Step 2: Water-fill app groups within their allowed windows ────────
    # Build a 24-h load array starting from non-app + base load.
    # For each app group, find the cheapest hours within its window to fill.

    current_total = base_load.values.copy() + non_app_load_mw

    app_schedule  = np.zeros(24)   # MW scheduled for app users
    app_active_arr = np.zeros(24, dtype=float)

    for group in app_groups:
        allowed    = group["allowed"]
        session_h  = group["session_h"]
        mw_per_h   = group["mw_per_ev_per_h"] * group["n_evs"]
        energy_mwh = group["energy_mwh"]

        # Greedy: assign each charging-hour of this group to the cheapest
        # available slot in its window, placing exactly session_h consecutive
        # hours (to honour the "plug in and charge" reality — not scattered).
        # Find the consecutive window-start that minimises the peak in that block.

        best_start_idx = 0
        best_peak      = float("inf")

        # Slide a window of session_h slots over the allowed hours
        for start_idx in range(len(allowed) - session_h + 1):
            block = allowed[start_idx: start_idx + session_h]
            peak  = max(current_total[h] + mw_per_h for h in block)
            if peak < best_peak:
                best_peak      = peak
                best_start_idx = start_idx

        # Assign this group to its best block
        chosen_block = allowed[best_start_idx: best_start_idx + session_h]
        for h in chosen_block:
            current_total[h]  += mw_per_h
            app_schedule[h]   += mw_per_h
            app_active_arr[h] += group["n_evs"]

    # ── Step 3: Combine app + non-app ─────────────────────────────────────
    opt_ev_load_arr = non_app_load_mw + app_schedule

    # Scale to ensure energy conservation (rounding may cause tiny drift)
    raw = opt_ev_load_arr.sum()
    if raw > 0:
        opt_ev_load_arr *= (total_ev_mwh / raw)

    opt_ev_active_arr = (non_app_active + app_active_arr).round().astype(int)

    actual_rate = total_app_evs / (total_app_evs + total_non_app_evs) if N_EVS > 0 else 0.0

    return (
        pd.Series(opt_ev_load_arr, index=range(24), name="Opt_EV_Load_MW"),
        pd.Series(opt_ev_active_arr, index=range(24), name="Opt_EVs_Active"),
        actual_rate,
    )


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main() -> None:
    hourly_mean, hourly_std, phi, dof = load_zone2_statistics()

    print(f"\nEV fleet  : {N_EVS:,} vehicles × {EV_KWH_PER_DAY} kWh/day "
          f"= {EV_DAILY_MWH:.1f} MWh total")
    print(f"App adoption target : {APP_ADOPTION_RATE*100:.0f}%  "
          f"(randomised per run via binomial draw)\n")

    print("Charger types & session durations:")
    for key, kw in CHARGER_KW.items():
        print(f"  {key:<6} {kw:>4} kW  →  {_session_hours(key)} hour(s) per session")

    print(f"\nCharger mix — Morning (06:00–13:00), window ≤ {MORNING_MAX_WINDOW_H}h:")
    for key, frac in MORNING_CHARGER_MIX.items():
        print(f"  {key:<6} {frac*100:.0f}%")

    print(f"\nCharger mix — Evening/overnight, deadline {EVENING_DEADLINE_H:02d}:00:")
    for key, frac in EVENING_CHARGER_MIX.items():
        print(f"  {key:<6} {frac*100:.0f}%")

    print("\nUnmanaged EV load profile (behaviour-based, all runs identical):")
    print(f"  {'Hour':<6} {'EV MW':>8} {'Active EVs':>12}")
    for h in range(24):
        bar = "█" * int(EV_ACTIVE_PROFILE[h] / N_EVS * 40)
        print(f"  {h:02d}:00  {EV_LOAD_PROFILE[h]:>8.2f} {EV_ACTIVE_PROFILE[h]:>12,}  {bar}")
    print()

    hours  = range(24)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    colours   = ["#378ADD", "#1D9E75", "#D85A30"]

    for sim in range(1, SIMULATION_REPEATS + 1):
        run_seed = RANDOM_SEED + sim
        # Use a different sub-seed for app assignment so it is independent of
        # the AR(1) demand seed but still reproducible.
        app_seed = RANDOM_SEED * 100 + sim

        total_demand = simulate_day_ar1(hourly_mean, hourly_std, phi, dof, seed=run_seed)
        ev_load, base_load, ev_active = decompose_ev_from_demand(total_demand)

        opt_ev_load, opt_ev_active, actual_rate = optimize_ev_load_constrained(
            base_load, EV_DAILY_MWH, N_EVS, app_seed
        )
        opt_total_demand = base_load + opt_ev_load

        zone_capacity_mw = float(total_demand.max())
        colour = colours[sim - 1]

        # ── TOP PANEL ──
        axes[0].plot(hours, total_demand, color=colour, linewidth=2, linestyle="-",
                     label=f"Run {sim} Baseline Demand")
        axes[0].fill_between(hours, base_load, total_demand, color=colour, alpha=0.08)
        axes[0].plot(hours, opt_total_demand, color=colour, linewidth=2, linestyle="--",
                     label=f"Run {sim} Managed Demand ({actual_rate*100:.1f}% app)")

        # ── BOTTOM PANEL ──
        axes[1].plot(hours, ev_active, color=colour, linewidth=2, linestyle="-",
                     label=f"Run {sim} Unmanaged")
        axes[1].plot(hours, opt_ev_active, color=colour, linewidth=2, linestyle="--",
                     label=f"Run {sim} Managed ({actual_rate*100:.1f}% app)")

        # ── CONSOLE OUTPUT ──
        print(f"{'─'*95}")
        print(f"  Run {sim}  |  Fleet = {EV_DAILY_MWH:.2f} MWh  |  "
              f"App users: {actual_rate*100:.1f}% (target {APP_ADOPTION_RATE*100:.0f}%)")
        print(f"{'─'*95}")
        print(f"  {'Hour':<6} {'Base MW':>9} {'Unmgd Tot':>10} {'Unmgd EVs':>10} | "
              f"{'Mgd Tot':>10} {'Mgd EVs':>10}")
        print(f"  {'':─<6} {'':─>9} {'':─>10} {'':─>10} ┼ {'':─>10} {'':─>10}")
        for h in hours:
            print(f"  {h:02d}:00  {base_load[h]:>9.1f} {total_demand[h]:>10.1f} "
                  f"{ev_active[h]:>10,} | "
                  f"{opt_total_demand[h]:>10.1f} {opt_ev_active[h]:>10,}")
        print()

    # ── DECORATION ──
    axes[0].axhline(zone_capacity_mw, color="red", linestyle=":", linewidth=1.5,
                    label="Zone Capacity Limit")
    axes[1].axhline(N_EVS, color="purple", linestyle=":", linewidth=1.5,
                    label="Total EV Fleet Size")
    axes[0].plot(hours, hourly_mean, color="black", linewidth=2,
                 linestyle=":", label="Historical mean")

    m_str = "  ".join(f"{k} {v*100:.0f}%" for k, v in MORNING_CHARGER_MIX.items())
    e_str = "  ".join(f"{k} {v*100:.0f}%" for k, v in EVENING_CHARGER_MIX.items())
    axes[0].set_title(
        f"Zone {FOCUS_ZONE}: AR(1) Demand — Unmanaged vs. Constrained Smart Charging "
        f"({APP_ADOPTION_RATE*100:.0f}% app adoption, randomised)\n"
        f"Morning mix: {m_str}   |   Evening mix: {e_str}\n"
        f"Morning window ≤ {MORNING_MAX_WINDOW_H}h  |  Evening deadline {EVENING_DEADLINE_H:02d}:00",
        fontsize=9,
    )
    axes[0].set_ylabel("Demand (MW)")
    axes[0].legend(loc="lower left", fontsize=8, ncol=2)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].set_ylim(bottom=1800, top=zone_capacity_mw * 1.15)

    axes[1].set_ylabel("EVs Actively Charging (Count)")
    axes[1].set_xlabel("Hour of Day (0–23)")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].set_ylim(bottom=0, top=N_EVS * 1.1)

    plt.xticks(range(24))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()