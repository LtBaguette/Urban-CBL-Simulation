"""Grid-only block scheduler — V5 heuristic with price term removed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simulation_v5 import (
    CHARGER_KW,
    N_SLOTS,
    SLOTS_PER_HOUR,
    _allowed_slots,
    _charger_mix_for_slot,
    _scale_app_load,
    _session_slots,
)


def _block_grid_cost(
    block: list[int],
    current_total: np.ndarray,
    mw_fleet: float,
    grid_load_penalty: float,
    slot_duration_h: float,
) -> float:
    return sum(
        grid_load_penalty * (current_total[s] + mw_fleet) * mw_fleet * slot_duration_h
        for s in block
    )


def optimize_ev_load_grid_oriented(
    base_load: pd.Series,
    arrival_weights: pd.Series,
    grid_load_penalty: float,
    app_seed: int,
    *,
    n_evs: int,
    total_ev_mwh: float,
    app_adoption_rate: float,
    charger_kw: dict[str, float] | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    """
    Grid-optimized scheduler: each app group picks the allowed consecutive block
    that minimises penalised grid load (ignores electricity price).
    """
    charger_kw = charger_kw or CHARGER_KW
    rng = np.random.default_rng(app_seed)
    slot_duration_h = 1.0 / SLOTS_PER_HOUR

    non_app_load_mw = np.zeros(N_SLOTS)
    non_app_active = np.zeros(N_SLOTS, dtype=float)
    app_groups: list[dict] = []
    total_app_evs = 0.0
    total_non_app_evs = 0.0

    for arrival_slot, arrival_weight in arrival_weights.items():
        n_arriving = arrival_weight * n_evs
        mix = _charger_mix_for_slot(int(arrival_slot))
        for charger_key, mix_fraction in mix.items():
            n_total = n_arriving * mix_fraction
            n_app = float(rng.binomial(round(n_total), app_adoption_rate))
            n_no_app = n_total - n_app
            session_s = _session_slots(charger_key)
            mw_per_ev = charger_kw[charger_key] / 1_000
            allowed = _allowed_slots(int(arrival_slot), charger_key)
            for offset in range(session_s):
                slot = (int(arrival_slot) + offset) % N_SLOTS
                non_app_load_mw[slot] += n_no_app * mw_per_ev
                non_app_active[slot] += n_no_app
            if n_app > 0:
                app_groups.append(
                    {
                        "allowed": allowed,
                        "session_s": session_s,
                        "mw_fleet": mw_per_ev * n_app,
                        "n_evs": n_app,
                    }
                )
                total_app_evs += n_app
            total_non_app_evs += n_no_app

    current_total = base_load.reindex(range(N_SLOTS)).values.copy() + non_app_load_mw
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
            score = _block_grid_cost(
                block, current_total, mw_fleet, grid_load_penalty, slot_duration_h
            )
            if score < best_score:
                best_score = score
                best_start_idx = start_idx
        for slot in allowed[best_start_idx : best_start_idx + session_s]:
            current_total[slot] += mw_fleet
            app_schedule[slot] += mw_fleet
            app_active_arr[slot] += group["n_evs"]

    non_app_mwh = float(non_app_load_mw.sum() / SLOTS_PER_HOUR)
    app_target_mwh = max(0.0, total_ev_mwh - non_app_mwh)
    app_schedule = _scale_app_load(app_schedule, app_target_mwh)

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
    )
