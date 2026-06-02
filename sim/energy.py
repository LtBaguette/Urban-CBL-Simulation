from __future__ import annotations

import pandas as pd

from sim.config import SimConfig


def delivered_energy_mwh(ev_load: pd.Series, cfg: SimConfig) -> float:
    return float(ev_load.sum() * cfg.dt_hours)


def assert_energy_balance(
    ev_load: pd.Series,
    target_mwh: float,
    cfg: SimConfig,
    *,
    scenario: str = "",
) -> None:
    gap = delivered_energy_mwh(ev_load, cfg) - target_mwh
    if abs(gap) > cfg.energy_tolerance_mwh:
        label = f" ({scenario})" if scenario else ""
        raise ValueError(
            f"Energy not conserved{label}: delivered "
            f"{delivered_energy_mwh(ev_load, cfg):.6f} MWh, target {target_mwh:.6f} MWh"
        )


def spill_remaining_energy(
    ev_load: pd.Series,
    remaining_mwh: float,
    slot_order: pd.Index,
    cfg: SimConfig,
    *,
    grid_load: pd.Series | None = None,
    zone_capacity_mw: float | None = None,
) -> tuple[pd.Series, float]:
    ev = ev_load.copy()
    fleet_max_mw = cfg.fleet.fleet_max_mw
    total_load = grid_load + ev if grid_load is not None else None

    for slot in slot_order:
        if remaining_mwh <= cfg.energy_tolerance_mwh:
            break
        headroom_mw = float("inf")
        if grid_load is not None and zone_capacity_mw is not None and total_load is not None:
            headroom_mw = zone_capacity_mw - float(total_load.loc[slot])
        if headroom_mw <= 0:
            continue
        add_mwh = min(remaining_mwh, fleet_max_mw * cfg.dt_hours, headroom_mw * cfg.dt_hours)
        if add_mwh <= 0:
            continue
        ev.loc[slot] += add_mwh / cfg.dt_hours
        remaining_mwh -= add_mwh
        if total_load is not None:
            total_load.loc[slot] = float(grid_load.loc[slot]) + float(ev.loc[slot])

    return ev, remaining_mwh


def finalize_ev_load(
    ev_load: pd.Series,
    target_mwh: float,
    slot_order: pd.Index,
    cfg: SimConfig,
    *,
    grid_load: pd.Series | None = None,
    zone_capacity_mw: float | None = None,
    scenario: str = "",
) -> pd.Series:
    ev = ev_load.copy()
    remaining = target_mwh - delivered_energy_mwh(ev, cfg)
    if abs(remaining) <= cfg.energy_tolerance_mwh:
        assert_energy_balance(ev, target_mwh, cfg, scenario=scenario)
        return ev

    ev, remaining = spill_remaining_energy(
        ev,
        remaining,
        slot_order,
        cfg,
        grid_load=grid_load,
        zone_capacity_mw=zone_capacity_mw,
    )

    if remaining > cfg.energy_tolerance_mwh:
        for slot in slot_order:
            if remaining <= cfg.energy_tolerance_mwh:
                break
            add_mwh = min(remaining, cfg.fleet.fleet_max_mw * cfg.dt_hours)
            ev.loc[slot] += add_mwh / cfg.dt_hours
            remaining -= add_mwh

    if remaining > cfg.energy_tolerance_mwh:
        raise ValueError(
            f"Could not allocate {remaining:.4f} MWh for {scenario or 'scenario'}"
        )

    assert_energy_balance(ev, target_mwh, cfg, scenario=scenario)
    return ev


def assert_load_invariants(ev_load: pd.Series, cfg: SimConfig, *, scenario: str = "") -> None:
    if (ev_load < -1e-12).any():
        raise ValueError(f"Negative EV load in {scenario}")
    if (ev_load > cfg.fleet.fleet_max_mw + 1e-6).any():
        raise ValueError(f"EV load exceeds fleet max in {scenario}")
