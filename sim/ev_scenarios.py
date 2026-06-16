from __future__ import annotations

import pandas as pd

from sim.config import SimConfig
from sim.energy import assert_energy_balance, finalize_ev_load, spill_remaining_energy


def plug_in_window_mask(index: pd.DatetimeIndex, cfg: SimConfig) -> pd.Series:
    return (index.hour >= cfg.plug_in_hour) | (index.hour < cfg.ready_by_hour)


def charging_window_slots(index: pd.DatetimeIndex, cfg: SimConfig) -> list[pd.Timestamp]:
    in_window = plug_in_window_mask(index, cfg)
    evening = index[in_window & (index.hour >= cfg.plug_in_hour)]
    morning = index[in_window & (index.hour < cfg.ready_by_hour)]
    return list(evening) + list(morning)


def build_baseline_ev_load(index: pd.DatetimeIndex, cfg: SimConfig) -> pd.Series:
    ev = pd.Series(0.0, index=index)
    peak_mask = (index.hour >= cfg.peak_hour_start) & (index.hour < cfg.peak_hour_end)
    ev.loc[peak_mask] = cfg.fleet.daily_mwh / (peak_mask.sum() * cfg.dt_hours)
    return ev.rename("EV_Load_MW")


def apply_smart_charging_shift(
    grid_load: pd.Series,
    ev_load: pd.Series,
    shift_fraction: float,
    zone_capacity_mw: float,
    price: pd.Series,
    cfg: SimConfig,
) -> pd.Series:
    shifted = ev_load.copy()
    peak_mask = (shifted.index.hour >= cfg.peak_hour_start) & (
        shifted.index.hour < cfg.peak_hour_end
    )
    offpeak_mask = shifted.index.hour < cfg.offpeak_hour_end
    shift_mwh = shift_fraction * cfg.fleet.daily_mwh
    n_peak = int(peak_mask.sum())
    shifted.loc[peak_mask] -= shift_mwh / (n_peak * cfg.dt_hours)

    remaining_mwh = shift_mwh
    total_load = grid_load + shifted
    for slot in price.loc[offpeak_mask].sort_values().index:
        if remaining_mwh <= 0:
            break
        headroom_mw = zone_capacity_mw - total_load.loc[slot]
        if headroom_mw <= 0:
            continue
        add_mwh = min(
            remaining_mwh,
            headroom_mw * cfg.dt_hours,
            cfg.fleet.fleet_max_mw * cfg.dt_hours,
        )
        shifted.loc[slot] += add_mwh / cfg.dt_hours
        remaining_mwh -= add_mwh
        total_load.loc[slot] = grid_load.loc[slot] + shifted.loc[slot]

    if remaining_mwh > cfg.energy_tolerance_mwh:
        shifted, remaining_mwh = spill_remaining_energy(
            shifted,
            remaining_mwh,
            price.sort_values().index,
            cfg,
            grid_load=grid_load,
            zone_capacity_mw=zone_capacity_mw,
        )

    assert_energy_balance(shifted, cfg.fleet.daily_mwh, cfg, scenario="smart_charging_shift")
    return shifted.rename("EV_Load_MW")


def scenario_immediate_plug_in(
    index: pd.DatetimeIndex, price: pd.Series, cfg: SimConfig
) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    remaining_mwh = cfg.fleet.daily_mwh
    for slot in charging_window_slots(index, cfg):
        if remaining_mwh <= 0:
            break
        add_mwh = min(remaining_mwh, cfg.fleet.fleet_max_mw * cfg.dt_hours)
        ev.loc[slot] = add_mwh / cfg.dt_hours
        remaining_mwh -= add_mwh
    spill_order = price.loc[plug_in_window_mask(index, cfg)].sort_values().index
    return finalize_ev_load(
        ev,
        cfg.fleet.daily_mwh,
        spill_order,
        cfg,
        scenario="immediate_plug_in",
    )


def scenario_unmanaged_evening(
    index: pd.DatetimeIndex, price: pd.Series, cfg: SimConfig
) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    mask = (index.hour >= cfg.unmanaged_start) & (index.hour < cfg.unmanaged_end)
    n_slots = int(mask.sum())
    if n_slots:
        ev.loc[mask] = cfg.fleet.daily_mwh / (n_slots * cfg.dt_hours)
    return finalize_ev_load(
        ev,
        cfg.fleet.daily_mwh,
        price.sort_values().index,
        cfg,
        scenario="unmanaged_evening",
    )


def scenario_smart_flat_spread(
    index: pd.DatetimeIndex, price: pd.Series, cfg: SimConfig
) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    mask = plug_in_window_mask(index, cfg)
    n_slots = int(mask.sum())
    if n_slots:
        ev.loc[mask] = cfg.fleet.daily_mwh / (n_slots * cfg.dt_hours)
    return finalize_ev_load(
        ev,
        cfg.fleet.daily_mwh,
        price.loc[mask].sort_values().index,
        cfg,
        scenario="smart_flat_spread",
    )


def scenario_smart_price_aware(
    index: pd.DatetimeIndex, price: pd.Series, cfg: SimConfig
) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    window = plug_in_window_mask(index, cfg)
    remaining_mwh = cfg.fleet.daily_mwh
    window_slots = price.loc[window].sort_values().index
    for slot in window_slots:
        if remaining_mwh <= 0:
            break
        add_mwh = min(remaining_mwh, cfg.fleet.fleet_max_mw * cfg.dt_hours)
        ev.loc[slot] = add_mwh / cfg.dt_hours
        remaining_mwh -= add_mwh
    return finalize_ev_load(
        ev,
        cfg.fleet.daily_mwh,
        window_slots,
        cfg,
        scenario="smart_price_aware",
    )


def scenario_smart_grid_aware(
    grid_load: pd.Series,
    index: pd.DatetimeIndex,
    price: pd.Series,
    zone_capacity_mw: float,
    cfg: SimConfig,
) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    window = plug_in_window_mask(index, cfg)
    remaining_mwh = cfg.fleet.daily_mwh
    total_load = grid_load.copy()
    effective_cost = price + cfg.grid_load_penalty_eur_per_mw * grid_load
    window_slots = effective_cost.loc[window].sort_values().index
    for slot in window_slots:
        if remaining_mwh <= 0:
            break
        headroom_mw = zone_capacity_mw - total_load.loc[slot] - ev.loc[slot]
        if headroom_mw <= 0:
            continue
        add_mwh = min(
            remaining_mwh,
            cfg.fleet.fleet_max_mw * cfg.dt_hours,
            headroom_mw * cfg.dt_hours,
        )
        if add_mwh <= 0:
            continue
        ev.loc[slot] += add_mwh / cfg.dt_hours
        remaining_mwh -= add_mwh
        total_load.loc[slot] = grid_load.loc[slot] + ev.loc[slot]
    return finalize_ev_load(
        ev,
        cfg.fleet.daily_mwh,
        window_slots,
        cfg,
        grid_load=grid_load,
        zone_capacity_mw=zone_capacity_mw,
        scenario="smart_grid_aware",
    )
