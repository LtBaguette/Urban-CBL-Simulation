from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class FleetConfig:
    n_evs: int
    kwh_per_day: float
    charger_kw: float

    @property
    def daily_mwh(self) -> float:
        return self.n_evs * self.kwh_per_day / 1_000

    @property
    def fleet_max_mw(self) -> float:
        return self.n_evs * self.charger_kw / 1_000


@dataclass(frozen=True)
class SimConfig:
    fleet: FleetConfig
    focus_zone: str
    reference_day: str
    dt_hours: float
    dt_minutes: int
    peak_hour_start: int
    peak_hour_end: int
    offpeak_hour_end: int
    plug_in_hour: int
    ready_by_hour: int
    unmanaged_start: int
    unmanaged_end: int
    capacity_region: str
    max_congestion_derate: float
    derate_per_rating_point: float
    intervention_pcts: tuple[int, ...]
    reference_annual_savings_eur: dict[int, float]
    reference_savings_tolerance_pct: float
    energy_tolerance_mwh: float
    grid_load_penalty_eur_per_mw: float
    dso_eur_per_overload_minute_year: float
    dso_eur_per_high_stress_minute_year: float
    dso_eur_per_peak_stress_point_year: float
    dso_eur_per_congestion_stress_integral_day_year: float
    dso_sensitivity_pct: float
    congestion_index_base: float
    stress_warning_threshold: float
    plots_mode: str
    output_dir: Path
    graphs_dir: Path
    project_root: Path = PROJECT_ROOT

    @property
    def app_scenarios_dir(self) -> Path:
        return self.output_dir / "app_scenarios"

    @property
    def is_stakeholder_plots(self) -> bool:
        return self.plots_mode == "stakeholder"

    def ensure_graphs_dir(self) -> Path:
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        return self.graphs_dir


def load_config(path: Path | None = None) -> SimConfig:
    cfg_path = path or DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    fleet_raw = raw["fleet"]
    fleet = FleetConfig(
        n_evs=int(fleet_raw["n_evs"]),
        kwh_per_day=float(fleet_raw["kwh_per_day"]),
        charger_kw=float(fleet_raw["charger_kw"]),
    )
    zone = raw["zone"]
    timing = raw["timing"]
    cap = raw["capacity"]
    ref = {int(k): float(v) for k, v in raw["reference_annual_savings_eur"].items()}
    val = raw["validation"]
    out = raw.get("outputs", {})
    plots = raw.get("plots", {})
    graphs_name = str(out.get("graphs_dir", "Graphs"))

    return SimConfig(
        fleet=fleet,
        focus_zone=str(zone["focus_id"]),
        reference_day=str(zone["reference_day"]),
        dt_hours=float(timing["dt_hours"]),
        dt_minutes=int(timing["dt_minutes"]),
        peak_hour_start=int(timing["peak_hour_start"]),
        peak_hour_end=int(timing["peak_hour_end"]),
        offpeak_hour_end=int(timing["offpeak_hour_end"]),
        plug_in_hour=int(timing["plug_in_hour"]),
        ready_by_hour=int(timing["ready_by_hour"]),
        unmanaged_start=int(timing["unmanaged_start"]),
        unmanaged_end=int(timing["unmanaged_end"]),
        capacity_region=str(cap["region"]),
        max_congestion_derate=float(cap["max_congestion_derate"]),
        derate_per_rating_point=float(cap["derate_per_rating_point"]),
        intervention_pcts=tuple(int(x) for x in raw["intervention"]["shift_pcts"]),
        reference_annual_savings_eur=ref,
        reference_savings_tolerance_pct=float(val["reference_savings_tolerance_pct"]),
        energy_tolerance_mwh=float(val["energy_tolerance_mwh"]),
        grid_load_penalty_eur_per_mw=float(raw["grid_aware"]["load_penalty_eur_per_mw"]),
        dso_eur_per_overload_minute_year=float(
            raw["dso_value"]["eur_per_overload_minute_year"]
        ),
        dso_eur_per_high_stress_minute_year=float(
            raw["dso_value"]["eur_per_high_stress_minute_year"]
        ),
        dso_eur_per_peak_stress_point_year=float(
            raw["dso_value"]["eur_per_peak_stress_point_year"]
        ),
        dso_eur_per_congestion_stress_integral_day_year=float(
            raw["dso_value"].get("eur_per_congestion_stress_integral_day_year", 2000.0)
        ),
        dso_sensitivity_pct=float(raw["dso_value"]["sensitivity_pct"]),
        congestion_index_base=float(raw["congestion_index_base"]),
        stress_warning_threshold=float(raw["stress_warning_threshold"]),
        plots_mode=str(plots.get("mode", "stakeholder")),
        output_dir=PROJECT_ROOT / str(out.get("base_dir", "sim_outputs")),
        graphs_dir=PROJECT_ROOT / graphs_name,
    )
