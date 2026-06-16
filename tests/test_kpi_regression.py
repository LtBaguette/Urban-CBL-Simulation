from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sim.config import load_config
from sim.validate import summarize_reference_gap, validate_reference_savings

BASE = Path(__file__).resolve().parents[1]
KPI_APP = BASE / "sim_outputs" / "app_scenarios" / "app_scenarios_kpi.csv"
KPI_ZONE2 = BASE / "sim_outputs" / "zone2_kpi_comparison.csv"

# Golden values from Phase A run (loose relative tolerances).
APP_GOLDEN = {
    "immediate_plug_in": {
        "annual_customer_savings_eur": 0.0,
        "zone_capacity_mw": 2291.27,
        "dso_savings_warning": False,
    },
    "unmanaged_evening": {"dso_savings_warning": True},
    "smart_price_aware": {
        "annual_customer_savings_eur": 1.727e6,
        "peak_stress_ratio": 1.176,
    },
    "smart_grid_aware": {
        "annual_customer_savings_eur": 1.701e6,
        "peak_stress_ratio": 1.176,
    },
}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def app_kpi():
    if not KPI_APP.exists():
        pytest.skip("Run scripts/run_app_scenarios.py first")
    return pd.read_csv(KPI_APP)


def test_app_kpi_capacity_constant(app_kpi):
    caps = app_kpi["zone_capacity_mw"].unique()
    assert len(caps) == 1


def test_app_kpi_golden_columns(app_kpi):
    indexed = app_kpi.set_index("scenario")
    for scenario, expected in APP_GOLDEN.items():
        row = indexed.loc[scenario]
        for col, val in expected.items():
            if col == "dso_savings_warning":
                assert bool(row[col]) == val
            elif col in ("annual_savings_eur", "annual_customer_savings_eur"):
                assert row[col] == pytest.approx(val, rel=0.02, abs=5000)
            elif col == "zone_capacity_mw":
                assert row[col] == pytest.approx(val, rel=0.001)
            else:
                assert row[col] == pytest.approx(val, rel=0.01)


def test_app_kpi_has_dso_columns(app_kpi):
    for col in (
        "annual_dso_savings_eur",
        "annual_total_savings_eur",
        "dso_savings_warning",
    ):
        assert col in app_kpi.columns


def test_app_smart_grid_ne_price_cost(app_kpi):
    indexed = app_kpi.set_index("scenario")
    assert (
        indexed.loc["smart_grid_aware", "annual_ev_energy_cost_eur"]
        != indexed.loc["smart_price_aware", "annual_ev_energy_cost_eur"]
    )


def test_reference_savings_within_tolerance(cfg):
    if not KPI_ZONE2.exists():
        pytest.skip("Run scripts/run_zone2.py first")
    kpi = pd.read_csv(KPI_ZONE2)
    validate_reference_savings(kpi, cfg, fail=True)


def test_reference_gap_is_systematic(cfg):
    if not KPI_ZONE2.exists():
        pytest.skip("Run scripts/run_zone2.py first")
    kpi = pd.read_csv(KPI_ZONE2)
    lines = summarize_reference_gap(kpi, cfg)
    assert any("-5.9" in line or "-5.8" in line for line in lines)
    assert any("systematic" in line.lower() for line in lines)
