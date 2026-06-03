from pathlib import Path

import pytest

from sim.config import load_config
from sim.customer_savings import (
    REFERENCE_SCENARIO,
    compute_monthly_customer_savings,
    save_monthly_customer_savings,
)

BASE = Path(__file__).resolve().parents[1]
KPI_APP = BASE / "sim_outputs" / "app_scenarios" / "app_scenarios_kpi.csv"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_monthly_matches_annual_over_twelve(cfg):
    if not KPI_APP.exists():
        pytest.skip("Run scripts/run_app_scenarios.py first")
    df = compute_monthly_customer_savings(cfg)
    row = df.loc[df["scenario"] == "smart_price_aware"].iloc[0]
    assert row["monthly_fleet_savings_eur"] == pytest.approx(
        row["annual_customer_savings_eur"] / 12, rel=1e-6
    )
    assert row["monthly_savings_per_ev_eur"] == pytest.approx(
        row["monthly_fleet_savings_eur"] / cfg.fleet.n_evs, rel=1e-6
    )


def test_reference_has_zero_monthly_savings(cfg):
    if not KPI_APP.exists():
        pytest.skip("Run scripts/run_app_scenarios.py first")
    df = compute_monthly_customer_savings(cfg)
    ref = df.loc[df["scenario"] == REFERENCE_SCENARIO].iloc[0]
    assert ref["monthly_savings_per_ev_eur"] == 0.0


def test_smart_scenarios_positive_per_ev(cfg):
    if not KPI_APP.exists():
        pytest.skip("Run scripts/run_app_scenarios.py first")
    df = compute_monthly_customer_savings(cfg)
    smart = df[df["scenario"].str.startswith("smart_")]
    assert (smart["monthly_savings_per_ev_eur"] > 0).all()


def test_save_monthly_csv(cfg, tmp_path):
    if not KPI_APP.exists():
        pytest.skip("Run scripts/run_app_scenarios.py first")
    out = tmp_path / "customer_monthly_savings.csv"
    df = save_monthly_customer_savings(cfg, path=out)
    assert out.exists()
    assert "monthly_savings_per_ev_eur" in df.columns
