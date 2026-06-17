from pathlib import Path

import pandas as pd
import pytest

from sim.config import load_config
from sim.method_comparison import (
    METHOD_ORDER,
    REFERENCE_METHOD,
    UNIFIED_COLUMNS,
    build_method_comparison,
)

BASE = Path(__file__).resolve().parents[1]
APP_KPI = BASE / "sim_outputs" / "app_scenarios" / "app_scenarios_kpi.csv"
V5_KPI = BASE / "sim_outputs" / "simulation_v5" / "simulation_v5_kpi.csv"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_all_methods_present(cfg):
    if not APP_KPI.exists() or not V5_KPI.exists():
        pytest.skip("Run run_app_scenarios.py and run_simulation_v5.py (el diablo) first")
    df = build_method_comparison(cfg)
    assert len(df) == len(METHOD_ORDER)
    assert list(df["method_key"]) == [k for k, _ in METHOD_ORDER]
    for rate in df["app_adoption_rate"]:
        assert rate == pytest.approx(0.6, abs=0.01)


def test_unified_columns(cfg):
    if not APP_KPI.exists() or not V5_KPI.exists():
        pytest.skip("Run run_app_scenarios.py and run_simulation_v5.py (el diablo) first")
    df = build_method_comparison(cfg)
    assert list(df.columns) == UNIFIED_COLUMNS


def test_reference_zeros(cfg):
    if not APP_KPI.exists() or not V5_KPI.exists():
        pytest.skip("Run run_app_scenarios.py and run_simulation_v5.py (el diablo) first")
    df = build_method_comparison(cfg).set_index("method_key")
    ref = df.loc[REFERENCE_METHOD]
    assert ref["annual_customer_savings_eur"] == pytest.approx(0.0, abs=1.0)
    assert ref["annual_dso_savings_eur"] == pytest.approx(0.0, abs=1.0)
    assert ref["annual_total_savings_eur"] == pytest.approx(0.0, abs=1.0)
    assert ref["rank_by_total_savings"] >= len(METHOD_ORDER) - 1


def test_smart_price_beats_baseline(cfg):
    if not APP_KPI.exists() or not V5_KPI.exists():
        pytest.skip("Run run_app_scenarios.py and run_simulation_v5.py (el diablo) first")
    df = build_method_comparison(cfg).set_index("method_key")
    assert (
        df.loc["smart_price_aware", "annual_total_savings_eur"]
        > df.loc["unmanaged_evening", "annual_total_savings_eur"]
    )
