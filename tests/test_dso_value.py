import pytest

from sim.config import load_config
from sim.dso_value import annual_dso_savings_eur, dso_rate_bounds


def test_dso_negative_when_peak_stress_worse():
    cfg = load_config()
    # Same overload/stress minutes but higher peak stress → negative DSO value
    dso = annual_dso_savings_eur(
        overload_minutes=870,
        high_stress_minutes=975,
        peak_stress_ratio=1.19,
        reference_overload_minutes=870,
        reference_high_stress_minutes=975,
        reference_peak_stress_ratio=1.176,
        cfg=cfg,
    )
    assert dso < 0
    # 0.014 worse → 1.4 centiles × rate (annual, not ×365)
    assert dso == pytest.approx(-1.4 * cfg.dso_eur_per_peak_stress_point_year, rel=0.01)


def test_dso_positive_from_congestion_integral():
    cfg = load_config()
    dso = annual_dso_savings_eur(
        overload_minutes=870,
        high_stress_minutes=975,
        peak_stress_ratio=1.176,
        reference_overload_minutes=870,
        reference_high_stress_minutes=975,
        reference_peak_stress_ratio=1.176,
        cfg=cfg,
        congestion_stress_integral_saved=0.218,
    )
    assert dso == pytest.approx(0.218 * cfg.dso_eur_per_congestion_stress_integral_day_year * 365)


def test_dso_sensitivity_bounds():
    cfg = load_config()
    bounds = dso_rate_bounds(cfg)
    low, high = bounds["overload"]
    assert low == cfg.dso_eur_per_overload_minute_year * 0.8
    assert high == cfg.dso_eur_per_overload_minute_year * 1.2
