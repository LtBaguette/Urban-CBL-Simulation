from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from sim.config import load_config
from sim.residuals import load_residuals_config, run_residuals


def test_residuals_run_produces_kpi_and_energy(tmp_path):
    sim_cfg = load_config()
    _, res_cfg = load_residuals_config(sim_cfg)
    res_cfg = replace(
        res_cfg,
        simulation_repeats=1,
        save_outputs=False,
        show_plot=False,
    )
    sim_cfg = replace(sim_cfg, output_dir=tmp_path / "sim_outputs", graphs_dir=tmp_path / "Graphs")

    kpi_df, fig_path = run_residuals(sim_cfg, res_cfg)
    assert len(kpi_df) == 1
    row = kpi_df.iloc[0]
    assert row["ev_energy_managed_mwh"] == pytest.approx(row["ev_energy_target_mwh"], rel=1e-4)
    assert fig_path is not None
    assert fig_path.exists()


def test_residuals_kpi_csv_after_full_run():
    path = Path(__file__).resolve().parents[1] / "sim_outputs" / "residuals" / "residuals_kpi.csv"
    if not path.exists():
        pytest.skip("Run: python scripts/run_residuals.py")
    df = pd.read_csv(path)
    assert len(df) >= 1
    assert "peak_reduction_mw" in df.columns
