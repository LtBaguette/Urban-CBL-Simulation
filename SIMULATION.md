# Zone Z2 simulation

## Run (Phase A package)

```bash
pip install -r requirements.txt
python scripts/run_zone2.py
python scripts/run_app_scenarios.py
python scripts/generate_graphs.py
python scripts/run_residuals.py   # optional: hourly AR(1) + partial app adoption
python scripts/run_simulation_v5.py
python scripts/run_method_comparison.py
python -m pytest tests/ -q
```

Configuration: [`config/default.yaml`](config/default.yaml)

## Layout

| Path | Role |
|------|------|
| `sim/config.py` | Load YAML settings |
| `sim/data_loaders.py` | Datasets 5–7 loaders |
| `sim/capacity.py` | Fixed zone capacity (Dataset 5 + derating) |
| `sim/energy.py` | Energy balance + spill |
| `sim/ev_scenarios.py` | Charging schedulers |
| `sim/metrics.py` | Stress frame + KPIs |
| `sim/validate.py` | Reference savings check |
| `sim/deterministic.py` | Main simulation runners |
| `sim/residuals.py` | Hourly AR(1) demand + charger mix + partial smart charging |

Legacy entry points `zone2_simulation.py` and `zone2_app_charging_sim.py` call the same runners.

## All-method comparison (unified KPIs)

Six methods on **identical fields**, reference = `immediate_plug_in`:

| Method key | Label |
|------------|-------|
| `immediate_plug_in` | Immediate plug-in (reference) |
| `unmanaged_evening` | Baseline (evening peak) |
| `smart_flat_spread` | Smart (flat spread) |
| `smart_price_aware` | Smart (price-aware) |
| `smart_grid_aware` | Smart (grid + price) |
| `simulation_v5` | Simulation V5 (managed, ~60% app) |

Outputs: `sim_outputs/method_comparison.csv`, `Graphs/graph_method_comparison.png`. Ranked by `annual_total_savings_eur` (customer + DSO). All methods use **`residuals.app_adoption_rate`** (default 60%): blended EV load = 40% immediate plug-in + 60% method schedule; V5 already models partial adoption internally.

## Simulation V5

| Output | Path |
|--------|------|
| V5 per-run CSV | `sim_outputs/simulation_v5/run_NNN_15min.csv` |
| V5 KPI (incl. € savings) | `sim_outputs/simulation_v5/simulation_v5_kpi.csv` |
| Chart | `Graphs/simulation_v5_15min.png` |

## Residuals prototype (hourly, stochastic)

Uses the **same fleet** as `config/default.yaml` (5,000 EVs × 25 kWh) and the **same fixed zone capacity** as the APP pipeline. Differs from APP scenarios: **hourly** resolution, **AR(1)** random grid days, **60% app adoption** with Home/Fast/Super mix.

| Output | Path |
|--------|------|
| Per-run hourly CSV | `sim_outputs/residuals/run_NNN_hourly.csv` |
| Summary KPI | `sim_outputs/residuals/residuals_kpi.csv` |
| Chart | `Graphs/residuals_ar1_managed.png` |

Tune `residuals:` in [`config/default.yaml`](config/default.yaml). Do not compare € savings directly to `app_scenarios_kpi.csv` (no wholesale prices in this model).

## Phase B (APP credibility)

- **DSO vs customer savings** in `app_scenarios_kpi.csv` (`annual_dso_savings_eur`, `dso_savings_warning`)
- Rate assumptions: [`config/dso_assumptions.md`](config/dso_assumptions.md) and `dso_value` in YAML (±20% sensitivity)
- **Charts:** all PNGs under [`Graphs/`](Graphs/) (stakeholder pack + profiles; CSVs stay in `sim_outputs/`)
- **Monthly customer savings:** `sim/customer_savings.py` → `sim_outputs/app_scenarios/customer_monthly_savings.csv` and `Graphs/graph_customer_monthly_savings.png`
- Reference gap diagnosis printed by `scripts/run_zone2.py` (`summarize_reference_gap`)

## Tests

- `tests/test_zone2_invariants.py` — energy balance, capacity constant, grid ≠ price
- `tests/test_kpi_regression.py` — KPI snapshot tolerances
- `tests/test_capacity.py` — capacity metadata sanity
- `tests/test_dso_value.py` — DSO warnings and rate bounds

## Other legacy scripts

`Baseline_Simulation_Dynamic*.py` — exploratory hourly t-distribution plots only.

`Baseline_Simulation_ResidualsV4.5*.py` — thin wrapper; same as `scripts/run_residuals.py`.
