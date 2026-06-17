# CBL Urban — Zone Z2 EV Charging Simulation

Python simulation of electric-vehicle charging strategies for **Zone Z2 (Eindhoven)**. It compares customer energy savings and DSO grid value across multiple charging methods, using TenneT congestion data, zonal load profiles, and Dutch wholesale electricity prices.

## **If you did not clone the repository from github you can ignore the "Data files" part of Prerequisites and continue from the Setup section**
## Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **pip**
- **Data files** — **not included in the repository**. You must **manually add** Datasets 5, 6, and 7 under `Data_Set/Data_Set/` before running any simulation:

  | Dataset | Folder | Key files |
  |---------|--------|-----------|
  | **5** — Grid congestion & constraints | `Dataset 5 – Grid Congestion & Constraints/` | `tennetcongestie.csv`, `tennetgebieden.csv`, `congestie_pc6.csv`, … |
  | **6** — Electricity load (demand) | `Dataset 6 – Electricity Load (Demand)/` | `eindhoven_zonal_load.csv`, `eindhoven_districts.csv` |
  | **7** — Electricity prices | `Dataset 7 – Electricity Prices/` | `european_wholesale_electricity_price_data_hourly/Netherlands.csv` |

  Expected layout:

  ```
  Data_Set/Data_Set/
  ├── Dataset 5 – Grid Congestion & Constraints/
  ├── Dataset 6 – Electricity Load (Demand)/
  └── Dataset 7 – Electricity Prices/
  ```

  If these folders are missing, simulations will fail on startup with file-not-found errors.

## Setup

Clone the repository and install dependencies from the project root:

```bash
git clone <https://github.com/LtBaguette/Urban-CBL-Simulation>
cd Urban-CBL-Simulation

python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

All run commands below assume your shell is in the project root and the virtual environment is active.

## Quick start (full pipeline)

Run the **core** scripts in this order — later steps depend on CSV outputs from earlier ones:

```bash
python scripts/run_app_scenarios.py
python scripts/run_simulation_v5.py          # el diablo — 30 stochastic seeds by default
python scripts/run_method_comparison.py
python scripts/generate_graphs.py            # run last: needs method_comparison.csv
```

Optional extras:

```bash
python scripts/run_zone2.py                  # baseline interventions (10/15/20% peak shift)
python scripts/run_residuals.py              # hourly AR(1) prototype (separate model)
python scripts/run_simulation_v5_price_spike.py   # high-price window stress test
python -m pytest tests/ -q
```

On a fresh clone, the core pipeline takes a few minutes depending on your machine (`run_simulation_v5.py` is the slowest step).

`run_method_comparison.py` already writes `graph_method_comparison.png` and `all_methods_profiles.png`; `generate_graphs.py` regenerates those plus the stakeholder chart pack.

## What each script does

| Script | Purpose | Main outputs |
|--------|---------|--------------|
| `scripts/run_app_scenarios.py` | Deterministic APP charging scenarios | `sim_outputs/app_scenarios/` |
| `scripts/run_simulation_v5.py` | **el diablo** (`simulation_v5`) — 15-min stochastic optimizer (30 seeds by default) | `sim_outputs/simulation_v5/`, `Graphs/simulation_v5_15min.png` |
| `scripts/run_method_comparison.py` | Unified KPI comparison of all 8 methods; writes comparison CSV and charts | `sim_outputs/method_comparison.csv`, `Graphs/graph_method_comparison.png`, `Graphs/all_methods_profiles.png` |
| `scripts/generate_graphs.py` | Stakeholder chart pack + profile charts (requires `method_comparison.csv`; also refreshes comparison PNGs) | `Graphs/*.png` |
| `scripts/run_zone2.py` | *(Optional)* Baseline zone interventions (10/15/20% peak shift) | `sim_outputs/zone2_*.csv` |
| `scripts/run_residuals.py` | *(Optional)* Hourly AR(1) stochastic demand prototype | `sim_outputs/residuals/`, `Graphs/residuals_ar1_managed.png` |
| `scripts/run_simulation_v5_price_spike.py` | *(Optional)* Price-spike sensitivity test | `sim_outputs/simulation_v5_price_spike/`, `Graphs/simulation_v5_15min_price_spike.png` |

Price-oriented and grid-oriented baselines are computed inside `run_method_comparison.py` (not in earlier steps).

## Key outputs

| Output | Location |
|--------|----------|
| CSV results | `sim_outputs/` |
| Charts | `Graphs/` |
| Configuration | `config/default.yaml` |
| DSO rate assumptions | `config/dso_assumptions.md` |

Notable charts after a full run:

**Method comparison** (from `run_method_comparison.py`; refreshed by `generate_graphs.py`):

- `Graphs/graph_method_comparison.png` — ranked comparison of **5** charging methods (see below; full data in CSV)
- `Graphs/all_methods_profiles.png` — EV / total load / stress profiles for all 8 methods

**Stakeholder pack** (from `generate_graphs.py`):

- `Graphs/graph_all_savings.png` — stacked annual savings by party for 5 deterministic APP scenarios
- `Graphs/graph_customer_monthly_savings.png` — per-EV monthly savings for the 5 methods on the comparison chart
- `Graphs/app_charging_profiles.png` — charging load profiles for all APP scenarios
- `Graphs/simulation_initial_profile.png` — total load and stress for `unmanaged_evening` vs `smart_grid_aware` only (no EV-load panel)

**el diablo** (from `run_simulation_v5.py`):

- `Graphs/simulation_v5_15min.png` — stochastic runs (mean curve when many seeds)

## Charging methods compared

All **8** methods are evaluated on identical KPI fields and written to `sim_outputs/method_comparison.csv`. The reference scenario is **immediate plug-in** (unmanaged charging at arrival).

`Graphs/graph_method_comparison.png` shows a **subset of 5** methods (evening peak, flat spread, and price-aware smart scenarios are omitted from the chart but remain in the CSV):

| Method | Label | On comparison chart |
|--------|-------|---------------------|
| `immediate_plug_in` | Immediate plug-in (reference) | Yes |
| `unmanaged_evening` | Baseline (evening peak) | No |
| `smart_flat_spread` | Smart (flat spread) | No |
| `smart_price_aware` | Smart (price-aware) | No |
| `smart_grid_aware` | Simulation Algorithm | Yes |
| `simulation_v5` | el diablo | Yes |
| `price_oriented_baseline` | Price-oriented | Yes |
| `grid_oriented_baseline` | Grid-oriented | Yes |

By default, **60%** of the fleet follows each method's schedule and **40%** stays on immediate plug-in (`residuals.app_adoption_rate` in `config/default.yaml`).

**el diablo** (`simulation_v5`) runs **`residuals.v5_simulation_repeats`** stochastic days (default **30**); method-comparison KPIs and the V5 chart use the mean across those seeds.

**Chart label note:** `graph_all_savings.png` ends with **Simulation Algorithm** (`smart_grid_aware`, deterministic APP). `graph_method_comparison.png` includes **el diablo** (`simulation_v5`, stochastic optimizer) instead — these are different methods despite similar-sounding names.

## Configuration

Edit `config/default.yaml` to change:

- Fleet size and daily energy (`fleet.n_evs`, `fleet.kwh_per_day`)
- Zone and timing windows (`zone`, `timing`)
- App adoption rate (`residuals.app_adoption_rate`)
- Stochastic repeat count (`residuals.v5_simulation_repeats`, default 30)
- DSO monetization rates (`dso_value`) — overload minutes, high-stress minutes, peak stress ratio, peak MW reduction, and congestion integral (see [`config/dso_assumptions.md`](config/dso_assumptions.md))
- el diablo optimizer weights (`residuals.v5_objective`)

Re-run the affected scripts after changing settings.

## Project layout

```
Urban-CBL-Simulation/
├── config/           # YAML settings (default.yaml, dso_assumptions.md)
├── Data_Set/         # Input datasets (add Datasets 5–7 manually; not in git)
├── Graphs/           # Generated PNG charts
├── scripts/          # Entry-point runners
├── sim/              # Core simulation library (incl. simulation_v5.py — el diablo)
├── sim_outputs/      # Generated CSV results
├── tests/            # Pytest suite
└── SIMULATION.md     # Detailed model notes and output schemas
```

## Tests

```bash
python -m pytest tests/ -q
```

Some tests require prior simulation outputs and will skip otherwise:

| Tests | Run first |
|-------|-----------|
| `test_method_comparison.py` | `run_app_scenarios.py`, `run_simulation_v5.py` |
| `test_kpi_regression.py` (APP) | `run_app_scenarios.py` |
| `test_kpi_regression.py` (zone2) | `run_zone2.py` |
| `test_customer_monthly_savings.py` | `run_app_scenarios.py` |
| `test_residuals.py` | `run_residuals.py` |

## Further reading

See [`SIMULATION.md`](SIMULATION.md) for detailed model notes, output schemas, and method-comparison methodology.
