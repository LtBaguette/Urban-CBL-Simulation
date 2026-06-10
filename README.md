# CBL Urban — Zone Z2 EV Charging Simulation

Python simulation of electric-vehicle charging strategies for **Zone Z2 (Eindhoven)**. It compares customer energy savings and DSO grid value across multiple charging methods, using TenneT congestion data, zonal load profiles, and Dutch wholesale electricity prices.

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
git clone <repository-url>
cd "CBL Urban"

python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

All run commands below assume your shell is in the project root and the virtual environment is active.

## Quick start (full pipeline)

Run scripts **in this order** — later steps depend on CSV outputs from earlier ones:

```bash
python scripts/run_zone2.py
python scripts/run_app_scenarios.py
python scripts/generate_graphs.py
python scripts/run_residuals.py
python scripts/run_simulation_v5.py
python scripts/run_method_comparison.py
```

Optional extras:

```bash
python scripts/run_simulation_v5_price_spike.py   # high-price window stress test
python -m pytest tests/ -q                          # run test suite
```

On a fresh clone, the full pipeline takes a few minutes depending on your machine.

## What each script does

| Script | Purpose | Main outputs |
|--------|---------|--------------|
| `scripts/run_zone2.py` | Baseline zone interventions (10/15/20% peak shift) | `sim_outputs/zone2_*.csv` |
| `scripts/run_app_scenarios.py` | Deterministic APP charging scenarios | `sim_outputs/app_scenarios/` |
| `scripts/generate_graphs.py` | Stakeholder charts from existing CSVs | `Graphs/*.png` |
| `scripts/run_residuals.py` | Hourly AR(1) stochastic demand prototype | `sim_outputs/residuals/` |
| `scripts/run_simulation_v5.py` | **el diablo** — 15-min stochastic optimizer | `sim_outputs/simulation_v5/` |
| `scripts/run_simulation_v5_price_spike.py` | Price-spike sensitivity test | `sim_outputs/simulation_v5_price_spike/` |
| `scripts/run_method_comparison.py` | Unified KPI comparison of all 8 methods | `sim_outputs/method_comparison.csv`, `Graphs/graph_method_comparison.png` |

## Key outputs

| Output | Location |
|--------|----------|
| CSV results | `sim_outputs/` |
| Charts | `Graphs/` |
| Configuration | `config/default.yaml` |
| DSO rate assumptions | `config/dso_assumptions.md` |

Notable charts after a full run:

- `Graphs/graph_method_comparison.png` — ranked comparison of all charging methods
- `Graphs/all_methods_profiles.png` — EV / total load / stress profiles
- `Graphs/simulation_v5_15min.png` — el diablo stochastic runs

## Charging methods compared

All methods are evaluated on identical KPI fields. The reference scenario is **immediate plug-in** (unmanaged charging at arrival).

| Method | Label |
|--------|-------|
| `immediate_plug_in` | Immediate plug-in (reference) |
| `unmanaged_evening` | Baseline (evening peak) |
| `smart_flat_spread` | Smart (flat spread) |
| `smart_price_aware` | Smart (price-aware) |
| `smart_grid_aware` | Smart (grid + price) |
| `simulation_v5` | el diablo |
| `price_oriented_baseline` | Price-oriented (baseline) |
| `grid_oriented_baseline` | Grid-oriented (baseline) |

By default, **60%** of the fleet follows each method's schedule and **40%** stays on immediate plug-in (`residuals.app_adoption_rate` in `config/default.yaml`).

## Configuration

Edit `config/default.yaml` to change:

- Fleet size and daily energy (`fleet.n_evs`, `fleet.kwh_per_day`)
- Zone and timing windows (`zone`, `timing`)
- App adoption rate (`residuals.app_adoption_rate`)
- DSO monetization rates (`dso_value`)
- el diablo optimizer weights (`residuals.v5_objective`)

Re-run the affected scripts after changing settings.

## Project layout

```
CBL Urban/
├── config/           # YAML settings and DSO assumptions
├── Data_Set/         # Input datasets (add Datasets 5–7 manually; not in git)
├── Graphs/           # Generated PNG charts
├── scripts/          # Entry-point runners
├── sim/              # Core simulation library (incl. simulation_v5.py — el diablo)
├── sim_outputs/      # Generated CSV results
└── tests/            # Pytest suite
```

## Tests

```bash
python -m pytest tests/ -q
```

Some tests require prior simulation outputs. Run `run_app_scenarios.py` and `run_simulation_v5.py` first if tests are skipped.

## Further reading

See [`SIMULATION.md`](SIMULATION.md) for detailed model notes, output schemas, and method-comparison methodology.
