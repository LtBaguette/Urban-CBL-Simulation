# Phase 3: Optimization Model & Decision Engine

**Project:** Smart charging for Eindhoven Zone Z2  
**Focus area:** When should EVs charge to avoid grid overload while meeting user needs?  
**Course task:** Urban Mobility Start-Ups for Liveable Cities — Task 3 (Slide 19)

---

## Locked-in design choices

| Step | Choice |
|------|--------|
| Step 1 — KPIs | **2, 3, 4** (customer savings, congestion relief, peak MW reduction) |
| Step 2 — Objective | **Combined** (electricity cost + grid-load penalty) |
| Step 3 — Method | **Heuristic** (greedy slot-filling + water-filling) |

---

## 1. Decision variables

For each 15-minute time slot `t` in the charging window (18:00–07:00, 52 slots):

- `x[t]` ≥ 0 — EV fleet charging power at slot `t` (MW)

Fleet parameters (from `config/default.yaml`):
- 5,000 EVs × 25 kWh/day = **125 MWh/day** total energy
- Max fleet power: **55 MW** (11 kW/EV)
- Zone capacity: **~2,291 MW** (Dataset 5 + 6)

---

## 2. Objective function (Combined)

Minimize the total effective charging cost over the plug-in window:

```
minimize  Σ_t  ( price[t] + λ × grid_load[t] ) × x[t] × dt
```

Where:
- `price[t]` — Netherlands wholesale electricity price (Dataset 7), EUR/MWh
- `grid_load[t]` — baseline Zone Z2 grid demand (Dataset 6), MW
- `λ` = **0.75** EUR/MW (`grid_aware.load_penalty_eur_per_mw`)
- `dt` = **0.25 h** (15-minute slots)

**Implementation:** `scenario_smart_grid_aware()` in `sim/ev_scenarios.py` ranks slots by `effective_cost = price + λ × grid_load` and greedily allocates energy to the lowest-cost slots first.

---

## 3. Constraints

| Constraint | Formula | Notes |
|------------|---------|-------|
| Energy balance | `Σ_t x[t] × dt = 125 MWh` | Every EV fully charged |
| Time window | `x[t] = 0` outside 18:00–07:00 | User plug-in / ready-by rules |
| Fleet power cap | `x[t] ≤ 55 MW` | Per-slot charger limit |
| Grid capacity (soft) | Skip slot if `grid_load[t] + x[t] > zone_capacity` | No hard overload allowed |

---

## 4. Heuristic algorithm

### Primary optimizer — `smart_grid_aware` (KPIs 2 & 3)

1. Compute `effective_cost[t] = price[t] + λ × grid_load[t]` for all slots in the charging window.
2. Sort slots ascending by effective cost.
3. For each slot (cheapest first):
   - Compute grid headroom: `zone_capacity − grid_load[t] − x[t]`
   - Add up to `min(remaining_energy, fleet_max × dt, headroom × dt)`
   - Update remaining energy.
4. If energy remains, spill to next-cheapest slots (`finalize_ev_load`).

This is a **greedy heuristic** — fast, handles capacity headroom, and approximates the combined objective without an LP solver.

### Secondary optimizer — water-filling (KPI 4)

`optimize_ev_load_constrained()` in `sim/residuals.py`:
- 60% of fleet uses the smart app; 40% charges immediately (unmanaged).
- For each flexible EV group, search allowed hour blocks and pick the block that **minimizes peak total load**.
- Runs on stochastic grid days (AR(1) noise from Dataset 6).

---

## 5. Results: baseline vs optimized

**Baseline:** `immediate_plug_in` (charge at full power when plugged in)  
**Optimized (KPIs 2 & 3):** `smart_grid_aware` (combined objective)  
**Optimized (KPI 4):** residuals water-filling with 60% app adoption

| KPI | Metric | Baseline | Optimized | Improvement |
|-----|--------|----------|-----------|-------------|
| **2** | Customer savings | €0/yr | **€1,700,739/yr** | Drivers save on wholesale arbitrage |
| **3** | Congestion stress integral saved | 0 | **0.218** | Charging shifted off stressed 15-min intervals |
| **4** | Peak MW reduction (mean, 5 runs) | 0 MW | **4.82 MW** | Evening peaks flattened (std 0.08 MW) |

Full table: `sim_outputs/phase3/optimization_kpi_comparison.csv`

---

## 6. Charts for your report

| Chart | KPI | Path |
|-------|-----|------|
| Stacked savings (customer + DSO) | 2 | `Graphs/graph_all_savings.png` |
| Congestion relief (KPI 3) | 3 | `Graphs/graph_all_savings.png` or `Graphs/app_charging_profiles.png` |
| Managed vs unmanaged (stochastic) | 4 | `Graphs/residuals_ar1_managed.png` |

---

## 7. Product connection (startup narrative)

| KPI | Product feature | Value proposition |
|-----|-----------------|-------------------|
| **2 — Customer savings** | **Smart Schedule mode** | App automatically charges in cheapest wholesale hours. For 5,000 EVs in Zone Z2: ~€1.7M/year collective savings (~€340/driver/year). Revenue: driver subscription or revenue-share on savings. |
| **3 — Congestion relief** | **Grid-friendly scoring** | Combined objective penalizes charging during high grid-load intervals. DSOs see measurable stress-integral reduction → B2B SaaS analytics contract. |
| **4 — Peak reduction** | **Realistic rollout model** | Even at 60% app adoption, peaks drop ~5 MW on stochastic days. Shows the grid stays stable as EV penetration grows — key pitch to municipalities and DSOs. |

**Customer segments:** EV drivers (KPI 2), DSOs / municipalities (KPIs 3 & 4).  
**Key activity:** Run combined-objective heuristic on live price + grid signals; push schedules to the app.

---

## 8. How to reproduce

```powershell
cd "c:\Users\topra\OneDrive\Documents\CBL Urban"
pip install -r requirements.txt
python scripts/run_app_scenarios.py
python scripts/run_residuals.py
python scripts/generate_graphs.py
python scripts/build_phase3_outputs.py
```
