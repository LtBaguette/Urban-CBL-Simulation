# DSO grid-value assumptions (Zone Z2 APP scenarios)

Monetized benefits for distribution system operators (DSOs) are **relative to immediate plug-in**
(the APP reference behaviour).

- **Overload / high-stress minutes:** daily count × rate × 365 (same stressed day repeated).
- **Peak stress ratio:** `(Δ stress) / 0.01 × rate` once per year (daily peak index, **not** ×365).
- **Congestion integral:** only when the scenario does **not** worsen daily peak stress vs reference.

## Rates (`config/default.yaml` → `dso_value`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `eur_per_overload_minute_year` | 48 | EUR per year for each **daily** overload minute avoided (stress ≥ 1.0) |
| `eur_per_high_stress_minute_year` | 20 | EUR per year for each **daily** minute avoided above stress threshold (0.95) |
| `eur_per_peak_stress_point_year` | 20,000 | EUR per year for each **0.01** lower daily peak stress ratio vs reference |
| `eur_per_congestion_stress_integral_day_year` | 2,000 | EUR per year for each unit of daily stress relief on high-stress slots (see below) |

**Congestion stress integral:** on intervals where reference stress ≥ threshold (0.95),
sum `max(0, stress_reference − stress_scenario)` per day. Smart charging can lower stress on
congested slots without changing the daily peak; this term captures that relief.

## Sensitivity

For stakeholder ranges, vary all three rates by **`sensitivity_pct`** (default **±20%**):

- Low case: `rate × (1 − 0.20)`
- High case: `rate × (1 + 0.20)`

These are **order-of-magnitude planning values**, not tariff-backed DSO revenues. Calibrate
against DSO deferral studies before external reporting.

**Evening peak example:** stress ratio 1.190 vs reference 1.176 → 0.014 worse →  
`−(0.014/0.01) × 20,000 ≈ −€28k/year` peak penalty (not €1M+). Tune
`eur_per_peak_stress_point_year` to match your deferral €/MW or €/0.01 assumptions.

## Customer vs DSO split

- **Customers:** `annual_customer_savings_eur` = lower EV energy bills vs immediate plug-in.
- **DSOs:** `annual_dso_savings_eur` = monetized grid-stress improvement vs the same reference.
- **`dso_savings_warning`:** `true` when DSO savings are negative (scenario worsens the grid).

Negative DSO savings (e.g. evening peak habit) are valid: drivers may save on bills while
the grid sees higher peaks or stress.
