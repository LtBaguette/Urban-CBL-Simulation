#!/usr/bin/env python3
"""Run el diablo with a short high-price window to test price-aware shifting."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.simulation_v5 import DEFAULT_PRICE_SPIKE, PriceSpike, main_price_spike
from sim.config import DEFAULT_CONFIG_PATH

if __name__ == "__main__":
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    spike_cfg = raw.get("residuals", {}).get("price_spike_test", {})
    spike = PriceSpike(
        start_hour=int(spike_cfg.get("start_hour", DEFAULT_PRICE_SPIKE.start_hour)),
        start_minute=int(spike_cfg.get("start_minute", DEFAULT_PRICE_SPIKE.start_minute)),
        duration_minutes=int(
            spike_cfg.get("duration_minutes", DEFAULT_PRICE_SPIKE.duration_minutes)
        ),
        price_eur_mwh=float(
            spike_cfg.get("price_eur_mwh", DEFAULT_PRICE_SPIKE.price_eur_mwh)
        ),
    )
    chart = main_price_spike(spike)
    print(f"\nDone. Chart: {chart}")
