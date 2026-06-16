"""Price spike experiment helpers."""

from sim.simulation_v5 import (
    DEFAULT_PRICE_SPIKE,
    N_SLOTS,
    apply_price_spike,
    load_slot_price_profile,
)
from sim.config import load_config


def test_spike_overwrites_one_slot():
    cfg = load_config()
    base = load_slot_price_profile(cfg)
    spiked = apply_price_spike(base, DEFAULT_PRICE_SPIKE)
    spike_slots = DEFAULT_PRICE_SPIKE.slot_indices()
    assert len(spike_slots) == 1
    slot = spike_slots[0]
    assert slot == 12  # 03:00
    assert spiked[slot] == DEFAULT_PRICE_SPIKE.price_eur_mwh
    unchanged = [s for s in range(N_SLOTS) if s != slot]
    assert (spiked[unchanged] == base[unchanged]).all()
