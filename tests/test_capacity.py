from sim.capacity import resolve_zone_capacity_mw
from sim.config import load_config
from sim.data_loaders import load_zone2_15min_demand


def test_capacity_below_underrated_grid_peak():
    cfg = load_config()
    grid = load_zone2_15min_demand(cfg)
    cap, meta = resolve_zone_capacity_mw(grid, cfg)
    assert cap < grid.max()
    assert meta["cap_noord_brabant_mw"] > 2000
    assert meta["capacity_source"] == "dataset6_grid_peak_with_dataset5_derating"
