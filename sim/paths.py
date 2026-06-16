from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data_Set" / "Data_Set"
DS5_DIR = next(DATA_DIR.glob("Dataset 5*"))
DS6_DIR = next(DATA_DIR.glob("Dataset 6*"))
DS7_DIR = next(DATA_DIR.glob("Dataset 7*"))

ZONAL_LOAD_FILE = DS6_DIR / "eindhoven_zonal_load.csv"
DISTRICTS_FILE = DS6_DIR / "eindhoven_districts.csv"
TENNET_GEBIEDEN_FILE = DS5_DIR / "tennetgebieden.csv"
TENNET_CONGESTION_FILE = DS5_DIR / "tennetcongestie.csv"
CONGESTION_PC6_FILE = DS5_DIR / "congestie_pc6.csv"
PRICE_FILE = (
    DS7_DIR
    / "european_wholesale_electricity_price_data_hourly"
    / "Netherlands.csv"
)

EINDHOVEN_PC_PREFIXES = ("561", "562", "563", "564")
