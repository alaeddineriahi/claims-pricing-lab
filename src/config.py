"""Paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"

URLS = {
    "freMTPL2freq.parquet": "https://data.openml.org/datasets/0004/41214/dataset_41214.pq",
    "freMTPL2sev.parquet": "https://data.openml.org/datasets/0004/41215/dataset_41215.pq",
}

SEED = 42
CLAIMNB_CAP = 4          # counts above 4 treated as data errors (Noll, Salzmann & Wüthrich 2018)
EXPOSURE_CAP = 1.0       # exposures above one year are capped
LARGE_CLAIM_QUANTILE = 0.995  # individual claims above this are capped; the excess becomes a loading
SPLIT = {"train": 0.7, "valid": 0.1, "test": 0.2}

for d in (RAW, PROCESSED, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
