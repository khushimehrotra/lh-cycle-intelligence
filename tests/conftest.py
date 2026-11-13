"""Lets `pytest` work on a fresh clone (and in CI): if the simulated data or trained models
are missing, build them first with scripts/run_pipeline.py."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDED = [ROOT / "data/raw/readings.csv", ROOT / "data/raw/cycles.csv",
          ROOT / "data/processed/surge_features.csv",
          ROOT / "saved_models/surge_xgb.pkl", ROOT / "saved_models/pcos_rf.pkl"]

if not all(p.exists() for p in NEEDED):
    print("\nData/models missing: running the pipeline first...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_pipeline.py")], cwd=ROOT, check=True)
