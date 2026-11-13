"""Reproduce everything: simulated data -> features -> trained models.

    python scripts/run_pipeline.py                 # default settings
    python scripts/run_pipeline.py --overlap 1.0   # harder PCOS/healthy overlap
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("\n>>", "python", *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--overlap", default="0.7")
    p.add_argument("--users", default="300")
    a = p.parse_args()
    run("src/data_gen/synth_lh.py", "--overlap", a.overlap, "--users", a.users, "--out", "data/raw")
    run("src/features/build_features.py")
    run("src/models/train_surge.py")
    run("src/models/train_pcos.py")
