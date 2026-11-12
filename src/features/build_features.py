"""
Turn raw daily readings into one row of features per reading.

IMPORTANT: every feature uses only information the app would have at that moment
(past readings and past cycles). Using future data would "leak" the answer.

Run from the project root:
    python src/features/build_features.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SURGE_EST = 25.0  # mIU/mL, surge threshold used in the paper


def build(readings: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    df = readings.copy()
    df["cycle_idx"] = df.cycle_id.str.extract(r"_C(\d+)$").astype(int)
    df = df.sort_values(["user_id", "cycle_idx", "cycle_day"]).reset_index(drop=True)

    # ---- within-cycle history (only past readings) ----
    g = df.groupby("cycle_id")["lh_est"]
    for k in (1, 2, 3):
        df[f"lag{k}"] = g.shift(k)
    df["diff1"] = df.lh_est - df.lag1
    df["diff2"] = df.lh_est - df.lag2
    past = g.shift(1)                       # exclude today from the baseline
    grp_past = past.groupby(df.cycle_id)
    df["base_mean7"] = grp_past.transform(lambda s: s.rolling(7, min_periods=2).mean())
    df["base_max7"] = grp_past.transform(lambda s: s.rolling(7, min_periods=2).max())
    df["base_std7"] = grp_past.transform(lambda s: s.rolling(7, min_periods=3).std())
    df["ratio_to_base"] = df.lh_est / df.base_mean7
    df["max_so_far"] = g.transform(lambda s: s.shift(1).cummax())
    df["prev_reading_gap"] = df.groupby("cycle_id")["cycle_day"].diff()

    # ---- history from earlier cycles of the same user ----
    cy = cycles.copy()
    cy["cycle_idx"] = cy.cycle_id.str.extract(r"_C(\d+)$").astype(int)
    cy = cy.sort_values(["user_id", "cycle_idx"])
    peak = (df.loc[df.groupby("cycle_id")["lh_est"].idxmax(), ["cycle_id", "cycle_day", "lh_est"]]
              .rename(columns={"cycle_day": "peak_day", "lh_est": "peak_est"}))
    cy = cy.merge(peak, on="cycle_id", how="left")
    cy["prev_len_mean"] = cy.groupby("user_id")["cycle_len"].transform(
        lambda s: s.shift(1).expanding().mean())
    cy["prev_len_last"] = cy.groupby("user_id")["cycle_len"].shift(1)
    cy["prev_peak_day"] = cy.groupby("user_id")["peak_day"].shift(1)
    cy["prev_peak_est"] = cy.groupby("user_id")["peak_est"].shift(1)
    df = df.merge(cy[["cycle_id", "prev_len_mean", "prev_len_last",
                      "prev_peak_day", "prev_peak_est"]], on="cycle_id", how="left")
    df["day_vs_prev_peak"] = df.cycle_day - df.prev_peak_day
    df["day_frac_of_prev_len"] = df.cycle_day / df.prev_len_mean
    return df


FEATURES = ["cycle_day", "lh_est", "lag1", "lag2", "lag3", "diff1", "diff2",
            "base_mean7", "base_max7", "base_std7", "ratio_to_base", "max_so_far",
            "prev_reading_gap", "prev_len_mean", "prev_len_last",
            "prev_peak_day", "prev_peak_est", "day_vs_prev_peak", "day_frac_of_prev_len"]
TARGET = "surge_next_24h"

if __name__ == "__main__":
    readings = pd.read_csv(ROOT / "data/raw/readings.csv")
    cycles = pd.read_csv(ROOT / "data/raw/cycles.csv")
    out = build(readings, cycles)
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "data/processed/surge_features.csv", index=False)
    print(out.shape, f"| positive rate of {TARGET}: {out[TARGET].mean():.1%}")
