#!/usr/bin/env python3
"""
Synthetic LH / biosensor data generator (Step 1 of the project).

Simulates what the Pd-Ti3C2Tx CSPE biosensor would send: one charge-transfer
resistance (Rct, kOhm) reading per day, plus ground-truth labels for ML.

Calibration comes from Table 1 of Shrivastav et al., Electrochimica Acta 560 (2026) 148659.
Everything else (cycle physiology, noise levels, PCOS behaviour) is an ASSUMPTION
you should tune; the paper does not give these.

Usage:
    python synth_lh.py --users 300 --cycles 6 --noise 0.5 --out data
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- calibration
# Paper Table 1 (lab Autolab system). Rct DEcreases as LH increases.
CONC = np.array([1, 5, 10, 20, 30, 50, 75, 100, 150, 200], float)       # mIU/mL
RCT = np.array([77.20, 76.87, 75.17, 72.43, 70.10, 66.02,
                64.71, 62.74, 61.27, 59.68])                              # kOhm
# Portable prototype reads ~1.85-2.4 kOhm lower than the lab system (Fig 6e vs 6f).
PORTABLE_OFFSET_KOHM = -2.1
LOQ = 8.0       # mIU/mL, from the abstract: readings below this are unreliable
SURGE_THRESHOLD = 25.0  # mIU/mL, surge threshold used in the paper


def lh_to_rct(lh):
    """Piecewise-linear interpolation of Table 1 (the curve is NOT one straight line)."""
    return np.interp(np.clip(lh, CONC[0], CONC[-1]), CONC, RCT)


def rct_to_lh(rct):
    """Inverse calibration: what the app does with a raw reading. Saturates at 1 and 200."""
    return np.interp(rct, RCT[::-1], CONC[::-1])


# ---------------------------------------------------------------- physiology
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_user(rng, uid, pcos, overlap=0.7):
    """overlap in [0,1]: 0 = PCOS and healthy users are cleanly separated (easy),
    1 = heavy mixing (many PCOS users look normal, some healthy users look irregular)."""
    base, mean_len, len_sd = rng.uniform(3.5, 8), rng.normal(28, 2), rng.uniform(1, 2.5)
    if pcos:
        if rng.random() < 1 - 0.5 * overlap:      # classic elevated-LH PCOS phenotype
            base = rng.uniform(9, 20)
        else:                                     # PCOS with near-normal LH
            base = rng.uniform(4, 11)
        if rng.random() < 1 - 0.5 * overlap:      # oligo-ovulation: long, variable cycles
            mean_len, len_sd = rng.uniform(33, 50), rng.uniform(3, 7)
        else:                                     # PCOS with fairly normal cycle length
            mean_len, len_sd = rng.normal(30, 3), rng.uniform(2, 4)
    else:
        if rng.random() < 0.30 * overlap:         # healthy but higher LH
            base = rng.uniform(7, 13)
        if rng.random() < 0.25 * overlap:         # healthy but irregular cycles
            mean_len, len_sd = rng.uniform(31, 38), rng.uniform(3, 6)
    return dict(
        user_id=uid, pcos_user=int(pcos), base=base, mean_len=mean_len, len_sd=len_sd,
        luteal=rng.normal(13.5, 1.0),
        device_offset=rng.normal(PORTABLE_OFFSET_KOHM, 0.4),            # per-device bias
        meas_hour=rng.uniform(6, 10),                                    # morning reading
    )


def pick_cycle_kind(rng, pcos, overlap=0.7):
    r = rng.random()
    if pcos:
        flat, erratic = 0.55 - 0.25 * overlap, 0.25 - 0.10 * overlap   # rest: delayed ovulation
        return "pcos_flat" if r < flat else "pcos_erratic" if r < flat + erratic else "pcos_delayed"
    return "anovulatory_healthy" if r < 0.07 + 0.06 * overlap else "normal"


def simulate_cycle(rng, u, kind):
    """Return hourly true LH trace plus ground-truth event times (hours from cycle start)."""
    cyc_len = int(np.clip(round(rng.normal(u["mean_len"], u["len_sd"])), 21, 70))
    t = np.arange(cyc_len * 24, dtype=float)
    base = u["base"] * (0.9 + 0.2 * rng.random())
    day_noise = np.repeat(rng.normal(0, 1, cyc_len), 24)
    bio_noise = np.exp(0.08 * day_noise + 0.06 * rng.normal(0, 1, t.size))
    t_on = ovulation = None

    if kind in ("normal", "pcos_delayed"):
        luteal = u["luteal"]
        ovulation = (cyc_len - luteal) * 24 + 12 + rng.normal(0, 4)
        t_on = ovulation - rng.uniform(26, 36)             # surge onset ~24-36 h pre-ovulation
        amp = rng.uniform(12, 60) if kind == "normal" else rng.uniform(10, 35)
        rise = sigmoid((t - (t_on + 6)) / 3.0)
        fall = np.exp(-np.clip(t - (t_on + 16), 0, None) / 12.0)
        surge = amp * rise * fall
        ramp = 1 + 0.5 * np.clip((t - (t_on - 72)) / 72, 0, 1)         # late-follicular rise
        post = 1 - 0.3 * sigmoid((t - (t_on + 48)) / 6)                # lower in luteal phase
        lh = base * ramp * post * bio_noise + surge
    elif kind in ("pcos_flat", "anovulatory_healthy"):
        lh = base * bio_noise * np.exp(0.10 * rng.normal(0, 1, t.size))
    elif kind == "pcos_erratic":
        lh = base * bio_noise * np.exp(0.10 * rng.normal(0, 1, t.size))
        for _ in range(rng.integers(2, 6)):                # mini-spikes, never a sustained surge
            c, w = rng.uniform(48, t.size - 48), rng.uniform(5, 10)
            lh = lh + rng.uniform(0.8, 1.8) * base * np.exp(-0.5 * ((t - c) / w) ** 2)
    else:
        raise ValueError(kind)

    return dict(cyc_len=cyc_len, lh=np.clip(lh, 0.5, 200), t_on=t_on, ovulation=ovulation)


# ---------------------------------------------------------------- measurement
def measure(rng, u, cyc, noise_kohm, dropout=0.05):
    rows = []
    for d in range(cyc["cyc_len"]):
        if rng.random() < dropout:                          # user forgot / bad strip
            continue
        hr = int(np.clip(d * 24 + u["meas_hour"] + rng.normal(0, 1), 0, cyc["cyc_len"] * 24 - 1))
        lh_true = cyc["lh"][hr]
        rct_true = float(lh_to_rct(lh_true))
        rct_meas = rct_true + u["device_offset"] + rng.normal(0, noise_kohm)
        lh_est = float(rct_to_lh(rct_meas - PORTABLE_OFFSET_KOHM))   # app only knows NOMINAL offset
        h2s = (cyc["t_on"] - hr) if cyc["t_on"] is not None else np.nan
        rows.append(dict(
            cycle_day=d + 1, sample_hour=hr, lh_true=round(lh_true, 2),
            rct_true_kohm=round(rct_true, 3), raw_impedance_kohm=round(rct_meas, 3),
            lh_est=round(lh_est, 2), below_loq=int(lh_est < LOQ),
            hours_to_surge_onset=round(h2s, 1) if h2s == h2s else np.nan,
            surge_next_24h=int(0 < h2s <= 24) if h2s == h2s else 0,
            surge_next_48h=int(0 < h2s <= 48) if h2s == h2s else 0,
        ))
    return rows


def generate(n_users, n_cycles, noise, seed, pcos_frac=0.25, overlap=0.7):
    rng = np.random.default_rng(seed)
    readings, cycles = [], []
    for i in range(n_users):
        u = make_user(rng, f"U{i:04d}", rng.random() < pcos_frac, overlap)
        for c in range(n_cycles):
            kind = pick_cycle_kind(rng, u["pcos_user"], overlap)
            cyc = simulate_cycle(rng, u, kind)
            cid = f"{u['user_id']}_C{c}"
            for r in measure(rng, u, cyc, noise):
                readings.append(dict(user_id=u["user_id"], cycle_id=cid, **r))
            cycles.append(dict(
                user_id=u["user_id"], cycle_id=cid, pcos_user=u["pcos_user"], kind=kind,
                cycle_len=cyc["cyc_len"], has_surge=int(cyc["t_on"] is not None),
                surge_onset_hr=cyc["t_on"], ovulation_hr=cyc["ovulation"],
                peak_lh=float(cyc["lh"].max())))
    return pd.DataFrame(readings), pd.DataFrame(cycles)


def plot_examples(readings, cycles, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = ["normal", "pcos_flat", "pcos_erratic"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 6), sharex="col")
    for j, k in enumerate(kinds):
        cid = cycles.loc[cycles.kind == k, "cycle_id"].iloc[0]
        d = readings[readings.cycle_id == cid]
        ax[0, j].plot(d.cycle_day, d.lh_true, "-o", ms=3, label="true LH")
        ax[0, j].plot(d.cycle_day, d.lh_est, "x--", ms=5, label="LH from sensor")
        ax[0, j].axhline(SURGE_THRESHOLD, color="grey", ls=":", lw=1)
        ax[0, j].axhline(LOQ, color="red", ls=":", lw=1)
        ax[0, j].set_title(k); ax[0, j].set_ylabel("LH (mIU/mL)")
        ax[1, j].plot(d.cycle_day, d.raw_impedance_kohm, "-o", ms=3, color="tab:green")
        ax[1, j].set_ylabel("Rct (kOhm)"); ax[1, j].set_xlabel("cycle day")
    ax[0, 0].legend()
    fig.suptitle("Synthetic cycles (grey = 25 mIU/mL surge threshold, red = 8 mIU/mL LoQ)")
    fig.tight_layout(); fig.savefig(path, dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--noise", type=float, default=0.5, help="Rct noise SD in kOhm (ASSUMED)")
    p.add_argument("--overlap", type=float, default=0.7,
                   help="0 = easy (PCOS clearly different), 1 = hard (heavy mixing)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    readings, cycles = generate(a.users, a.cycles, a.noise, a.seed, overlap=a.overlap)
    readings.to_csv(out / "readings.csv", index=False)
    cycles.to_csv(out / "cycles.csv", index=False)
    plot_examples(readings, cycles, out / "example_cycles.png")
    print(f"{len(readings):,} readings, {len(cycles):,} cycles")
    print(cycles.kind.value_counts().to_string())
    print(f"readings below LoQ (8 mIU/mL): {readings.below_loq.mean():.0%}")
    print(f"RMSE of LH from sensor vs truth: "
          f"{np.sqrt(((readings.lh_est - readings.lh_true) ** 2).mean()):.2f} mIU/mL")
