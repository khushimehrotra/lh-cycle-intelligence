"""Compute the surge-model features for the LATEST reading of a live cycle.
Must match build_features.py exactly (tests/test_api.py checks this)."""
import numpy as np

FEATURES = ["cycle_day", "lh_est", "lag1", "lag2", "lag3", "diff1", "diff2",
            "base_mean7", "base_max7", "base_std7", "ratio_to_base", "max_so_far",
            "prev_reading_gap", "prev_len_mean", "prev_len_last",
            "prev_peak_day", "prev_peak_est", "day_vs_prev_peak", "day_frac_of_prev_len"]
nan = np.nan


def latest_features(days, lh, prev_lengths, prev_last_days=None, prev_last_lh=None):
    """days, lh: current cycle readings (oldest first, last = 'now').
    prev_lengths: lengths of earlier cycles (oldest first), may be empty.
    prev_last_days / prev_last_lh: readings of the most recent earlier cycle (for its peak)."""
    days, lh = np.asarray(days, float), np.asarray(lh, float)
    i = len(lh) - 1
    f = {"cycle_day": days[i], "lh_est": lh[i]}
    for k in (1, 2, 3):
        f[f"lag{k}"] = lh[i - k] if i - k >= 0 else nan
    f["diff1"] = lh[i] - f["lag1"]
    f["diff2"] = lh[i] - f["lag2"]
    win = lh[max(0, i - 7):i]                       # previous (up to) 7 readings
    f["base_mean7"] = win.mean() if len(win) >= 2 else nan
    f["base_max7"] = win.max() if len(win) >= 2 else nan
    f["base_std7"] = win.std(ddof=1) if len(win) >= 3 else nan
    f["ratio_to_base"] = lh[i] / f["base_mean7"] if f["base_mean7"] == f["base_mean7"] else nan
    f["max_so_far"] = lh[:i].max() if i > 0 else nan
    f["prev_reading_gap"] = days[i] - days[i - 1] if i > 0 else nan
    f["prev_len_mean"] = float(np.mean(prev_lengths)) if len(prev_lengths) else nan
    f["prev_len_last"] = float(prev_lengths[-1]) if len(prev_lengths) else nan
    if prev_last_days is not None and len(prev_last_lh):
        j = int(np.argmax(prev_last_lh))
        f["prev_peak_day"], f["prev_peak_est"] = float(prev_last_days[j]), float(prev_last_lh[j])
    else:
        f["prev_peak_day"] = f["prev_peak_est"] = nan
    f["day_vs_prev_peak"] = days[i] - f["prev_peak_day"]
    f["day_frac_of_prev_len"] = days[i] / f["prev_len_mean"] if len(prev_lengths) else nan
    return {k: f[k] for k in FEATURES}


def surge_events(lh, ratio=2.0, floor=12.0):
    """Clusters of consecutive readings that are >= `ratio` x the median of the previous 7 readings
    (and >= `floor` mIU/mL). Catches surges that a fixed 25 mIU/mL threshold misses in once-daily data.
    Returns a list of lists of reading indices. One cluster = one clear surge; several = erratic spikes."""
    lh = np.asarray(lh, float)
    hits = [i for i in range(3, len(lh))
            if lh[i] >= ratio * np.median(lh[max(0, i - 7):i]) and lh[i] >= floor]
    events = []
    for h in hits:
        if events and h - events[-1][-1] <= 1:
            events[-1].append(h)
        else:
            events.append([h])
    return events
