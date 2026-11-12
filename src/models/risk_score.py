"""Simple, transparent 0-100 risk score for preventive follow-up.
NOT a diagnosis. Weights are a first guess and should be reviewed clinically."""
import numpy as np

W_PCOS, W_IRREG, W_NOSURGE = 0.60, 0.25, 0.15


def irregularity(cycle_lengths):
    """0 (regular) .. 1 (very irregular): cycles outside 21-35 days, or large spread."""
    L = np.asarray(cycle_lengths, float)
    out_of_range = float(np.mean((L < 21) | (L > 35)))
    spread = float(L.std(ddof=1) / 7.0) if len(L) >= 2 else 0.0
    return min(1.0, max(out_of_range, spread))


def compute_risk(pcos_prob, cycle_lengths, surge_detected_flags):
    irreg = irregularity(cycle_lengths)
    no_surge = 1.0 - float(np.mean(surge_detected_flags))
    score = 100 * (W_PCOS * pcos_prob + W_IRREG * irreg + W_NOSURGE * no_surge)
    band = "low" if score < 30 else "moderate" if score < 60 else "elevated"
    return dict(score=round(score, 1), band=band,
                components=dict(pcos_pattern=round(float(pcos_prob), 3),
                                irregularity=round(irreg, 3),
                                cycles_without_surge=round(no_surge, 3)))
