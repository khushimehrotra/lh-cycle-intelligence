"""
FastAPI backend.   Start (from the project root):
    uvicorn api.main:app --reload
Docs / try it in the browser:  http://127.0.0.1:8000/docs
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calibration import rct_to_lh, LOQ, SURGE_THRESHOLD            # noqa: E402
from features.live_features import latest_features, surge_events    # noqa: E402
from models.train_pcos import cycle_features                        # noqa: E402
from models.risk_score import compute_risk                          # noqa: E402
from guidance.rules import get_guidance, PHASES, GOALS              # noqa: E402

DISCLAIMER = ("Research prototype trained on simulated data. Not a medical device and not a "
              "diagnosis. Discuss any concerns with a clinician.")

app = FastAPI(title="LH Monitoring API (prototype)", version="0.1")
surge_bundle = joblib.load(ROOT / "saved_models/surge_xgb.pkl")
pcos_bundle = joblib.load(ROOT / "saved_models/pcos_rf.pkl")


# ------------------------------------------------------------------ schemas
class Reading(BaseModel):
    cycle_day: int = Field(ge=1, le=90, description="Day of cycle (day 1 = first day of period)")
    raw_impedance_kohm: float = Field(gt=20, lt=120, description="Raw Rct from the sensor, kOhm")
    timestamp: Optional[datetime] = None
    battery_level: Optional[int] = Field(None, ge=0, le=100)


class Cycle(BaseModel):
    readings: list[Reading] = Field(min_length=1)
    cycle_length: Optional[int] = Field(None, ge=15, le=90,
                                        description="Total cycle length if known; else last logged day")


class SurgeRequest(BaseModel):
    current_cycle: list[Reading] = Field(min_length=1)
    previous_cycles: list[Cycle] = []


class GuidanceRequest(BaseModel):
    phase: str = Field("baseline", description=f"one of {PHASES}")
    goal: str = Field("tracking", description=f"one of {GOALS}")
    risk_band: Optional[str] = Field(None, description="low / moderate / elevated, from /analyze-pcos-risk")
    cycle_irregular: bool = False
    no_surge_detected: bool = False


class PCOSRequest(BaseModel):
    cycles: list[Cycle] = Field(min_length=1, max_length=3)


# ------------------------------------------------------------------ helpers
def to_arrays(readings):
    rs = sorted(readings, key=lambda r: r.cycle_day)
    days = [r.cycle_day for r in rs]
    if len(set(days)) != len(days):
        raise HTTPException(422, "Duplicate cycle_day values in one cycle.")
    return np.array(days, float), np.array([rct_to_lh(r.raw_impedance_kohm) for r in rs])


def cycle_len(c: Cycle, days):
    return int(c.cycle_length or days.max())


def infer_phase(lh, band):
    events = surge_events(lh)
    if lh[-1] >= SURGE_THRESHOLD or (events and events[-1][-1] == len(lh) - 1):
        return "surge_window"
    if events:
        return "post_surge"
    return "approaching_surge" if band in ("moderate", "high") else "baseline"


def surge_band(p):
    return "low" if p < 0.30 else "moderate" if p < 0.60 else "high"


# Observed on held-out SIMULATED users (see train_surge.py): share of readings in each band that were
# followed by a surge within 24 h. The overall rate was ~2.5%. The score is NOT a calibrated probability.
BAND_NOTE = {"low": "In simulated testing, about 0.3% of readings in this band were followed by a surge within 24 h.",
             "moderate": "In simulated testing, about 5% of readings in this band were followed by a surge within 24 h.",
             "high": "In simulated testing, about 20% of readings in this band were followed by a surge within 24 h."}


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok", "models": ["surge_xgb", "pcos_rf"]}


@app.post("/convert-impedance")
def convert(raw_impedance_kohm: float):
    """Raw sensor reading -> estimated LH. Handy for testing the calibration."""
    lh = rct_to_lh(raw_impedance_kohm)
    return {"lh_est_miu_ml": round(lh, 2), "below_loq": lh < LOQ}


@app.post("/predict-surge")
def predict_surge(req: SurgeRequest):
    days, lh = to_arrays(req.current_cycle)
    prev_lengths, prev_days, prev_lh = [], None, None
    for c in req.previous_cycles:
        d, l = to_arrays(c.readings)
        prev_lengths.append(cycle_len(c, d))
        prev_days, prev_lh = d, l
    feats = latest_features(days, lh, prev_lengths, prev_days, prev_lh)
    X = pd.DataFrame([feats])[surge_bundle["features"]]
    p = float(surge_bundle["model"].predict_proba(X)[0, 1])

    latest = float(lh[-1])
    notes = []
    phase = infer_phase(lh, surge_band(p))
    if phase == "surge_window":
        notes.append("Latest reading is a clear rise above your recent baseline: a surge may be under way.")
    if latest < LOQ:
        notes.append("Latest reading is below the sensor's quantification limit (8 mIU/mL): treat as approximate.")
    if len(lh) < 4:
        notes.append("Fewer than 4 readings this cycle: forecast has low confidence.")
    if not prev_lengths:
        notes.append("No previous cycles supplied: forecast ignores your cycle history.")
    band = surge_band(p)
    return {"surge_score_next_24h": round(p, 3), "band": band, "phase": phase, "band_meaning": BAND_NOTE[band],
            "latest_lh_est_miu_ml": round(latest, 2), "below_loq": latest < LOQ,
            "surge_in_progress": phase == "surge_window", "notes": notes, "disclaimer": DISCLAIMER}


@app.post("/analyze-pcos-risk")
def analyze_pcos(req: PCOSRequest):
    rows, lengths, surge_flags, summaries = [], [], [], []
    for n, c in enumerate(req.cycles, 1):
        days, lh = to_arrays(c.readings)
        if len(lh) < 5:
            raise HTTPException(422, f"Cycle {n} has {len(lh)} readings; at least 5 are needed.")
        L = cycle_len(c, days)
        rows.append(cycle_features(lh, days, L))
        lengths.append(L)
        n_events = len(surge_events(lh))
        has_surge = n_events == 1
        surge_flags.append(has_surge)
        summaries.append({"cycle": n, "length_days": L, "n_readings": len(lh),
                          "baseline_lh_median": round(float(np.median(lh)), 1),
                          "peak_lh": round(float(lh.max()), 1),
                          "peak_to_baseline": round(float(lh.max() / np.median(lh)), 2),
                          "surge_detected": has_surge, "separate_spikes": n_events})
    X = pd.DataFrame(rows)[pcos_bundle["features"]]
    probs = pcos_bundle["model"].predict_proba(X)[:, 1]
    p = float(np.mean(probs))
    risk = compute_risk(p, lengths, surge_flags)

    msg = []
    if not any(surge_flags):
        msg.append("No clear LH surge detected in the supplied cycle(s).")
    if any(x["separate_spikes"] >= 2 for x in summaries):
        msg.append("Several separate LH spikes without one clear surge in at least one cycle.")
    if max(lengths) > 35 or min(lengths) < 21:
        msg.append("At least one cycle length is outside the typical 21-35 day range.")
    if np.median([s["baseline_lh_median"] for s in summaries]) > 12:
        msg.append("Baseline LH is on the higher side.")
    if len(req.cycles) == 1:
        msg.append("Only one cycle supplied: results are less reliable than with 2-3 cycles.")
    if not msg:
        msg.append("No notable PCOS-type pattern in the supplied cycle(s).")
    return {"pcos_pattern_probability": round(p, 3),
            "per_cycle_probability": [round(float(x), 3) for x in probs],
            "risk_score": risk["score"], "risk_band": risk["band"], "risk_components": risk["components"],
            "pattern_summary": msg, "cycles": summaries,
            "no_surge_detected": not any(surge_flags),
            "cycle_irregular": risk["components"]["irregularity"] >= 0.5,
            "disclaimer": DISCLAIMER + " A PCOS diagnosis needs clinical criteria beyond LH."}


@app.post("/guidance")
def guidance(req: GuidanceRequest):
    """Lifestyle & nutrition tips from cycle phase, goal and pattern flags (not from an LH number)."""
    try:
        return get_guidance(req.phase, req.goal, req.risk_band, req.cycle_irregular, req.no_surge_detected)
    except ValueError as e:
        raise HTTPException(422, str(e))
