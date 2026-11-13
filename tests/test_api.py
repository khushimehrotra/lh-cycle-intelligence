"""Run from the project root:  pytest -q"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/features"))
from api.main import app                                   # noqa: E402
from calibration import lh_to_rct, PORTABLE_OFFSET_KOHM    # noqa: E402
from live_features import latest_features, FEATURES        # noqa: E402

client = TestClient(app)
feat = pd.read_csv(ROOT / "data/processed/surge_features.csv")


def payload(df):
    """Rebuild API readings (raw impedance) from a cycle in readings.csv."""
    return [{"cycle_day": int(r.cycle_day), "raw_impedance_kohm": float(r.raw_impedance_kohm)}
            for r in df.itertuples()]


def test_live_features_match_training_features():
    """Train/serve parity: the API must compute the same features as training."""
    cid = feat[feat.cycle_id.str.endswith("_C2")].cycle_id.iloc[5]
    cyc = feat[feat.cycle_id == cid].reset_index(drop=True)
    user = cyc.user_id[0]
    prev = feat[(feat.user_id == user) & (feat.cycle_idx < 2)]
    cycles = pd.read_csv(ROOT / "data/raw/cycles.csv").set_index("cycle_id")
    prev_lengths = [int(cycles.loc[c, "cycle_len"]) for c in sorted(prev.cycle_id.unique())]
    last = prev[prev.cycle_idx == 1]
    for i in (3, 10, len(cyc) - 1):
        got = latest_features(cyc.cycle_day[:i + 1], cyc.lh_est[:i + 1], prev_lengths,
                              last.cycle_day.values, last.lh_est.values)
        want = cyc.loc[i, FEATURES].astype(float)
        np.testing.assert_allclose([got[k] for k in FEATURES], want.values, rtol=1e-6, atol=1e-6)


def test_calibration_round_trip():
    for lh in (2, 8, 25, 60, 150):
        raw = float(lh_to_rct(lh)) + PORTABLE_OFFSET_KOHM
        r = client.post("/convert-impedance", params={"raw_impedance_kohm": raw}).json()
        assert abs(r["lh_est_miu_ml"] - lh) < 0.5


def test_predict_surge_runs_and_ranks():
    r = pd.read_csv(ROOT / "data/raw/readings.csv")
    ok = r[r.cycle_id.str.endswith("_C0")]
    # a healthy cycle: probability just before surge should beat probability early in cycle
    cid = ok[ok.surge_next_24h == 1].cycle_id.iloc[0]
    d = ok[ok.cycle_id == cid].sort_values("cycle_day")
    pos_day = d[d.surge_next_24h == 1].cycle_day.iloc[0]
    early = client.post("/predict-surge", json={"current_cycle": payload(d[d.cycle_day <= 5])}).json()
    late = client.post("/predict-surge", json={"current_cycle": payload(d[d.cycle_day <= pos_day])}).json()
    assert 0 <= early["surge_score_next_24h"] <= 1
    assert late["surge_score_next_24h"] > early["surge_score_next_24h"]


def test_pcos_endpoint_separates_groups():
    r = pd.read_csv(ROOT / "data/raw/readings.csv")
    cy = pd.read_csv(ROOT / "data/raw/cycles.csv")
    scores = {}
    for label, kind in (("healthy", "normal"), ("pcos", "pcos_flat")):
        cids = cy[cy.kind == kind].cycle_id.iloc[:15]
        scores[label] = np.mean([client.post("/analyze-pcos-risk", json={"cycles": [
            {"readings": payload(r[r.cycle_id == c])}]}).json()["pcos_pattern_probability"] for c in cids])
    assert scores["pcos"] > scores["healthy"] + 0.3


def test_bad_input_is_rejected():
    assert client.post("/analyze-pcos-risk", json={"cycles": []}).status_code == 422
    short = [{"cycle_day": d, "raw_impedance_kohm": 74.0} for d in (1, 2, 3)]
    assert client.post("/analyze-pcos-risk", json={"cycles": [{"readings": short}]}).status_code == 422
    dup = [{"cycle_day": 1, "raw_impedance_kohm": 74.0}] * 2
    assert client.post("/predict-surge", json={"current_cycle": dup}).status_code == 422


def test_phase_detection_on_simulated_cycles():
    r = pd.read_csv(ROOT / "data/raw/readings.csv")
    cy = pd.read_csv(ROOT / "data/raw/cycles.csv")
    def reaches_surge_window(cid):
        d = r[r.cycle_id == cid].sort_values("cycle_day")
        for k in range(4, len(d) + 1):
            out = client.post("/predict-surge", json={"current_cycle": payload(d.iloc[:k])}).json()
            if out["phase"] == "surge_window":
                return True
        return False
    normal = [reaches_surge_window(c) for c in cy[cy.kind == "normal"].cycle_id.iloc[:12]]
    flat = [reaches_surge_window(c) for c in cy[cy.kind == "pcos_flat"].cycle_id.iloc[:12]]
    assert np.mean(normal) >= 0.8, "should notice most real surges"
    assert np.mean(flat) <= 0.15, "should rarely call a surge on flat cycles"
