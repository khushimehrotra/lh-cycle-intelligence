"""
Mock app: a simulated biosensor + your ML API.   Start (2nd terminal, API already running):
    streamlit run app/streamlit_app.py
"""
import subprocess
import sys
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_gen"))
import synth_lh as sim                                   # noqa: E402  (our simulator = fake hardware)

SURGE_T, LOQ = sim.SURGE_THRESHOLD, sim.LOQ
PROFILES = {"Healthy (normal ovulation)": (False, "normal"),
            "PCOS: flat LH": (True, "pcos_flat"),
            "PCOS: erratic spikes": (True, "pcos_erratic"),
            "PCOS: delayed ovulation": (True, "pcos_delayed")}
DISCLAIMER = ("Research prototype running on SIMULATED data. Not a medical device and not a diagnosis.")

st.set_page_config(page_title="LH Monitor (prototype)", page_icon="🩺", layout="wide")


# ------------------------------------------------------------------ simulation (stands in for Bluetooth)
def new_simulation(profile, seed, noise):
    pcos, kind = PROFILES[profile]
    rng = np.random.default_rng(seed)
    user = sim.make_user(rng, "DEMO", pcos, overlap=0.3)
    prev = []
    for _ in range(2):                                   # two earlier cycles already "on the phone"
        cyc = sim.simulate_cycle(rng, user, kind)
        rows = sim.measure(rng, user, cyc, noise)
        prev.append({"readings": [{"cycle_day": r["cycle_day"], "raw_impedance_kohm": r["raw_impedance_kohm"]}
                                  for r in rows], "cycle_length": cyc["cyc_len"]})
    return dict(key=(profile, seed, noise), rng=rng, user=user, noise=noise, prev=prev,
                cyc=sim.simulate_cycle(rng, user, kind), day=0, readings=[], scores=[],
                battery=90.0, start=dt.datetime.now().replace(hour=8, minute=0, second=0, microsecond=0),
                last_payload=None, pcos=None, error=None)


@st.cache_resource(show_spinner="First start: building the simulated data and models (about 30 seconds)...")
def get_local_client():
    """Run the FastAPI app inside this process, so no separate server is needed (e.g. on Streamlit Cloud).
    If the trained models are missing, build them first."""
    from fastapi.testclient import TestClient
    if not ((ROOT / "saved_models" / "surge_xgb.pkl").exists() and (ROOT / "saved_models" / "pcos_rf.pkl").exists()):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py")], cwd=ROOT, check=True)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from api.main import app as fastapi_app
    return TestClient(fastapi_app)


def api_post(path, payload):
    url = st.session_state.api_url.strip()
    try:
        if url.lower() in ("", "local"):
            r = get_local_client().post(path, json=payload)
        else:
            r = requests.post(url.rstrip("/") + path, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.session_state.sim["error"] = f"Could not reach the API ({e}). Use 'local', or start the API with `uvicorn api.main:app`."
        return None


def log_measurement(s):
    """One simulated sensor reading -> API -> surge score."""
    d = s["day"] + 1
    cyc, u, rng = s["cyc"], s["user"], s["rng"]
    if d > cyc["cyc_len"]:
        s["error"] = "This simulated cycle has ended. Start a new simulated user in the sidebar."
        return False
    hr = int(np.clip((d - 1) * 24 + u["meas_hour"] + rng.normal(0, 1), 0, cyc["cyc_len"] * 24 - 1))
    raw = float(sim.lh_to_rct(cyc["lh"][hr]) + u["device_offset"] + rng.normal(0, s["noise"]))
    s["day"] = d
    s["battery"] = max(5.0, s["battery"] - 0.3)
    ts = s["start"] + dt.timedelta(days=d - 1)
    s["last_payload"] = {"timestamp": ts.isoformat() + "Z", "raw_impedance_kohm": round(raw, 2),
                         "battery_level": int(s["battery"])}
    s["readings"].append({"cycle_day": d, "raw_impedance_kohm": round(raw, 2)})
    res = api_post("/predict-surge", {"current_cycle": s["readings"], "previous_cycles": s["prev"]})
    if res is None:
        s["readings"].pop(); s["day"] = d - 1
        return False
    s["scores"].append({"cycle_day": d, **res})
    s["error"] = None
    return True


def cb_measure():
    log_measurement(st.session_state.sim)


def cb_week():
    s = st.session_state.sim
    for _ in range(7):
        if not log_measurement(s):
            break


def cb_skip():
    s = st.session_state.sim
    s["day"] += 1                                         # a missed day: no reading logged


def cb_analyze():
    s = st.session_state.sim
    cycles = list(s["prev"])
    if len(s["readings"]) >= 5:
        cycles.append({"readings": s["readings"], "cycle_length": None})
    s["pcos"] = api_post("/analyze-pcos-risk", {"cycles": cycles[-3:]})


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Simulated user")
st.session_state.setdefault("api_url", "local")
st.sidebar.text_input("API address", key="api_url", help="Type local to run the models inside the app, or an address such as http://127.0.0.1:8000")
if st.session_state.api_url.strip().lower() in ("", "local"):
    get_local_client()
profile = st.sidebar.selectbox("Cycle pattern to simulate", list(PROFILES))
noise = st.sidebar.slider("Sensor noise (kΩ)", 0.2, 1.5, 0.5, 0.1)
seed = int(st.sidebar.number_input("Random seed", 0, 9999, 7))
GOAL_LABELS = {"Just tracking my cycle": "tracking", "Trying to conceive": "trying_to_conceive",
               "Avoiding pregnancy": "avoiding_pregnancy"}
goal_label = st.sidebar.selectbox("My goal", list(GOAL_LABELS))
debug = st.sidebar.checkbox("Show hidden true LH (debug)")
if st.sidebar.button("Start new simulated user") or "sim" not in st.session_state \
        or st.session_state.sim["key"] != (profile, seed, noise):
    st.session_state.sim = new_simulation(profile, seed, noise)
s = st.session_state.sim

# ------------------------------------------------------------------ main page
st.title("🩺 LH Monitor: prototype")
st.warning(DISCLAIMER)
c1, c2, c3 = st.columns([2, 1, 1])
c1.button("🔬 Simulate Biosensor Measurement", on_click=cb_measure, type="primary", key="sim_btn")
c2.button("⏩ +7 days", on_click=cb_week, key="week_btn")
c3.button("Skip a day (missed reading)", on_click=cb_skip, key="skip_btn")
if s["error"]:
    st.error(s["error"])

tab1, tab2, tab3 = st.tabs(["Today & surge forecast", "PCOS pattern & risk", "Lifestyle & nutrition tips"])

with tab1:
    last = s["scores"][-1] if s["scores"] else None
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cycle day", s["day"])
    m2.metric("Latest LH (estimated)", f"{last['latest_lh_est_miu_ml']} mIU/mL" if last else "-")
    m3.metric("Surge score, next 24 h", f"{last['surge_score_next_24h']:.2f}" if last else "-")
    m4.metric("Level", last["band"].upper() if last else "-")
    if last:
        if last["surge_in_progress"]:
            st.success("🌟 Latest reading is a clear rise above your usual baseline: a surge may be under way.")
        elif last["band"] == "high":
            st.info("Elevated chance of a surge within about 24 h.")
        st.caption(last["band_meaning"])
        for n in last["notes"]:
            st.caption("• " + n)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    if debug:
        t = np.arange(len(s["cyc"]["lh"])) / 24 + 1
        keep = t <= max(s["day"], 1) + 1
        a1.plot(t[keep], s["cyc"]["lh"][keep], color="lightgrey", lw=1, label="hidden true LH")
    if s["scores"]:
        df = pd.DataFrame(s["scores"])
        a1.plot(df.cycle_day, df.latest_lh_est_miu_ml, "-o", ms=4, label="LH from sensor")
        a2.bar(df.cycle_day, df.surge_score_next_24h, color="tab:orange")
    a1.axhline(SURGE_T, color="grey", ls=":", lw=1); a1.text(0.5, SURGE_T + 0.5, "reference level (25)", fontsize=8)
    a1.axhline(LOQ, color="red", ls=":", lw=1); a1.text(0.5, LOQ + 0.5, "quantification limit (8)", fontsize=8, color="red")
    a1.set_ylabel("LH (mIU/mL)")
    if s["scores"] or debug:
        a1.legend(loc="upper left", fontsize=8)
    a2.axhline(0.3, color="grey", ls=":", lw=1); a2.axhline(0.6, color="grey", ls=":", lw=1)
    a2.set_ylim(0, 1); a2.set_ylabel("surge score"); a2.set_xlabel("cycle day")
    a2.set_xlim(0, max(30, s["day"] + 2))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("What the 'hardware' just sent (JSON the real device will send)"):
        st.json(s["last_payload"] or {"info": "press the measurement button"})
    with st.expander("All readings this cycle"):
        st.dataframe(pd.DataFrame(s["readings"]))

with tab2:
    st.write("Uses the 2 earlier cycles on the phone plus the current cycle (once it has 5+ readings).")
    st.button("Analyze cycles", on_click=cb_analyze, key="analyze_btn")
    r = s["pcos"]
    if r:
        k1, k2, k3 = st.columns(3)
        k1.metric("Risk score (0-100)", r["risk_score"])
        k2.metric("Level", r["risk_band"].upper())
        k3.metric("PCOS-pattern probability", f"{r['pcos_pattern_probability']:.2f}")
        st.progress(min(int(r["risk_score"]), 100) / 100)
        for line in r["pattern_summary"]:
            st.write("• " + line)
        st.dataframe(pd.DataFrame(r["cycles"]), hide_index=True)
        with st.expander("How the score is built"):
            st.json(r["risk_components"])
        st.caption(r["disclaimer"])


with tab3:
    last = s["scores"][-1] if s["scores"] else None
    pc = s["pcos"] or {}
    g = api_post("/guidance", {"phase": last["phase"] if last else "baseline",
                               "goal": GOAL_LABELS[goal_label],
                               "risk_band": pc.get("risk_band"),
                               "cycle_irregular": bool(pc.get("cycle_irregular", False)),
                               "no_surge_detected": bool(pc.get("no_surge_detected", False))})
    if g:
        st.subheader(g["headline"])
        st.info("**What food can and can't do:** " + g["basis"])
        if not s["pcos"]:
            st.caption("Run **Analyze cycles** in the PCOS tab to add pattern-based tips.")
        for cat in dict.fromkeys(i["category"] for i in g["items"]):
            st.markdown(f"#### {cat}")
            for i in [x for x in g["items"] if x["category"] == cat]:
                src = f" · source: {i['source']}" if i["source"] else ""
                st.markdown(f"**{i['title']}**  \n{i['text']}")
                st.caption(f"Evidence: {i['evidence_label']}{src}")
        with st.expander("When to see a clinician"):
            for line in g["when_to_seek_care"]:
                st.write("• " + line)
        with st.expander("Sources"):
            for src in g["sources"]:
                st.markdown(f"- [{src['title']}]({src['url']})")
        st.caption(g["disclaimer"])
