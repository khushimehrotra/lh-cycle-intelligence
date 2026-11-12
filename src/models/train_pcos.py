"""
PCOS-pattern classifier (cycle level) + 1-3 cycle user-level score.

Each CYCLE is summarised into a handful of numbers (baseline, peak, spikiness...)
computed only from what the sensor/app sees (lh_est), then fed to a classifier.
Evaluation is 5-fold, split BY USER, so nobody appears in both train and test.

Run from the project root (after synth_lh.py):
    python src/models/train_pcos.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
LOQ, SURGE = 8.0, 25.0


def cycle_features(lh: np.ndarray, days: np.ndarray, cycle_len: int) -> dict:
    """lh = estimated LH values of one cycle (mIU/mL), days = their cycle days."""
    base = np.median(lh)
    hi = lh > 1.5 * base
    # longest run of consecutive readings clearly above baseline
    run = best = 0
    for v in hi:
        run = run + 1 if v else 0
        best = max(best, run)
    peaks = [i for i in range(1, len(lh) - 1)
             if lh[i] > lh[i - 1] and lh[i] > lh[i + 1] and lh[i] > 1.3 * base]
    return dict(
        cycle_len=cycle_len,
        n_readings=len(lh),
        base_median=base,
        base_mean=lh.mean(),
        lh_std=lh.std(),
        lh_cv=lh.std() / max(lh.mean(), 1e-6),
        lh_max=lh.max(),
        peak_to_base=lh.max() / max(base, 1e-6),
        frac_above_surge=(lh >= SURGE).mean(),
        frac_below_loq=(lh < LOQ).mean(),
        n_spikes=len(peaks),
        max_run_high=best,
        day_to_day_change=np.abs(np.diff(lh)).mean() if len(lh) > 1 else 0.0,
        peak_day_frac=days[np.argmax(lh)] / cycle_len,
    )


def build_cycle_table(readings, cycles):
    rows = []
    meta = cycles.set_index("cycle_id")
    for cid, d in readings.groupby("cycle_id"):
        d = d.sort_values("cycle_day")
        if len(d) < 5:
            continue
        m = meta.loc[cid]
        f = cycle_features(d.lh_est.values, d.cycle_day.values, int(m.cycle_len))
        rows.append(dict(cycle_id=cid, user_id=m.user_id, kind=m.kind,
                         has_surge=int(m.has_surge), pcos_user=int(m.pcos_user), **f))
    return pd.DataFrame(rows)


def make_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                               class_weight="balanced", random_state=0, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, random_state=0, eval_metric="logloss"),
    }


if __name__ == "__main__":
    readings = pd.read_csv(ROOT / "data/raw/readings.csv")
    cycles = pd.read_csv(ROOT / "data/raw/cycles.csv")
    tab = build_cycle_table(readings, cycles)
    tab.to_csv(ROOT / "data/processed/cycle_features.csv", index=False)
    FEATS = [c for c in tab.columns if c not in ("cycle_id", "user_id", "kind", "has_surge", "pcos_user")]
    X, y, g = tab[FEATS], tab.pcos_user, tab.user_id
    print(f"{len(tab):,} cycles | PCOS-user cycles: {y.mean():.0%}\n")

    for name, model in make_models().items():
        oof = np.zeros(len(tab))
        for tr, te in GroupKFold(5).split(X, y, g):
            model.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        tab[f"p_{name}"] = oof
        print(f"{name:<13} cycle-level ROC-AUC {roc_auc_score(y, oof):.3f} | "
              f"PR-AUC {average_precision_score(y, oof):.3f}")

    # --- where does the model flag / miss? (RandomForest, threshold 0.5) ---
    tab["flagged"] = tab["p_RandomForest"] >= 0.5
    print("\nShare of cycles flagged as PCOS-pattern, by cycle type:")
    print(tab.groupby("kind").flagged.mean().round(2).to_string())

    # --- user-level score: average of up to 3 cycles per user ---
    u = (tab.sort_values("cycle_id").groupby("user_id")
            .agg(p=("p_RandomForest", lambda s: s.iloc[:3].mean()), y=("pcos_user", "first")))
    print(f"\nUser-level (mean of 3 cycles) ROC-AUC: {roc_auc_score(u.y, u.p):.3f}")

    # --- final model on all data ---
    final = make_models()["RandomForest"].fit(X, y)
    (ROOT / "saved_models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATS}, ROOT / "saved_models/pcos_rf.pkl")
    imp = pd.Series(final.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\nTop features:\n" + imp.head(6).round(3).to_string())
    print("\nSaved saved_models/pcos_rf.pkl")
