"""
Train the surge forecaster: 'will an LH surge start within the next 24 h?'

Compares three models, split BY USER (no person appears in both train and test):
  1. calendar baseline  - logistic regression on cycle-day features only
  2. XGBoost            - all features

Run from the project root (after build_features.py):
    python src/models/train_surge.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src/features"))
from build_features import FEATURES, TARGET  # noqa: E402

CALENDAR_ONLY = ["cycle_day", "prev_len_mean", "day_frac_of_prev_len", "day_vs_prev_peak"]


def recall_at_precision(y, p, min_precision=0.5):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return rec[ok].max() if ok.any() else 0.0


def report(name, y, p):
    print(f"{name:<22} ROC-AUC {roc_auc_score(y, p):.3f} | "
          f"PR-AUC {average_precision_score(y, p):.3f} | "
          f"recall@precision>=0.5: {recall_at_precision(y, p):.2f}")


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data/processed/surge_features.csv")
    X, y, groups = df[FEATURES], df[TARGET], df["user_id"]

    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    print(f"train {len(tr):,} rows / test {len(te):,} rows | "
          f"users in test: {groups.iloc[te].nunique()} | positives: {yte.mean():.1%}")

    base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))
    base.fit(Xtr[CALENDAR_ONLY], ytr)
    report("calendar baseline", yte, base.predict_proba(Xte[CALENDAR_ONLY])[:, 1])

    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),
                        eval_metric="aucpr", random_state=0)
    xgb.fit(Xtr, ytr)
    p = xgb.predict_proba(Xte)[:, 1]
    report("XGBoost (all feats)", yte, p)

    imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop features:\n" + imp.head(8).round(3).to_string())

    (ROOT / "saved_models").mkdir(exist_ok=True)
    joblib.dump({"model": xgb, "features": FEATURES}, ROOT / "saved_models/surge_xgb.pkl")
    print("\nSaved saved_models/surge_xgb.pkl")
