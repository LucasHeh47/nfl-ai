#!/usr/bin/env python3
"""
Model wrapper for inference.

API:
  - load_model(model_dir) -> (clf, meta)
  - predict_week(model_dir, season, week) -> DataFrame with per-game predictions
"""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
from joblib import load

DATA_DIR = Path("data")
MODELS_DIR = Path("predictor/saved_models")

def _load_game_features(season: int) -> pd.DataFrame:
    p = DATA_DIR / f"features_games_{season}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run predictor/features.py first.")
    return pd.read_parquet(p)

def load_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    clf = load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "meta.json").read_text())
    return clf, meta

def predict_week(model_dir: str | Path, season: int, week: int) -> pd.DataFrame:
    # requires: import numpy as np, from joblib import load
    model_dir = Path(model_dir)  # ensure Path
    print(f"Predicting week with model: {model_dir.resolve()}")  # <-- fix
    clf, meta = load_model(model_dir)
    feats = _load_game_features(season)

    mask = feats["week"] == week
    X = feats.loc[mask, meta["feature_cols"]]
    if X.empty:
        return pd.DataFrame(columns=["game_id","week","home_team","away_team","home_win_prob","pred_margin"])

    # --- probabilities ---
    p = clf.predict_proba(X)[:, 1]

    # --- margins ---
    # 1) try learned margin regressor if present
    margin = None
    try:
        margin_path = Path(model_dir) / "margin.joblib"
        if margin_path.exists():
            margin_reg = load(margin_path)
            # you can reuse the same features or a different set stored in meta
            m_feats = feats.loc[mask, meta.get("margin_feature_cols", meta["feature_cols"])]
            margin = margin_reg.predict(m_feats).astype(float)
    except Exception:
        margin = None

    # 2) fallback heuristic from probability
    if margin is None:
        K = float(meta.get("prob_to_margin_k", 14.0))  # typical NFL spread scale
        margin = (p - 0.5) * K

    # --- pack output ---
    out = feats.loc[mask, ["game_id","week"]].copy()
    out["home_team"] = feats.loc[mask, "home_team"].values
    out["away_team"] = feats.loc[mask, "away_team"].values
    out["home_win_prob"] = p.astype(float)
    out["pred_margin"] = margin.astype(float)
    return out

