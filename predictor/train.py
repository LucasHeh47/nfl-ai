#!/usr/bin/env python3
"""
Train a simple baseline classifier on historical games.

Target: home team win (1) vs loss (0)
Features: diffs derived in features_games_<season>.parquet
Train/Val split: by week (<= through_week for train, == through_week+1 for val)
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, time
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from joblib import dump

DATA_DIR = Path("data")
MODELS_DIR = Path("predictor/saved_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "diff_cum_pd","diff_r3_pd","diff_cum_pf","diff_cum_pa","home_field"
]

def _load_game_features(season: int) -> pd.DataFrame:
    p = DATA_DIR / f"features_games_{season}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run predictor/features.py first.")
    df = pd.read_parquet(p)
    return df

def _load_actuals_from_stats(season: int, df_games: pd.DataFrame) -> pd.Series:
    """
    Minimal 'actual' win labels for training from team_stats by comparing PF/PA
    at the specific week. We infer home win if home's PF - PA > away's PF - PA in that week.
    (This is a rough proxy; feel free to replace with a better actual source later.)
    """
    # Expect that team_stats rows include weekly PF/PA. If not, this proxy will be noisy.
    # For a reliable target, you can export a scores table from schedules (if available).
    # For now we'll mark as NA and drop if we can't resolve.
    labels = []
    for _, r in df_games.iterrows():
        # without a robust scores source, skip actuals here
        labels.append(np.nan)
    return pd.Series(labels, index=df_games.index, dtype="float")

def _load_actuals_from_schedule_if_available(season: int, df_games: pd.DataFrame) -> pd.Series:
    path = DATA_DIR / f"schedules_{season}.csv"
    if not path.exists():
        return pd.Series([np.nan]*len(df_games), index=df_games.index)

    sch = pd.read_csv(path)
    sch.columns = [c.lower() for c in sch.columns]

    # score columns
    hsc = next((c for c in ["home_score","home_points","h_pts","hscore"] if c in sch.columns), None)
    asc = next((c for c in ["away_score","away_points","a_pts","ascore"] if c in sch.columns), None)
    if not (hsc and asc):
        return pd.Series([np.nan]*len(df_games), index=df_games.index)

    # normalize teams for key merge fallback
    for c in ["home_team","away_team"]:
        if c in sch.columns:
            sch[c] = sch[c].astype("string").str.upper().str.strip()

    # Preferred: merge directly on native game_id if both sides have it
    if "game_id" in sch.columns and "game_id" in df_games.columns:
        merged = df_games.merge(sch[["game_id", hsc, asc]], on="game_id", how="left")
    else:
        # Fallback multi-key merge
        cols = ["season","week","home_team","away_team"]
        for c in cols:
            if c in df_games.columns:
                df_games[c] = df_games[c].astype("string").str.upper().str.strip()
            if c in sch.columns:
                sch[c] = sch[c].astype("string").str.upper().str.strip()
        merged = df_games.merge(sch[cols + [hsc, asc]], on=cols, how="left")

    y = (pd.to_numeric(merged[hsc], errors="coerce") > pd.to_numeric(merged[asc], errors="coerce")).astype("float")
    y[pd.isna(merged[hsc]) | pd.isna(merged[asc])] = np.nan
    y.index = df_games.index
    return y


def train(season: int, through_week: int, save_name: str | None = None):
    df = _load_game_features(season)

    # Define target from schedule (preferred). If missing, labels will be NaN.
    y = _load_actuals_from_schedule_if_available(season, df)

    # Train on games strictly before or equal to through_week (requires y not NaN)
    train_mask = (df["week"] <= through_week) & (~y.isna())
    X_train = df.loc[train_mask, FEATURE_COLS]
    y_train = y.loc[train_mask].astype(int)

    if X_train.empty:
        raise RuntimeError("No training rows found. Ensure schedules include final scores or adjust target creation.")

    # Simple logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Validate on the immediate next week if labels exist
    val_mask = (df["week"] == (through_week + 1)) & (~y.isna())
    metrics = {}
    if val_mask.any():
        X_val = df.loc[val_mask, FEATURE_COLS]
        y_val = y.loc[val_mask].astype(int)
        p_val = clf.predict_proba(X_val)[:, 1]
        metrics = {
            "logloss": float(log_loss(y_val, p_val)),
            "brier": float(brier_score_loss(y_val, p_val)),
            "accuracy": float(accuracy_score(y_val, (p_val >= 0.5).astype(int))),
            "n_val": int(len(y_val))
        }

    # Save model
    stamp = save_name or f"lr_s{season}_w{through_week:02d}"
    out_dir = MODELS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    dump(clf, out_dir / "model.joblib")
    meta = {
        "name": save_name or out_dir.name,
        "season": season,
        "through_week": through_week,
        "feature_cols": FEATURE_COLS,
        "created_at": int(time.time()),
        "metrics": metrics
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] saved model to {out_dir}")
    if metrics:
        print(f"  metrics: {metrics}")
    return out_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--through-week", type=int, required=True)
    ap.add_argument("--name", type=str, default=None)
    args = ap.parse_args()
    train(args.season, args.through_week, args.name)
