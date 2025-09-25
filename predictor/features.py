#!/usr/bin/env python3
"""
Schedule-only feature builder.

Inputs:
  data/schedules_<season>.csv   (must include: season, week, home_team, away_team,
                                 home_score, away_score)

Outputs:
  data/features_teamweek_<season>.parquet  # one row per TEAM x WEEK (pre-game view)
  data/features_games_<season>.parquet     # one row per GAME (home vs away, diffs)
"""

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

DATA_DIR = Path("data")

def _read_schedule(season: int) -> pd.DataFrame:
    p = DATA_DIR / f"schedules_{season}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run predictor/refresh_schedules.py first.")
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]

    needed = ["season","week","home_team","away_team","home_score","away_score"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"{p} missing columns: {missing}")

    # normalize & types
    for c in ["home_team","away_team"]:
        df[c] = df[c].astype("string").str.upper().str.strip()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"]   = pd.to_numeric(df["week"],   errors="coerce").astype("Int64")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # game_id if not present
    if "game_id" not in df.columns:
        df["game_id"] = (
            df["season"].astype(str) + "_w" + df["week"].astype(str) + "_" +
            df["home_team"] + "_" + df["away_team"]
        )
    return df

def _build_teamweek_from_schedule(sched: pd.DataFrame) -> pd.DataFrame:
    """
    Make two rows per game (home + away) with raw PF/PA for that week,
    then compute pre-game aggregates (shifted by 1 week).
    """
    # long rows: one per team per game
    home = sched[["game_id","season","week","home_team","away_team","home_score","away_score"]].copy()
    home.rename(columns={
        "home_team":"team", "away_team":"opp_team",
        "home_score":"pf",  "away_score":"pa"
    }, inplace=True)
    home["is_home"] = 1

    away = sched[["game_id","season","week","home_team","away_team","home_score","away_score"]].copy()
    away.rename(columns={
        "away_team":"team", "home_team":"opp_team",
        "away_score":"pf",  "home_score":"pa"
    }, inplace=True)
    away["is_home"] = 0

    tw = pd.concat([home, away], ignore_index=True)
    tw = tw.sort_values(["team","week"]).reset_index(drop=True)

    # group-wise pre-game aggregates (use only prior weeks via shift)
    # cumulative through week-1
    tw["cum_pf"] = tw.groupby("team")["pf"].cumsum().shift(1).fillna(0.0)
    tw["cum_pa"] = tw.groupby("team")["pa"].cumsum().shift(1).fillna(0.0)
    tw["cum_point_diff"] = tw["cum_pf"] - tw["cum_pa"]

    # last-3 sums through week-1
    def _roll3(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(3, min_periods=1).sum().fillna(0.0)

    tw["r3_pf"] = tw.groupby("team", group_keys=False)["pf"].apply(_roll3)
    tw["r3_pa"] = tw.groupby("team", group_keys=False)["pa"].apply(_roll3)
    tw["r3_point_diff"] = tw["r3_pf"] - tw["r3_pa"]

    # rename to stable schema used by trainer
    tw.rename(columns={
        "cum_pf":"cum_points_for",
        "cum_pa":"cum_points_against",
        "r3_pf":"roll3_pf",
        "r3_pa":"roll3_pa",
    }, inplace=True)

    # keep compact set
    cols = ["game_id","season","week","team","opp_team","is_home",
            "cum_points_for","cum_points_against","cum_point_diff",
            "roll3_pf","roll3_pa","r3_point_diff","pf","pa"]
    return tw[cols]

def _build_games_from_teamweek(sched: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    # home rows for each game/week
    H = tw.merge(
        sched[["game_id","week","home_team","away_team"]],
        left_on=["game_id","team","week"],
        right_on=["game_id","home_team","week"],
        how="inner"
    )
    # away rows for each game/week
    A = tw.merge(
        sched[["game_id","week","home_team","away_team"]],
        left_on=["game_id","team","week"],
        right_on=["game_id","away_team","week"],
        how="inner"
    )

    H = H[["game_id","week","team","cum_point_diff","roll3_pf","roll3_pa","cum_points_for","cum_points_against"]]
    A = A[["game_id","week","team","cum_point_diff","roll3_pf","roll3_pa","cum_points_for","cum_points_against"]]

    H.columns = ["game_id","week","home_team","home_cum_pd","home_r3_pf","home_r3_pa","home_cum_pf","home_cum_pa"]
    A.columns = ["game_id","week","away_team","away_cum_pd","away_r3_pf","away_r3_pa","away_cum_pf","away_cum_pa"]

    games = H.merge(A, on=["game_id","week"], how="inner")

    # diffs (home - away)
    games["diff_cum_pd"] = games["home_cum_pd"] - games["away_cum_pd"]
    # convert r3_pf/pa to r3_pd home/away then diff
    games["home_r3_pd"] = games["home_r3_pf"] - games["home_r3_pa"]
    games["away_r3_pd"] = games["away_r3_pf"] - games["away_r3_pa"]
    games["diff_r3_pd"]  = games["home_r3_pd"] - games["away_r3_pd"]

    games["diff_cum_pf"] = games["home_cum_pf"] - games["away_cum_pf"]
    games["diff_cum_pa"] = games["home_cum_pa"] - games["away_cum_pa"]
    games["home_field"]  = 1  # flag; refine if you want neutral-site detection

    # keep columns trainer expects + team names (for convenience)
    keep = [
        "game_id","week","home_team","away_team",
        "diff_cum_pd","diff_r3_pd","diff_cum_pf","diff_cum_pa","home_field"
    ]
    return games[keep]

def build_and_save_features(season: int) -> tuple[Path, Path]:
    sched = _read_schedule(season)
    teamweek = _build_teamweek_from_schedule(sched)
    games = _build_games_from_teamweek(sched, teamweek)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p_tw = DATA_DIR / f"features_teamweek_{season}.parquet"
    p_gm = DATA_DIR / f"features_games_{season}.parquet"
    teamweek.to_parquet(p_tw, index=False)
    games.to_parquet(p_gm, index=False)
    print(f"[OK] wrote {p_tw}")
    print(f"[OK] wrote {p_gm}")
    return p_tw, p_gm

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()
    build_and_save_features(args.season)
