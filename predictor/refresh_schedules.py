#!/usr/bin/env python3
import nflreadpy as nfl
import pandas as pd
from pathlib import Path

try:
    import polars as pl
except ImportError:
    pl = None

DATA_DIR = Path("data")

def refresh_schedule(season: int):
    print(f"Fetching schedules for {season} ...")
    df = nfl.load_schedules([season])
    
    # Convert Polars -> Pandas if needed
    if pl is not None and isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Select useful columns
    keep_cols = [
        "game_id", "season", "week", "home_team", "away_team",
        "home_score", "away_score", "result", "spread_line", "total_line"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    out_file = DATA_DIR / f"schedules_{season}.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Saved {out_file} with {len(df)} rows and {len(df.columns)} columns.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g., 2023)")
    args = parser.parse_args()
    refresh_schedule(args.season)
