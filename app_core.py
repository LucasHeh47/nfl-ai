#!/usr/bin/env python3
import sys, os, io, zipfile
from pathlib import Path
from typing import Dict, List, Optional

import re
import argparse
import pandas as pd
import nflreadpy as nfl

# Polars is optional; use it if the loader returns polars
try:
    import polars as pl
except Exception:
    pl = None


# ---- Constants ----
TEAM_ABBRS: List[str] = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SF","SEA","TB","TEN","WAS"
]

TEAM_ALIASES: dict[str, list[str]] = {
    "LAR": ["LAR", "LA", "LOS ANGELES RAMS", "STL", "ST. LOUIS RAMS"],
    "LAC": ["LAC", "LA CHARGERS", "LOS ANGELES CHARGERS", "SD", "SAN DIEGO CHARGERS"],
    "LV":  ["LV", "LAS VEGAS RAIDERS", "OAK", "OAKLAND RAIDERS"],
    "WAS": ["WAS", "WSH", "WASHINGTON", "WASHINGTON COMMANDERS", "WASHINGTON FOOTBALL TEAM"],
    # conservative aliases for others (abbr and full name)
    "ARI": ["ARI", "ARIZONA CARDINALS"],
    "ATL": ["ATL", "ATLANTA FALCONS"],
    "BAL": ["BAL", "BALTIMORE RAVENS"],
    "BUF": ["BUF", "BUFFALO BILLS"],
    "CAR": ["CAR", "CAROLINA PANTHERS"],
    "CHI": ["CHI", "CHICAGO BEARS"],
    "CIN": ["CIN", "CINCINNATI BENGALS"],
    "CLE": ["CLE", "CLEVELAND BROWNS"],
    "DAL": ["DAL", "DALLAS COWBOYS"],
    "DEN": ["DEN", "DENVER BRONCOS"],
    "DET": ["DET", "DETROIT LIONS"],
    "GB":  ["GB",  "GREEN BAY PACKERS", "GNB"],
    "HOU": ["HOU", "HOUSTON TEXANS"],
    "IND": ["IND", "INDIANAPOLIS COLTS"],
    "JAX": ["JAX", "JAC", "JACKSONVILLE JAGUARS"],
    "KC":  ["KC",  "KAN", "KANSAS CITY CHIEFS"],
    "MIA": ["MIA", "MIAMI DOLPHINS"],
    "MIN": ["MIN", "MINNESOTA VIKINGS"],
    "NE":  ["NE",  "NEW ENGLAND PATRIOTS", "NWE"],
    "NO":  ["NO",  "NEW ORLEANS SAINTS", "NOR"],
    "NYG": ["NYG", "NEW YORK GIANTS"],
    "NYJ": ["NYJ", "NEW YORK JETS"],
    "PHI": ["PHI", "PHILADELPHIA EAGLES"],
    "PIT": ["PIT", "PITTSBURGH STEELERS"],
    "SF":  ["SF",  "SAN FRANCISCO 49ERS", "SFO"],
    "SEA": ["SEA", "SEATTLE SEAHAWKS"],
    "TB":  ["TB",  "TAMPA BAY BUCCANEERS", "TAM"],
    "TEN": ["TEN", "TENNESSEE TITANS"],
    "BUF": ["BUF", "BUFFALO BILLS"],
    "DAL": ["DAL", "DALLAS COWBOYS"],
    "LVR": ["LV"],  # safety
}

DATA_DIR = Path("data")
TEAM_DIR = DATA_DIR  # keeping flat; each team gets a single ZIP file

STAT_EXCLUDE_COLS = {
    "team","team_abbr","Tm","abbr","team_name","name",
    "season","season_year","year","yr",
    "week","wk","game","g","gms","games",
    "division","conference","conf","div"
}

RATE_HINTS = (
    "rate", "pct", "percent", "percentage", "avg", "average", "per_",
    "cpoe", "epa", "success", "sr", "dvoa", "ppa"
)



# ---- Schedule Loader ----
def _load_schedules_for_season(season: int) -> pd.DataFrame:
    """
    Load schedules for a single season via nflreadpy and return a pandas DataFrame.
    Tries season-specific first, then broad load + filter. Normalizes to pandas.
    """
    # Try the season-specific signature first
    try:
        df = nfl.load_schedules(seasons=[season])
    except TypeError:
        # Older signatures sometimes exist
        df = nfl.load_schedules()

    # If it's polars, convert to pandas for consistent downstream use
    if _is_polars(df):
        df = df.to_pandas()

    # Try to filter by season if a season column exists
    for scol in ["season", "season_year", "year", "yr"]:
        if scol in df.columns:
            df = df.loc[pd.to_numeric(df[scol], errors="coerce") == season].copy()
            break
    return df


def _export_schedule_csv(season: int) -> Path:
    """
    Save schedules_<season>.csv under ./data with normalized column names:
    season, week, home_team, away_team, kickoff_datetime
    Returns the saved path.
    """
    df = _load_schedules_for_season(season)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No schedule rows found for season {season}")

    # Lowercase all columns for easy matching
    df.columns = [str(c).lower() for c in df.columns]

    # Map common schedule field names
    season_col = next((c for c in ["season", "season_year", "year", "yr"] if c in df.columns), None)
    week_col   = next((c for c in ["week","wk","gameweek","g"] if c in df.columns), None)
    home_col   = next((c for c in ["home_team","home","home_team_abbr","home_abbr","homeabbr"] if c in df.columns), None)
    away_col   = next((c for c in ["away_team","away","away_team_abbr","away_abbr","awayabbr"] if c in df.columns), None)
    ko_col     = next((c for c in ["kickoff_datetime","kickoff","game_date","datetime","start_time"] if c in df.columns), None)

    if not all([season_col, week_col, home_col, away_col]):
        # Give a helpful error so you can see what columns exist
        raise RuntimeError(f"Schedule columns missing. Have: {list(df.columns)}")

    out = pd.DataFrame({
        "season": pd.to_numeric(df[season_col], errors="coerce").astype("Int64"),
        "week":   pd.to_numeric(df[week_col],   errors="coerce").astype("Int64"),
        "home_team": df[home_col].astype("string").str.upper().str.strip(),
        "away_team": df[away_col].astype("string").str.upper().str.strip(),
        "kickoff_datetime": df[ko_col] if ko_col else pd.Series([None]*len(df))
    })

    ensure_dir(DATA_DIR)
    out_path = DATA_DIR / f"schedules_{season}.csv"
    out.to_csv(out_path, index=False)
    return out_path


# ---- Check Stats ----

def _latest_season(df: pd.DataFrame) -> int | None:
    for scol in ["season","season_year","year","yr"]:
        if scol in df.columns:
            # coerce to numeric, drop NaNs, take max
            s = pd.to_numeric(df[scol], errors="coerce").dropna()
            if not s.empty:
                return int(s.max())
    return None

def _available_weeks(df: pd.DataFrame) -> list[int]:
    for wkcol in ["week","wk","game","g"]:
        if wkcol in df.columns:
            wk = pd.to_numeric(df[wkcol], errors="coerce").dropna().astype(int).unique().tolist()
            wk.sort()
            return wk
    return []

def _is_rate_like(stat_name: str) -> bool:
    n = stat_name.lower()
    return any(h in n for h in RATE_HINTS)

def _aggregate(df: pd.DataFrame, stat: str, week: str | int) -> tuple[float | int | None, dict]:
    """
    Return value and context dict. Uses most-recent season if a season column exists.
    - week == 'all' -> aggregate entire season for that team
    - week == int   -> pick that week (single row); if multiple rows exist, aggregate across those rows
    """
    ctx = {}

    # season filter -> latest
    s = _latest_season(df)
    if s is not None and "season" in df.columns:
        df = df[df["season"] == s]
        ctx["season"] = s
    elif s is not None and "season_year" in df.columns:
        df = df[df["season_year"] == s]
        ctx["season"] = s
    elif s is not None:
        ctx["season"] = s

    # locate week column (if any)
    wkcol = next((c for c in ["week","wk","game","g"] if c in df.columns), None)

    if week == "all" or wkcol is None:
        # season total/average
        if _is_rate_like(stat):
            val = pd.to_numeric(df[stat], errors="coerce").mean()
            ctx["agg"] = "mean"
        else:
            val = pd.to_numeric(df[stat], errors="coerce").sum()
            ctx["agg"] = "sum"
        return (None if pd.isna(val) else val), ctx

    # specific week
    dfw = df[pd.to_numeric(df[wkcol], errors="coerce") == int(week)]
    ctx["week"] = int(week)
    if dfw.empty:
        return None, ctx

    # if multiple rows (rare), aggregate same way
    if _is_rate_like(stat):
        val = pd.to_numeric(dfw[stat], errors="coerce").mean()
        ctx["agg"] = "mean"
    else:
        val = pd.to_numeric(dfw[stat], errors="coerce").sum()
        ctx["agg"] = "sum"
    return (None if pd.isna(val) else val), ctx

def _prompt_week(weeks: list[int]) -> str | int:
    if not weeks:
        # no week column; treat as "all"
        print("No week field found; using ALL (season aggregate).")
        return "all"

    # Show min..max and allow 'all'
    lo, hi = weeks[0], weeks[-1]
    while True:
        s = input(f"Week (number {lo}-{hi} or 'all' for season total): ").strip().lower()
        if s == "all":
            return "all"
        if s.isdigit():
            w = int(s)
            if lo <= w <= hi:
                return w
        print("Invalid input. Try again.")

def _read_team_stats_from_zip(team: str) -> "pd.DataFrame | pl.DataFrame":
    """Read team_stats.csv from data/TEAM.zip. Returns pandas DF (even if source was polars)."""
    zip_path = TEAM_DIR / f"{team}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}. Run: python app.py load --season 2025")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # prefer team_stats.csv; fallback if user renamed
        name = None
        for cand in ["team_stats.csv", "teamstats.csv", "teams.csv"]:
            if cand in zf.namelist():
                name = cand
                break
        if name is None:
            raise FileNotFoundError(f"{zip_path} has no team_stats.csv")
        data = zf.read(name)
    # Standardize to pandas for easy numeric detection & printing
    return pd.read_csv(io.BytesIO(data))

def _most_recent_row(df: pd.DataFrame) -> pd.Series:
    """Pick the most recent row using week/date/season heuristics."""
    # 1) week
    for wkcol in ["week","wk","game","g"]:
        if wkcol in df.columns:
            return df.sort_values(wkcol, kind="stable").iloc[-1]
    # 2) date-like
    for dcol in ["date","gamedate","game_date","datetime","timestamp"]:
        if dcol in df.columns:
            try:
                dts = pd.to_datetime(df[dcol], errors="coerce")
                return df.loc[dts.sort_values(kind="stable").index][-1]
            except Exception:
                pass
    # 3) season if present
    for scol in ["season","season_year","year","yr"]:
        if scol in df.columns:
            return df.sort_values(scol, kind="stable").iloc[-1]
    # 4) fallback last row
    return df.iloc[-1]

def _list_numeric_stats(df: pd.DataFrame) -> list[str]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # filter out common ID/label/time columns
    return [c for c in num_cols if c.lower() not in STAT_EXCLUDE_COLS]

def _print_3col_menu(items: list[str]):
    if not items:
        print("No numeric stats found in team_stats for this team.")
        return
    w = max(len(s) for s in items) + 2
    n = len(items)
    rows = (n + 2) // 3  # ceil split across 3 columns
    cols = [items[i*rows:(i+1)*rows] for i in range(3)]
    # normalize column lengths
    for i in range(3):
        while len(cols[i]) < rows:
            cols[i].append("")
    # print with indices
    for r in range(rows):
        line = []
        for c in range(3):
            idx = r + c*rows
            if idx < n:
                label = f"{idx+1:>2}. {cols[c][r]}"
            else:
                label = ""
            line.append(label.ljust(w))
        print("".join(line))

def _prompt_team(existing: list[str]) -> str:
    while True:
        t = input("Team (abbr, e.g., PHI): ").strip().upper()
        if t in existing:
            return t
        print(f"Invalid team. Choose one of: {', '.join(existing)}")

def _prompt_stat_choice(stats: list[str]) -> str:
    while True:
        s = input("Pick stat (name or number): ").strip()
        if s.isdigit():
            i = int(s) - 1
            if 0 <= i < len(stats):
                return stats[i]
        # allow case-insensitive name match
        matches = [c for c in stats if c.lower() == s.lower()]
        if matches:
            return matches[0]
        print("Not a valid choice. Try again.")

def query_stat(team: str | None, stat: str | None):
    # parameter check / interactive fallback
    existing = [abbr for abbr in TEAM_ABBRS if (TEAM_DIR / f"{abbr}.zip").exists()]
    if not existing:
        print("No team bundles found in ./data. Run: python app.py load --season 2025")
        return

    if team is None:
        print("Available teams with data:", ", ".join(existing))
        team = _prompt_team(existing)
    elif team not in existing:
        print(f"No bundle for {team}. Available: {', '.join(existing)}")
        return

    df = _read_team_stats_from_zip(team)

    stats = _list_numeric_stats(df)
    if not stats:
        print("No numeric stats available in that file.")
        return

    if stat is None:
        print("\nAvailable stats (numeric):")
        _print_3col_menu(stats)
        stat = _prompt_stat_choice(stats)
    else:
        # best-effort normalize stat name
        cand = [c for c in stats if c.lower() == stat.lower()]
        if not cand:
            print(f"Stat '{stat}' not found. Choose one of the listed stats:")
            _print_3col_menu(stats)
            stat = _prompt_stat_choice(stats)
        else:
            stat = cand[0]

    # week selection (number or 'all')
    weeks = _available_weeks(df)
    choice = _prompt_week(weeks)

    # compute aggregated/specific value
    val, ctx = _aggregate(df, stat, choice)

    # nice int if float is integral
    if isinstance(val, float) and val is not None and abs(val - round(val)) < 1e-9:
        val = int(round(val))

    # build context string
    bits = [team, "—", stat]
    if "season" in ctx:
        if isinstance(choice, int):
            bits.append(f"(season {ctx['season']}, week {choice})")
        else:
            bits.append(f"(season {ctx['season']})")
    elif isinstance(choice, int):
        bits.append(f"(week {choice})")
    if choice == "all":
        bits.append(f"[{ctx.get('agg','sum')}]")

    print("\n" + " ".join(bits) + f": {val if val is not None else 'NA'}")


# ---- Helpers ----
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _is_polars(df) -> bool:
    return (pl is not None) and (df.__class__.__module__.split('.')[0] == "polars")

def _empty_like(df):
    # 0-row copy, preserving schema, for both pandas and polars
    if _is_polars(df):
        return df.head(0)
    return df.iloc[0:0].copy()

def _csv_bytes(df) -> bytes:
    """
    Write a DataFrame (pandas or polars) to CSV bytes.
    """
    if _is_polars(df):
        # Convert to pandas for simplicity/compatibility
        df = df.to_pandas()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def _norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()

def _filter_df_by_team(df, team: str, candidates: List[str]):
    """
    Return rows where ANY candidate column matches ANY alias for `team`.
    Works with pandas or polars. Case-insensitive; matches abbr or full name.
    """
    aliases = TEAM_ALIASES.get(team, [team])
    aliases_norm = {_norm_str(a) for a in aliases}

    present = [c for c in candidates if c in (df.columns if not _is_polars(df) else df.columns)]
    if not present:
        return _empty_like(df)

    if _is_polars(df):
        # Build OR expression across columns and aliases
        expr = None
        for col in present:
            # Normalize column to uppercase string, trim, and fill nulls
            base = (
                pl.col(col)
                  .cast(pl.Utf8, strict=False)
                  .fill_null("")                 # handle nulls
                  .str.to_uppercase()
                  .str.strip_chars()             # <-- correct polars trim
            )
            col_expr = None
            for a in aliases_norm:
                e = (base == pl.lit(a))
                col_expr = e if col_expr is None else (col_expr | e)
            expr = col_expr if expr is None else (expr | col_expr)
        return df.filter(expr)

    # pandas path
    mask = None
    for col in present:
        col_vals = (
            df[col]
              .astype("string")
              .fillna("")
              .str.upper()
              .str.strip()
        )
        m = col_vals.isin(aliases_norm)
        mask = m if mask is None else (mask | m)
    return df.loc[mask].copy()



def _save_team_bundle(team: str, pieces: Dict[str, pd.DataFrame], out_dir: Path):
    """
    Create data/TEAM.zip containing:
      - pbp_offense.csv (posteam == TEAM)
      - pbp_defense.csv (defteam == TEAM)
      - player_stats.csv (player team == TEAM)
      - team_stats.csv (team == TEAM)
    """
    ensure_dir(out_dir)
    zip_path = out_dir / f"{team}.zip"

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in pieces.items():
            zf.writestr(f"{name}.csv", _csv_bytes(df))

    print(f"[OK] Wrote {zip_path}")


def _load_pbp_for_season(season: int) -> pd.DataFrame:
    # nflreadpy APIs can evolve; try season-specific first, then fallback.
    try:
        # many nflverse loaders accept seasons=[...]
        return nfl.load_pbp(seasons=[season])
    except TypeError:
        # older signature
        return nfl.load_pbp()


def _load_player_stats_for_season(season: int) -> pd.DataFrame:
    try:
        return nfl.load_player_stats([season])
    except TypeError:
        return nfl.load_player_stats(season)


def _load_team_stats_for_season(season: int) -> pd.DataFrame:
    """
    Some versions expose `load_team_stats(seasons=...)`, others `seasons=True`
    (meaning "all"). We’ll try season-specific first, then a broad load and filter.
    """
    try:
        return nfl.load_team_stats(seasons=[season])
    except Exception:
        try:
            df = nfl.load_team_stats(seasons=True)
            # Attempt to filter to the season if a season col exists
            season_col = _first_existing_col(df, ["season", "season_year", "yr", "year"])
            if season_col and season_col in df.columns:
                return df.loc[df[season_col] == season].copy()
            return df
        except Exception:
            # Last resort: older shapes
            return nfl.load_team_stats()


# ---- Rank Stats ----
def rank_stat(stat: str | None, week: str | int | None):
    # find teams we actually have data for
    existing = [abbr for abbr in TEAM_ABBRS if (TEAM_DIR / f"{abbr}.zip").exists()]
    if not existing:
        print("No team bundles found in ./data. Run: python app.py load --season 2025")
        return

    # Use a representative team to list numeric stats
    rep_team = existing[0]
    rep_df = _read_team_stats_from_zip(rep_team)
    stats = _list_numeric_stats(rep_df)
    if not stats:
        print("No numeric stats available in team_stats.")
        return

    # choose stat (prompt if missing / normalize if provided)
    if stat is None:
        print("\nAvailable stats (numeric):")
        _print_3col_menu(stats)
        stat = _prompt_stat_choice(stats)
    else:
        cand = [c for c in stats if c.lower() == stat.lower()]
        if not cand:
            print(f"Stat '{stat}' not found in {rep_team}. Choose one of the listed stats:")
            _print_3col_menu(stats)
            stat = _prompt_stat_choice(stats)
        else:
            stat = cand[0]

    # choose week or 'all' (prompt once using union of weeks across teams)
    if week is None:
        weeks_union: set[int] = set()
        for t in existing:
            try:
                weeks_union |= set(_available_weeks(_read_team_stats_from_zip(t)))
            except Exception:
                pass
        weeks_sorted = sorted(weeks_union)
        choice = _prompt_week(weeks_sorted)
    else:
        if isinstance(week, str):
            w = week.strip().lower()
            if w == "all":
                choice = "all"
            elif w.isdigit():
                choice = int(w)
            else:
                choice = "all"
        else:
            choice = int(week)

    # compute values per team
    results: list[tuple[str, float | int | None]] = []
    for team in existing:
        df = _read_team_stats_from_zip(team)
        if stat not in df.columns:
            results.append((team, None))
            continue
        val, _ctx = _aggregate(df, stat, choice)
        results.append((team, val))

    # keep only teams with a value
    present = [(t, v) for t, v in results if v is not None]
    # sort descending by default
    present.sort(key=lambda x: x[1], reverse=True)

    # pretty print
    season = _latest_season(rep_df)
    hdr = f"\nRanking — {stat}"
    if season is not None:
        hdr += f" (season {season})"
    if choice == "all":
        hdr += " [all]"
    elif isinstance(choice, int):
        hdr += f" [week {choice}]"
    print(hdr)
    print("-" * max(len(hdr), 30))

    max_team = max((len(t) for t, _ in present), default=4)
    print(f"{'RK':>3}  {'TEAM':<{max_team}}  VALUE")
    for i, (t, v) in enumerate(present, 1):
        if isinstance(v, float) and abs(v - round(v)) < 1e-9:
            v = int(round(v))
        print(f"{i:>3}  {t:<{max_team}}  {v}")

    missing = [t for t, v in results if v is None]
    if missing:
        print("\n(no data for: " + ", ".join(missing) + ")")
        

# ---- Core workflow ----
def load_data(season: int):
    print(f"Loading data for season {season}...")
    ensure_dir(DATA_DIR)

    # Pull datasets
    print("  • Loading play-by-play ...")
    pbp = _load_pbp_for_season(season)
    print(f"    PBP rows: {len(pbp):,}")

    print("  • Loading player stats ...")
    player_stats = _load_player_stats_for_season(season)
    print(f"    Player stat rows: {len(player_stats):,}")

    print("  • Loading team stats ...")
    team_stats = _load_team_stats_for_season(season)
    print(f"    Team stat rows: {len(team_stats):,}")
    
        # Export schedules for Predict page
    try:
        sched_path = _export_schedule_csv(season)
        print(f"[OK] Wrote {sched_path}")
    except Exception as e:
        print(f"[WARN] Could not write schedules for {season}: {e}")


    # Identify columns we might need
    # PBP commonly: posteam, defteam
    # Player stats commonly: team, recent_team, team_abbr, team_short
    # Team stats commonly: team, team_abbr
    for team in TEAM_ABBRS:
        print(f"  → Slicing for {team} ...")

        pbp_off = _filter_df_by_team(pbp, team, ["posteam", "posteam_abbr", "OffenseTeam", "offense_team"])
        pbp_def = _filter_df_by_team(pbp, team, ["defteam", "defteam_abbr", "DefenseTeam", "defense_team"])

        plr = _filter_df_by_team(player_stats, team, ["team", "recent_team", "team_abbr", "team_short", "Tm"])

        tms = _filter_df_by_team(
            team_stats,
            team,
            ["team", "team_abbr", "Tm", "team_name", "name", "club_code"]
        )


        pieces = {
            "pbp_offense": pbp_off,
            "pbp_defense": pbp_def,
            "player_stats": plr,
            "team_stats": tms,
        }

        _save_team_bundle(team, pieces, TEAM_DIR)

    print("\nAll teams processed. Files are in ./data as TEAM.zip bundles.")


def main():
    parser = argparse.ArgumentParser(description="AI NFL game predictor data tools.")
    sub = parser.add_subparsers(dest="cmd")

    # existing 'load'
    p_load = sub.add_parser("load", help="Download & slice data into ./data/TEAM.zip files")
    p_load.add_argument("--season", type=int, default=2025, help="Season year (default: 2025)")

    # existing 'stat'
    p_stat = sub.add_parser("stat", help="Look up a team stat from data/TEAM.zip")
    p_stat.add_argument("--team", type=str, help="Team abbr (e.g., PHI)")
    p_stat.add_argument("--stat", type=str, help="Stat column name (case-insensitive)")

    # NEW 'rank'
    p_rank = sub.add_parser("rank", help="Rank teams by a stat from data/TEAM.zip")
    p_rank.add_argument("--stat", type=str, help="Stat column name (case-insensitive)")
    p_rank.add_argument("--week", type=str, help="Week number or 'all' (prompt if omitted)")

    args = parser.parse_args()

    if args.cmd == "load":
        load_data(args.season)
    elif args.cmd == "stat":
        query_stat(args.team, args.stat)
    elif args.cmd == "rank":
        rank_stat(args.stat, args.week)
    else:
        print("Hello")
        print("Try:\n"
              "  python app.py load --season 2025\n"
              "  python app.py stat --team PHI\n"
              "  python app.py rank --stat completions --week all")




if __name__ == "__main__":
    main()
