#!/usr/bin/env python3
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import io, csv, datetime as dt
from predictor.model import predict_week
from predictor.refresh_schedules import refresh_schedule
from predictor.features import build_and_save_features
from predictor.train import train as train_model


# Import your existing code (the big CLI file). Rename it to `app_core.py`
# or adjust this import to match your filename.
from app_core import (
    TEAM_ABBRS, TEAM_DIR, DATA_DIR,
    _read_team_stats_from_zip, _list_numeric_stats, _available_weeks,
    _aggregate, _latest_season, load_data, ensure_dir
)

MODEL_DIR = "predictor/saved_models/current"

def _promote_to_current(model_dir: str):
    """Point predictor/saved_models/current to the newest trained model."""
    from pathlib import Path
    import os
    cur = Path("predictor/saved_models/current")
    try:
        if cur.exists() or cur.is_symlink():
            cur.unlink()
        os.symlink(Path(model_dir).resolve(), cur)
        return True, f"Promoted {model_dir} → {cur}"
    except Exception as e:
        return False, f"Could not promote model: {e}"

# Schedules are expected at: DATA_DIR / f"schedules_{season}.csv"
SCHEDULES_NAME = "schedules_{season}.csv"

app = Flask(__name__)
app.secret_key = "change-me"  # only used for flash messages


def _load_schedule(season: int) -> pd.DataFrame | None:
    path = (DATA_DIR / SCHEDULES_NAME.format(season=season))
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # Normalize column names we’ll try to use
        lower = {c: c.lower() for c in df.columns}
        df.rename(columns=lower, inplace=True)
        # common expected names: season, week, home_team, away_team, kickoff_datetime/kickoff
        return df
    except Exception:
        return None
    
def _baseline_home_win_prob(home: str, away: str, season: int, week: int) -> tuple[float | None, float | None, str]:
    """
    Very lightweight placeholder:
    - pulls 'points_for' and 'points_against' (if present) from team_stats up through week-1
    - computes a crude margin = (PF-PA)_home - (PF-PA)_away + home_bump
    - converts to probability with a sigmoid scale
    Returns: (home_win_prob, pred_margin, note)
    """
    # Read both team stats
    try:
        hdf = _read_team_stats_from_zip(home)
        adf = _read_team_stats_from_zip(away)
    except Exception:
        return None, None, "Missing team_stats for one side."

    # choose stat column names if they exist
    def pick(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None

    pf_cols = ["points_for", "points", "pf", "points_scored"]
    pa_cols = ["points_against", "pa", "points_allowed"]

    # filter to latest season and up to week-1
    def up_to_week(df):
        # latest season
        s = _latest_season(df)
        if s is not None:
            for scol in ["season","season_year","year","yr"]:
                if scol in df.columns:
                    df = df[pd.to_numeric(df[scol], errors="coerce") == s]
                    break
        wkcol = next((c for c in ["week","wk","game","g"] if c in df.columns), None)
        if wkcol:
            df = df[pd.to_numeric(df[wkcol], errors="coerce") <= (week - 1)]
        return df

    hdf = up_to_week(hdf)
    adf = up_to_week(adf)

    h_pf = pick(hdf, pf_cols); h_pa = pick(hdf, pa_cols)
    a_pf = pick(adf, pf_cols); a_pa = pick(adf, pa_cols)

    if not all([h_pf, h_pa, a_pf, a_pa]) or hdf.empty or adf.empty:
        return None, None, "Insufficient columns for baseline; showing placeholders."

    # aggregate to date
    h_diff = pd.to_numeric(hdf[h_pf], errors="coerce").sum() - pd.to_numeric(hdf[h_pa], errors="coerce").sum()
    a_diff = pd.to_numeric(adf[a_pf], errors="coerce").sum() - pd.to_numeric(adf[a_pa], errors="coerce").sum()

    margin = (h_diff - a_diff) * 0.08 + 2.0  # weak scale + small home bump
    # probability via logistic
    prob = 1 / (1 + pow(2.718281828, -margin / 6.5))

    return float(prob), float(margin), "Baseline = (PF-PA)_diff + home bump, logistic scaled."

# at top of web/app.py (near other imports)
from predictor.model import predict_week
MODELS_DIR = Path("predictor/saved_models")
MODEL_CURRENT = MODELS_DIR / "current"

def list_models() -> list[str]:
    """All concrete model folders (exclude 'current' symlink/dir)."""
    items = []
    for d in MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        if d.name == "current":
            continue
        # include only folders that look like trained models
        if (d / "meta.json").exists() and (d / "model.joblib").exists():
            items.append(d.name)
    # sort newest first by mtime
    items.sort(key=lambda n: (MODELS_DIR / n).stat().st_mtime, reverse=True)
    return items

def _get_current_model_dir() -> Path | None:
    if MODEL_CURRENT.exists():
        try:
            return MODEL_CURRENT.resolve()
        except Exception:
            return MODEL_CURRENT
    # fallback: newest concrete model
    names = list_models()
    return (MODELS_DIR / names[0]) if names else None

def _get_model_label_from_dir(p: Path | None) -> str:
    if not p:
        return "No model"
    meta_path = p / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            return meta.get("name") or p.name
        except Exception:
            pass
    return p.name

@app.route("/predict", methods=["GET", "POST"])
def predict_view():
    # defaults
    season = 2025
    week = 1
    explain = False
    games = None
    note = None

    # model list + default selection (current)
    models = list_models()
    current_dir = _get_current_model_dir()
    selected_model = None  # folder name chosen in the form

    if request.method == "POST":
        # form values
        try:
            season = int(request.form.get("season", season))
        except Exception:
            pass
        try:
            week = int(request.form.get("week", week))
        except Exception:
            pass
        explain = request.form.get("explain") == "on"
        selected_model = request.form.get("model") or None  # folder name

        # load schedule CSV
        sched = _load_schedule(season)
        if sched is None:
            note = f"Schedule file not found: {SCHEDULES_NAME.format(season=season)} in {DATA_DIR}."
            games = []
        else:
            # resolve columns present in your schedules file
            home_col = next((c for c in ["home_team","home","homeabbr","home_abbr"] if c in sched.columns), None)
            away_col = next((c for c in ["away_team","away","awayabbr","away_abbr"] if c in sched.columns), None)
            ko_col   = next((c for c in ["kickoff_datetime","kickoff","game_date","datetime","start_time"] if c in sched.columns), None)
            week_col = next((c for c in ["week","wk","gameweek"] if c in sched.columns), None)

            dfw = sched.copy()
            if week_col is not None:
                dfw = dfw[pd.to_numeric(dfw[week_col], errors="coerce") == week].copy()

            # normalize team strings
            for col in (home_col, away_col):
                if col:
                    dfw[col] = dfw[col].astype("string").str.upper().str.strip()

            # choose model path: selected folder OR current
            if selected_model:
                model_dir = (MODELS_DIR / selected_model)
            else:
                model_dir = current_dir

            pred_df = None
            model_label = _get_model_label_from_dir(model_dir)
            try:
                if model_dir is None or not model_dir.exists():
                    raise FileNotFoundError("No model chosen and no 'current' model available.")
                pred_df = predict_week(model_dir, season, week)
                # normalize columns for join safety
                if "home_team" in pred_df.columns:
                    pred_df["home_team"] = pred_df["home_team"].astype(str).str.upper().str.strip()
                if "away_team" in pred_df.columns:
                    pred_df["away_team"] = pred_df["away_team"].astype(str).str.upper().str.strip()
            except Exception as e:
                note = f"Could not run model ({model_label}): {e}"

            rows = []
            for _, r in dfw.iterrows():
                home = str(r.get(home_col, "")).strip().upper() if home_col else ""
                away = str(r.get(away_col, "")).strip().upper() if away_col else ""
                kickoff = r.get(ko_col, None)
                kickoff_str = str(kickoff) if pd.notna(kickoff) else None

                prob = None
                margin = None
                rationale = None

                if pred_df is not None:
                    m = pred_df[
                        (pred_df.get("week", week) == week) &
                        (pred_df["home_team"] == home) &
                        (pred_df["away_team"] == away)
                    ]
                    if not m.empty and "home_win_prob" in m.columns:
                        prob = float(m["home_win_prob"].iloc[0])
                        if "pred_margin" in m.columns and pd.notna(m["pred_margin"].iloc[0]):
                            margin = float(m["pred_margin"].iloc[0])
                        rationale = f"Model: {model_label}"
                    else:
                        rationale = f"No prediction for this matchup (Model: {model_label})."

                if explain and rationale and prob is not None:
                    # (future) add LLM explanation here
                    pass

                rows.append({
                    "home": home or "—",
                    "away": away or "—",
                    "kickoff": kickoff_str,
                    "home_win_pct": prob,
                    "pred_margin": margin,
                    "explain": rationale if explain else None
                })

            games = rows

            # CSV export
            if request.form.get("export") == "csv" and games is not None:
                output = io.StringIO()
                w = csv.writer(output)
                w.writerow(["season","week","home","away","kickoff","home_win_prob","pred_margin"])
                for g in games:
                    w.writerow([season, week, g["home"], g["away"], g["kickoff"], g["home_win_pct"], g["pred_margin"]])
                mem = io.BytesIO(output.getvalue().encode("utf-8"))
                fname = f"predictions_{season}_week_{week:02d}.csv"
                return send_file(mem, as_attachment=True, download_name=fname, mimetype="text/csv")

    # final label to show in the UI (selected or current)
    model_dir_for_label = (MODELS_DIR / selected_model) if selected_model else current_dir
    model_label = _get_model_label_from_dir(model_dir_for_label)

    return render_template(
        "predict.html",
        season=season,
        week=week,
        games=games,
        note=note,
        models=models,
        selected_model=selected_model,
        model_label=model_label,
    )

    

# ---- Helpers for the web layer ----
def _first_team_with_data():
    """Pick any team that has a data zip present."""
    for abbr in TEAM_ABBRS:
        if (TEAM_DIR / f"{abbr}.zip").exists():
            return abbr
    return None

def _all_teams_with_data():
    return [abbr for abbr in TEAM_ABBRS if (TEAM_DIR / f"{abbr}.zip").exists()]

def _get_stat_choices():
    """Return a list of numeric stat columns using a representative team."""
    rep = _first_team_with_data()
    if not rep:
        return []
    df = _read_team_stats_from_zip(rep)
    return _list_numeric_stats(df)

def _get_week_choices():
    """Union of available weeks across teams (for nicer dropdown)."""
    weeks = set()
    for t in _all_teams_with_data():
        try:
            df = _read_team_stats_from_zip(t)
            weeks |= set(_available_weeks(df))
        except Exception:
            pass
    weeks = sorted(weeks)
    return weeks

# ---- Routes ----
@app.route("/")
def index():
    has_data = len(_all_teams_with_data()) > 0
    return render_template("index.html", has_data=has_data)

@app.route("/load", methods=["GET", "POST"])
def load_view():
    if request.method == "POST":
        try:
            season = int(request.form.get("season", "2025"))
        except ValueError:
            season = 2025
        ensure_dir(DATA_DIR)
        load_data(season)
        flash(f"Loaded data for season {season}.", "success")
        return redirect(url_for("index"))
    return render_template("load.html")

@app.route("/train", methods=["GET", "POST"])
def train_view():
    note = None
    ok = False
    # sensible defaults
    season_default = 2025
    through_week_default = 3

    if request.method == "POST":
        # read inputs
        try:
            season = int(request.form.get("season", season_default))
        except Exception:
            season = season_default
        try:
            through_week = int(request.form.get("through_week", through_week_default))
        except Exception:
            through_week = through_week_default

        # optional model name; if blank, we generate one
        name = request.form.get("name", "").strip()
        if not name:
            name = f"lr_{season}_w{through_week:02d}"

        try:
            # 1) refresh schedules (ensures scores present)
            refresh_schedule(season)

            # 2) build features from schedules (no leakage)
            build_and_save_features(season)

            # 3) train model
            out_dir = train_model(season, through_week, name)  # returns path string

            # 4) promote to "current" so /predict always uses the latest model
            promoted, msg = _promote_to_current(out_dir)

            ok = True
            note = f"✅ Trained {name} for season {season} through week {through_week}. " + (msg if promoted else "")
        except Exception as e:
            note = f"❌ Training failed: {e}"

        return render_template("train.html",
                               note=note, ok=ok,
                               season=season, through_week=through_week, name=name)

    # GET
    return render_template("train.html",
                           note=note, ok=ok,
                           season=season_default,
                           through_week=through_week_default,
                           name=f"lr_{season_default}_w{through_week_default:02d}")


@app.route("/stat", methods=["GET", "POST"])
def stat_view():
    teams = _all_teams_with_data()
    stats = _get_stat_choices()
    weeks = _get_week_choices()

    # If no data yet, show a friendly message.
    if not teams:
        flash("No data found. Go to Load Data to fetch a season first.", "warning")
        return redirect(url_for("index"))

    result = None
    context_bits = None

    if request.method == "POST":
        team = request.form.get("team")
        stat = request.form.get("stat")
        week = request.form.get("week")  # "all" or "N"

        if week and week.isdigit():
            week_choice = int(week)
        else:
            week_choice = "all"

        try:
            df = _read_team_stats_from_zip(team)
            val, ctx = _aggregate(df, stat, week_choice)
            # pretty int if float is integral
            if isinstance(val, float) and val is not None and abs(val - round(val)) < 1e-9:
                val = int(round(val))

            context_bits = []
            if "season" in ctx:
                if isinstance(week_choice, int):
                    context_bits.append(f"season {ctx['season']}, week {week_choice}")
                else:
                    context_bits.append(f"season {ctx['season']}")
            elif isinstance(week_choice, int):
                context_bits.append(f"week {week_choice}")
            if week_choice == "all":
                context_bits.append(f"[{ctx.get('agg','sum')}]")

            result = {
                "team": team,
                "stat": stat,
                "value": "NA" if val is None else val,
            }
        except FileNotFoundError:
            flash("Required data not found. Load data first.", "danger")
        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template(
        "stat.html",
        teams=teams,
        stats=stats,
        weeks=weeks,
        result=result,
        context_bits=context_bits
    )

@app.route("/rank", methods=["GET", "POST"])
def rank_view():
    teams = _all_teams_with_data()
    stats = _get_stat_choices()
    weeks = _get_week_choices()

    if not teams:
        flash("No data found. Go to Load Data to fetch a season first.", "warning")
        return redirect(url_for("index"))

    table = None
    hdr = None

    if request.method == "POST":
        stat = request.form.get("stat")
        week = request.form.get("week")
        choice = "all" if week != None and week.lower() == "all" else (int(week) if week and week.isdigit() else "all")

        rows = []
        for t in teams:
            try:
                df = _read_team_stats_from_zip(t)
                if stat not in df.columns:
                    rows.append((t, None))
                    continue
                val, ctx = _aggregate(df, stat, choice)
                rows.append((t, val))
            except Exception:
                rows.append((t, None))

        present = [(t, v) for (t, v) in rows if v is not None]
        present.sort(key=lambda x: x[1], reverse=True)

        # pretty ints
        pretty_rows = []
        for rk, (t, v) in enumerate(present, 1):
            if isinstance(v, float) and abs(v - round(v)) < 1e-9:
                v = int(round(v))
            pretty_rows.append({"rk": rk, "team": t, "value": v})

        rep = _first_team_with_data()
        season = None
        if rep:
            season = _latest_season(_read_team_stats_from_zip(rep))

        hdr = f"{stat}"
        if season is not None:
            hdr += f" — season {season}"
        if choice == "all":
            hdr += " (ALL)"
        elif isinstance(choice, int):
            hdr += f" (week {choice})"

        table = pretty_rows

    return render_template(
        "rank.html",
        stats=stats,
        weeks=weeks,
        table=table,
        header_text=hdr
    )

if __name__ == "__main__":
    # Run: python3 web/app.py
    app.run(host="10.8.0.1", port=5000, debug=True)
