"""
tools.py

All analysis and prediction functions backing the IPL multi-agent system.

Architecture:
Support Agent
    -> Data Agent
    -> ML Agent
    -> RAG Agent

This file contains ONLY backend analysis / ML functions. Agent definitions
(tool wrappers, routing) live in data_agent.py, ml_agent.py, rag_agent.py,
and router.py.
"""

import difflib
import re
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import shap


# ============================================================
# PROJECT PATHS
# ============================================================

# tools.py lives directly in the project root (same folder as app.py) in
# this project's layout, so PROJECT_ROOT is just tools.py's own folder.
# If you move tools.py into a subfolder (e.g. src/tools.py), change this to
# Path(__file__).resolve().parent.parent instead.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MATCH_INFO_PATH = DATA_DIR / "match_info.csv"
ALL_MATCHES_PATH = DATA_DIR / "all_matches_clean.csv"
SCORE_FEATURES_PATH = DATA_DIR / "score_features.csv"
MODEL_PATH = MODEL_DIR / "score_model.pkl"


# ============================================================
# CHECK REQUIRED FILES (fail fast with a clear message, not a
# confusing pandas/joblib error deep inside some later function call)
# ============================================================

required_files = {
    "match_info.csv": MATCH_INFO_PATH,
    "all_matches_clean.csv": ALL_MATCHES_PATH,
    "score_features.csv": SCORE_FEATURES_PATH,
    "score_model.pkl": MODEL_PATH,
}

for name, path in required_files.items():
    if not path.exists():
        raise FileNotFoundError(f"{name} not found.\nExpected location:\n{path}")


# ============================================================
# LOAD DATA
# ============================================================

match_info = pd.read_csv(MATCH_INFO_PATH)
all_matches = pd.read_csv(ALL_MATCHES_PATH)
score_data = pd.read_csv(SCORE_FEATURES_PATH)
score_model = joblib.load(MODEL_PATH)


# ============================================================
# DATA CLEANING / DERIVED COLUMNS
# ============================================================

all_matches["season"] = all_matches["season"].astype(str)
all_matches["runs_off_bat"] = pd.to_numeric(all_matches["runs_off_bat"], errors="coerce").fillna(0)
all_matches["extras"] = pd.to_numeric(all_matches["extras"], errors="coerce").fillna(0)

for col in ["wides", "noballs"]:
    if col not in all_matches.columns:
        all_matches[col] = np.nan

all_matches["total_runs"] = all_matches["runs_off_bat"] + all_matches["extras"]
all_matches["is_wicket"] = all_matches["wicket_type"].notnull().astype(int)
all_matches["is_six"] = ((all_matches["runs_off_bat"] == 6) & (all_matches["non_boundary"] != 1)).astype(int)
all_matches["is_four"] = ((all_matches["runs_off_bat"] == 4) & (all_matches["non_boundary"] != 1)).astype(int)

# bowler-specific derived columns (run outs don't count as bowler wickets;
# byes/leg-byes aren't charged to the bowler; wides/no-balls are)
all_matches["bowler_wicket"] = (
    (all_matches["is_wicket"] == 1) & (all_matches["wicket_type"] != "run out")
).astype(int)
all_matches["bowler_runs_conceded"] = (
    all_matches["runs_off_bat"] + all_matches["wides"].fillna(0) + all_matches["noballs"].fillna(0)
)
all_matches["legal_ball"] = all_matches["wides"].isnull() & all_matches["noballs"].isnull()

_missing_new_tool_columns = [
    c for c in ['match_id', 'season', 'striker', 'bowler', 'batting_team', 'bowling_team', 'innings', 'runs_off_bat', 'legal_ball', 'bowler_wicket', 'bowler_runs_conceded']
    if c not in all_matches.columns
]
if _missing_new_tool_columns:
    raise ValueError(
        "Missing columns required by IPL data tools: "
        f"{_missing_new_tool_columns}"
    )


# ============================================================
# SCORE MODEL
# ============================================================

FEATURE_COLS_SCORE = [
    "current_score", "current_wickets", "current_run_rate",
    "runs_last_24balls", "balls_remaining",
]

missing_features = [col for col in FEATURE_COLS_SCORE if col not in score_data.columns]
if missing_features:
    raise ValueError(f"Missing score model features: {missing_features}")

X_BACKGROUND = score_data[FEATURE_COLS_SCORE].sample(min(100, len(score_data)), random_state=42)


# ============================================================
# NAME RESOLUTION HELPERS
# ============================================================

def _resolve_team_name(name):
    """Best-effort match against the exact team names stored in the dataset.
    Handles renamed/variant franchise names (e.g. 'Royal Challengers Bangalore'
    -> 'Royal Challengers Bengaluru') via fuzzy string matching. Falls back to
    the original input unchanged if nothing close enough is found."""
    all_teams = pd.concat([match_info["team1"], match_info["team2"]]).dropna().unique()
    if name in all_teams:
        return name
    close = difflib.get_close_matches(name, list(all_teams), n=1, cutoff=0.6)
    return close[0] if close else name


def _resolve_player_name(player):
    """Resolve a player name against both striker and bowler names.

    Dataset names are commonly stored as ``Initial Surname`` (for example
    ``V Kohli`` or ``TA Boult``). Resolution order is deliberately
    conservative:
      1. exact match;
      2. surname + first-initial match;
      3. unique surname/partial-name match;
      4. word-boundary-aware substring match.

    The resolver searches both batting and bowling names so a bowler who
    has little/no batting data can still be resolved.
    """
    striker_names = (
        all_matches["striker"].dropna().astype(str).unique().tolist()
    )
    bowler_names = (
        all_matches["bowler"].dropna().astype(str).unique().tolist()
    )
    players = sorted(set(striker_names + bowler_names))

    query_lower = str(player).strip().lower()
    if not query_lower:
        return None

    # 1. Exact match.
    for name in players:
        if name.lower() == query_lower:
            return name

    query_parts = query_lower.split()

    # 2. Full name -> dataset initial + surname.
    if query_parts:
        surname = query_parts[-1].rstrip(".")
        initial = query_parts[0][0]

        for name in players:
            parts = name.lower().replace(".", " ").split()
            if parts and parts[-1] == surname and parts[0][:1] == initial:
                return name

    # 3. Surname-only / unique token match.
    if len(query_parts) == 1:
        matches = [
            name for name in players
            if name.lower().split()[-1].rstrip(".") == query_lower
        ]
        if len(matches) == 1:
            return matches[0]

    # 4. Word-boundary-aware partial match.
    matches = []
    for name in players:
        name_lower = name.lower()
        if (
            re.search(r"\b" + re.escape(query_lower) + r"\b", name_lower)
            or re.search(r"\b" + re.escape(name_lower) + r"\b", query_lower)
        ):
            matches.append(name)

    if len(matches) == 1:
        return matches[0]

    return None


def resolve_player_name_for_agent(player):
    """Public wrapper used by data_agent.py for deterministic intent routing."""
    return _resolve_player_name(player)


# ============================================================
# DATA AGENT FUNCTIONS
# ============================================================

def team_win_percentage_data():
    """All teams' win percentage, sorted highest first."""
    results = []
    all_teams = pd.concat([match_info["team1"], match_info["team2"]]).dropna().unique()

    for team in all_teams:
        played = match_info[(match_info["team1"] == team) | (match_info["team2"] == team)]
        wins = int((played["winner"] == team).sum())
        total = int(len(played))
        results.append({
            "team": team, "matches": total, "wins": wins,
            "win_pct": round(wins / total * 100, 1) if total else 0.0
        })

    return sorted(results, key=lambda x: x["win_pct"], reverse=True)


def head_to_head_data(team_a, team_b):
    """Head-to-head record between two named teams. Team names are fuzzy-resolved
    against the dataset's exact naming first (handles renamed franchises)."""
    team_a = _resolve_team_name(team_a)
    team_b = _resolve_team_name(team_b)

    matches = match_info[
        ((match_info["team1"] == team_a) & (match_info["team2"] == team_b)) |
        ((match_info["team1"] == team_b) & (match_info["team2"] == team_a))
    ]

    return {
        "team_a": team_a, "team_b": team_b,
        "total_matches": int(len(matches)),
        "team_a_wins": int((matches["winner"] == team_a).sum()),
        "team_b_wins": int((matches["winner"] == team_b).sum()),
    }


def highest_team_score_data():
    """The highest TOTAL TEAM SCORE ever recorded in a single IPL innings.
    This is a team's combined score, NOT an individual player's runs —
    see top_run_scorers_data / player_highest_score_data for player-level totals."""
    main = all_matches[all_matches["innings"].isin([1, 2])]
    innings_totals = main.groupby(
        ["match_id", "season", "batting_team", "bowling_team"]
    )["total_runs"].sum().reset_index()

    if innings_totals.empty:
        return {"error": "No innings data available."}

    top = innings_totals.sort_values("total_runs", ascending=False).iloc[0]
    return {
        "team": top["batting_team"], "opponent": top["bowling_team"],
        "score": int(top["total_runs"]), "season": top["season"]
    }


def top_run_scorers_data(top_n=5):
    """Individual PLAYERS ranked by total career runs scored in IPL.
    This is a player's personal run total, NOT a single team innings score —
    see highest_team_score_data for the highest team total in one innings."""
    top_n = int(top_n)
    runs = all_matches.groupby("striker")["runs_off_bat"].sum().sort_values(ascending=False).head(top_n)
    return [{"player": player, "total_runs": int(r)} for player, r in runs.items()]


def player_highest_score_data(player):
    """A specific named player's highest individual score in a single IPL
    innings, with opponent and season. Distinct from top_run_scorers_data
    (career total) and highest_team_score_data (team total)."""
    resolved_name = _resolve_player_name(player)
    if resolved_name is None:
        return {"error": f"No batting data found for '{player}'. Check the exact spelling used in the dataset."}

    matches = all_matches[all_matches["striker"] == resolved_name]
    innings = matches.groupby(["match_id", "season", "bowling_team"])["runs_off_bat"].sum().reset_index()
    top = innings.sort_values("runs_off_bat", ascending=False).iloc[0]

    return {
        "player": resolved_name, "highest_score": int(top["runs_off_bat"]),
        "opponent": top["bowling_team"], "season": top["season"],
    }



def player_match_score_data(player, target_runs, target_balls):
    """Find IPL innings where a player scored exactly X runs from Y balls.

    This is intentionally separate from player_highest_score_data(), which
    answers a different question: the player's highest individual score.
    """
    resolved_name = _resolve_player_name(player)
    if resolved_name is None:
        return {
            "error": f"No IPL batting/bowling data found for '{player}'. Check the player name."
        }

    try:
        target_runs = int(target_runs)
        target_balls = int(target_balls)
    except (TypeError, ValueError):
        return {"error": "target_runs and target_balls must be integers."}

    if target_runs < 0 or target_balls < 0:
        return {"error": "target_runs and target_balls cannot be negative."}

    rows = all_matches[all_matches["striker"] == resolved_name].copy()
    if rows.empty:
        return {"error": f"No batting data found for '{resolved_name}'."}

    group_cols = [
        "match_id", "season", "batting_team", "bowling_team", "innings"
    ]

    innings_runs = (
        rows.groupby(group_cols, dropna=False)["runs_off_bat"]
        .sum()
        .reset_index(name="runs")
    )

    # For batter balls faced, wides and no-balls are not legal balls faced.
    innings_balls = (
        rows.groupby(group_cols, dropna=False)["legal_ball"]
        .sum()
        .reset_index(name="balls")
    )

    innings = innings_runs.merge(
        innings_balls,
        on=group_cols,
        how="left",
    )

    matches = innings[
        (innings["runs"] == target_runs)
        & (innings["balls"] == target_balls)
    ].copy()

    if matches.empty:
        return {
            "player": resolved_name,
            "runs": target_runs,
            "balls": target_balls,
            "found": False,
            "matches": [],
            "message": (
                f"No IPL innings found where {resolved_name} scored "
                f"{target_runs} runs from {target_balls} balls in the dataset."
            ),
        }

    results = []

    # Add match metadata when the columns are available. This keeps the tool
    # useful even if a particular processed dataset has fewer metadata fields.
    metadata = None
    if "match_id" in match_info.columns:
        metadata_cols = [
            c for c in ["match_id", "date", "venue", "city"]
            if c in match_info.columns
        ]
        if metadata_cols:
            metadata = match_info[metadata_cols].drop_duplicates("match_id")

    for _, row in matches.iterrows():
        item = {
            "match_id": row["match_id"],
            "season": str(row["season"]),
            "batting_team": row["batting_team"],
            "opponent": row["bowling_team"],
            "innings": int(row["innings"]) if pd.notna(row["innings"]) else None,
            "runs": int(row["runs"]),
            "balls": int(row["balls"]),
        }

        if metadata is not None:
            meta = metadata[metadata["match_id"].astype(str) == str(row["match_id"])]
            if not meta.empty:
                meta_row = meta.iloc[0]
                for col in ["date", "venue", "city"]:
                    if col in meta.columns and pd.notna(meta_row[col]):
                        item[col] = str(meta_row[col])

        results.append(item)

    return {
        "player": resolved_name,
        "runs": target_runs,
        "balls": target_balls,
        "found": True,
        "matches": results,
    }


def player_teams_data(player):
    """Return the IPL teams represented by a player in the dataset.

    Team membership is inferred from both batting_team (when the player
    batted) and bowling_team (when the player bowled).
    """
    resolved_name = _resolve_player_name(player)
    if resolved_name is None:
        return {"error": f"No IPL data found for '{player}'. Check the player name."}

    teams = set()

    batting_rows = all_matches[all_matches["striker"] == resolved_name]
    if not batting_rows.empty:
        teams.update(
            batting_rows["batting_team"].dropna().astype(str).unique().tolist()
        )

    bowling_rows = all_matches[all_matches["bowler"] == resolved_name]
    if not bowling_rows.empty:
        teams.update(
            bowling_rows["bowling_team"].dropna().astype(str).unique().tolist()
        )

    teams = sorted(teams)
    if not teams:
        return {"error": f"No team history could be determined for '{resolved_name}'."}

    return {
        "player": resolved_name,
        "team_count": len(teams),
        "teams": teams,
    }


def specific_bowler_stats_data(bowler):
    """Return aggregate IPL bowling statistics for one specific bowler."""
    resolved_name = _resolve_player_name(bowler)
    if resolved_name is None:
        return {"error": f"No IPL player found for '{bowler}'. Check the player name."}

    rows = all_matches[all_matches["bowler"] == resolved_name]
    if rows.empty:
        return {"error": f"No bowling data found for '{resolved_name}'."}

    wickets = int(rows["bowler_wicket"].sum())
    runs_conceded = int(rows["bowler_runs_conceded"].sum())
    legal_balls = int(rows["legal_ball"].sum())

    economy = (
        round(runs_conceded / (legal_balls / 6), 2)
        if legal_balls > 0
        else None
    )

    return {
        "bowler": resolved_name,
        "wickets": wickets,
        "runs_conceded": runs_conceded,
        "legal_balls": legal_balls,
        "economy": economy,
    }


def best_strike_rate_data(min_balls=500, top_n=5):
    """Highest career strike rate, qualified with a minimum balls-faced sample size."""
    stats = all_matches.groupby("striker").agg(
        runs=("runs_off_bat", "sum"), balls=("striker", "count")
    ).reset_index()

    stats = stats[stats["balls"] >= int(min_balls)].copy()
    if stats.empty:
        return []

    stats["strike_rate"] = (stats["runs"] / stats["balls"] * 100).round(2)
    top = stats.sort_values("strike_rate", ascending=False).head(int(top_n))
    return top[["striker", "runs", "balls", "strike_rate"]].to_dict(orient="records")


def bowler_stats_data(top_n=5, min_balls=300):
    """Bowlers with the most wickets, qualified with a minimum legal-balls-bowled sample."""
    grouped = all_matches.groupby("bowler").agg(
        wickets=("bowler_wicket", "sum"),
        runs_conceded=("bowler_runs_conceded", "sum"),
        legal_balls=("legal_ball", "sum")
    ).reset_index()

    grouped = grouped[grouped["legal_balls"] >= int(min_balls)].copy()
    if grouped.empty:
        return []

    grouped["economy"] = (grouped["runs_conceded"] / (grouped["legal_balls"] / 6)).round(2)
    top = grouped.sort_values("wickets", ascending=False).head(int(top_n))
    return top[["bowler", "wickets", "economy"]].to_dict(orient="records")


def venue_stats_data(top_n=5):
    """Venues with the highest average first/second-innings score.
    Groups by innings as well as match_id — grouping by match_id alone
    would sum BOTH innings of a match together, silently inflating the
    'average innings score' into something closer to a full-match total."""
    main = all_matches[all_matches["innings"].isin([1, 2])]
    innings_totals = main.groupby(["match_id", "venue", "innings"])["total_runs"].sum().reset_index()

    venue_avg = (
        innings_totals.groupby("venue")["total_runs"].mean()
        .sort_values(ascending=False).head(int(top_n))
    )

    return [{"venue": venue, "avg_score": round(score, 1)} for venue, score in venue_avg.items()]


def most_sixes_data(top_n=5):
    """Players with the most career sixes."""
    sixes = all_matches.groupby("striker")["is_six"].sum().sort_values(ascending=False).head(int(top_n))
    return [{"player": player, "sixes": int(count)} for player, count in sixes.items()]


def best_batting_average_data(min_innings=20, top_n=5):
    """Highest career batting average (runs per dismissal), qualified with a minimum innings count.
    Note: 'innings' here is approximated as distinct matches a player appeared as striker in."""
    runs = all_matches.groupby("striker")["runs_off_bat"].sum()
    innings_played = all_matches.groupby("striker")["match_id"].nunique()
    dismissals = all_matches[all_matches["player_dismissed"].notnull()].groupby("player_dismissed").size()

    df = pd.DataFrame({"runs": runs, "innings": innings_played}).join(
        dismissals.rename("dismissals"), how="left"
    )
    df["dismissals"] = df["dismissals"].fillna(0).astype(int)
    df = df[(df["innings"] >= int(min_innings)) & (df["dismissals"] > 0)].copy()

    if df.empty:
        return []

    df["average"] = (df["runs"] / df["dismissals"]).round(2)
    top = df.sort_values("average", ascending=False).head(int(top_n)).reset_index()
    top = top.rename(columns={"striker": "player"})
    return top[["player", "runs", "dismissals", "average"]].to_dict(orient="records")


def toss_decision_stats_data():
    """How often teams choose to bat vs field after winning the toss, and the toss-winner's
    win rate for each decision. Useful for 'does winning the toss matter' style questions."""
    df = match_info.dropna(subset=["winner", "toss_decision"])
    results = []
    for decision, group in df.groupby("toss_decision"):
        win_rate = (group["toss_winner"] == group["winner"]).mean()
        results.append({
            "decision": decision, "times_chosen": int(len(group)),
            "toss_winner_win_rate_pct": round(win_rate * 100, 1)
        })
    return results


# ============================================================
# ML AGENT FUNCTIONS
# ============================================================

def predict_score_data(current_score, current_wickets, balls_so_far, runs_last_24balls):
    """Predict a live innings' final score and explain the top 3 driving factors via SHAP."""
    current_score = int(current_score)
    current_wickets = int(current_wickets)
    balls_so_far = int(balls_so_far)
    runs_last_24balls = int(runs_last_24balls)

    if current_score < 0:
        raise ValueError("current_score cannot be negative.")
    if current_wickets < 0 or current_wickets > 10:
        raise ValueError("current_wickets must be between 0 and 10.")
    if balls_so_far <= 0 or balls_so_far > 120:
        raise ValueError("balls_so_far must be between 1 and 120.")
    if runs_last_24balls < 0:
        raise ValueError("runs_last_24balls cannot be negative.")

    current_run_rate = (current_score / balls_so_far) * 6
    balls_remaining = 120 - balls_so_far

    features = pd.DataFrame([{
        "current_score": current_score,
        "current_wickets": current_wickets,
        "current_run_rate": current_run_rate,
        "runs_last_24balls": runs_last_24balls,
        "balls_remaining": balls_remaining
    }])[FEATURE_COLS_SCORE]

    predicted = float(score_model.predict(features)[0])

    explainer = shap.LinearExplainer(score_model, X_BACKGROUND)
    shap_values = explainer(features)
    contributions = sorted(
        zip(features.columns, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True
    )
    top_factors = [{"factor": f, "impact_runs": round(float(v), 1)} for f, v in contributions[:3]]

    return {"predicted_final_score": round(predicted), "top_factors": top_factors}


def predict_score_range_data(current_score, current_wickets, balls_so_far, runs_last_24balls):
    """Predicts a final score AND a realistic error range, based on how many balls have been
    bowled — matches the model's actual measured accuracy at different innings stages
    (see Phase 4B error-by-over-stage analysis: ~26 runs MAE before over 10, ~19 runs
    before over 15, ~11 runs after)."""
    base = predict_score_data(current_score, current_wickets, balls_so_far, runs_last_24balls)
    predicted = base["predicted_final_score"]
    balls_so_far = int(balls_so_far)

    if balls_so_far < 60:
        margin = 26
    elif balls_so_far < 90:
        margin = 19
    else:
        margin = 11

    return {
        "predicted_final_score": predicted,
        "likely_range": f"{predicted - margin} to {predicted + margin}",
        "top_factors": base["top_factors"]
    }


def required_run_rate_data(target_score, current_score, balls_so_far):
    """Calculates the run rate required to reach a target score from the current match state.
    Pure arithmetic, no model involved — distinct from predict_score_data/predict_score_range_data,
    which predict where a team will END UP rather than what's needed to reach a given target."""
    target_score = int(target_score)
    current_score = int(current_score)
    balls_so_far = int(balls_so_far)

    if balls_so_far < 0 or balls_so_far > 120:
        return {"error": "balls_so_far must be between 0 and 120."}

    runs_needed = target_score - current_score
    balls_remaining = 120 - balls_so_far

    if balls_remaining <= 0:
        return {"error": "No balls remaining — innings is over."}
    if runs_needed <= 0:
        return {"target_already_reached": True, "runs_needed": runs_needed, "balls_remaining": balls_remaining}

    required_rate = (runs_needed / balls_remaining) * 6
    return {
        "runs_needed": runs_needed, "balls_remaining": balls_remaining,
        "required_run_rate": round(required_rate, 2)
    }


def estimate_winner_likelihood_data(team_a, team_b):
    """Compares two teams' historical win percentage and head-to-head record.
    IMPORTANT: this is a simple historical-comparison heuristic, NOT a trained
    predictive model. Project testing (Phase 4A) found team-level historical
    features do not reliably beat a majority-class baseline for IPL winner
    prediction — treat this as descriptive context, not a forecast."""
    team_a = _resolve_team_name(team_a)
    team_b = _resolve_team_name(team_b)

    stats = {row["team"]: row for row in team_win_percentage_data()}
    a = stats.get(team_a)
    b = stats.get(team_b)

    if a is None or b is None:
        return {
            "error": f"Team name(s) not recognized: '{team_a}', '{team_b}'. "
                     "Use the exact team names from the dataset."
        }

    return {
        "team_a": team_a, "team_a_overall_win_pct": a["win_pct"],
        "team_b": team_b, "team_b_overall_win_pct": b["win_pct"],
        "head_to_head": head_to_head_data(team_a, team_b),
        "disclaimer": (
            "This is a historical-comparison heuristic, not a validated predictive model. "
            "Testing in this project found team-level historical features (win rate, recent "
            "form, venue record) did not reliably outperform a majority-class baseline for "
            "IPL winner prediction."
        )
    }


print("IPL tools loaded successfully.")
print(f"Project root: {PROJECT_ROOT}")