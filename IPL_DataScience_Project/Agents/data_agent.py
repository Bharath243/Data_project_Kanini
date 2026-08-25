"""
data_agent.py — Safe IPL Data Agent.

The Data Agent has three safety layers:

1. High-confidence deterministic intent routing for question shapes where
   the correct tool is unambiguous.
2. LLM tool selection for questions that are not covered by deterministic
   intent rules.
3. A compatibility gate that refuses to execute an obviously unrelated
   tool selected by the LLM.

The LLM is therefore never trusted merely because it selected a tool.
"""

import re

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from tools import (
    team_win_percentage_data,
    head_to_head_data,
    highest_team_score_data,
    top_run_scorers_data,
    player_highest_score_data,
    player_match_score_data,
    player_teams_data,
    best_strike_rate_data,
    bowler_stats_data,
    specific_bowler_stats_data,
    venue_stats_data,
    most_sixes_data,
    best_batting_average_data,
    toss_decision_stats_data,
    resolve_player_name_for_agent,
)


class _Msg:
    """Minimal message stand-in exposing .content."""

    def __init__(self, content):
        self.content = content


def _build_data_tools():
    @tool
    def get_team_win_percentage() -> list:
        """Returns IPL team win percentages, sorted highest first."""
        return team_win_percentage_data()

    @tool
    def get_head_to_head(team_a: str, team_b: str) -> dict:
        """Returns the IPL head-to-head win record between two named teams."""
        return head_to_head_data(team_a, team_b)

    @tool
    def get_highest_team_score() -> dict:
        """Returns the highest total TEAM score in one IPL innings."""
        return highest_team_score_data()

    @tool
    def get_top_run_scorers(top_n: int = 5) -> list:
        """Returns players ranked by total career IPL runs."""
        return top_run_scorers_data(top_n)

    @tool
    def get_player_highest_score(player: str) -> dict:
        """Returns a named player's highest individual IPL innings score.

        Do NOT use this for a question specifying both runs and balls; use
        get_player_match_score for that query shape.
        """
        return player_highest_score_data(player)

    @tool
    def get_player_match_score(
        player: str,
        target_runs: int,
        target_balls: int,
    ) -> dict:
        """Finds IPL innings where a named player scored exactly X runs off Y balls."""
        return player_match_score_data(player, target_runs, target_balls)

    @tool
    def get_player_teams(player: str) -> dict:
        """Returns the IPL teams represented by a named player and their count."""
        return player_teams_data(player)

    @tool
    def get_best_strike_rate(top_n: int = 5) -> list:
        """Returns IPL players with the highest qualified career strike rate."""
        return best_strike_rate_data(top_n=top_n)

    @tool
    def get_bowler_stats(top_n: int = 5) -> list:
        """Returns TOP IPL wicket-takers.

        Use only for ranking questions such as 'who has the most wickets?'.
        For statistics about one named bowler use get_specific_bowler_stats.
        """
        return bowler_stats_data(top_n=top_n)

    @tool
    def get_specific_bowler_stats(bowler: str) -> dict:
        """Returns aggregate IPL bowling statistics for one named bowler."""
        return specific_bowler_stats_data(bowler)

    @tool
    def get_venue_stats(top_n: int = 5) -> list:
        """Returns IPL venues with the highest average innings score."""
        return venue_stats_data(top_n=top_n)

    @tool
    def get_most_sixes(top_n: int = 5) -> list:
        """Returns players with the most career IPL sixes."""
        return most_sixes_data(top_n=top_n)

    @tool
    def get_best_batting_average(top_n: int = 5) -> list:
        """Returns players with the highest qualified career batting average."""
        return best_batting_average_data(top_n=top_n)

    @tool
    def get_toss_decision_stats() -> list:
        """Returns toss decision frequency and toss-winner win rate."""
        return toss_decision_stats_data()

    return [
        get_team_win_percentage,
        get_head_to_head,
        get_highest_team_score,
        get_top_run_scorers,
        get_player_highest_score,
        get_player_match_score,
        get_player_teams,
        get_best_strike_rate,
        get_bowler_stats,
        get_specific_bowler_stats,
        get_venue_stats,
        get_most_sixes,
        get_best_batting_average,
        get_toss_decision_stats,
    ]


def _normalize(text: str) -> str:
    text = str(text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _extract_player_candidate(text: str):
    """Extract a conservative player-name candidate from common question forms."""
    q = _normalize(text)

    patterns = [
        # 'bowling stats for Trent Boult'
        r"(?:bowling\s+(?:stats|statistics|figures|record)|"
        r"(?:stats|statistics|record)\s+for)\s+(.+?)\s*[?.!]?$",
        # 'how many teams did MS Dhoni play for?'
        r"how many\s+(?:ipl\s+)?teams?\s+(?:did|has)\s+(.+?)\s+"
        r"(?:play|played|represent|represented)\s+for\b",
        # 'MS Dhoni played for how many teams?'
        r"(.+?)\s+played\s+for\s+how many\s+(?:ipl\s+)?teams?\b",
        # 'Suresh Raina scored 87 ...'
        r"(?:in which match\s+)?(.+?)\s+scored\s+\d+\s+runs?\s+"
        r"(?:off|from|in)\s+\d+\s+balls?\b",
        # 'Suresh Raina 87 off 25 balls'
        r"(?:in which match\s+)?(.+?)\s+\d+\s+off\s+\d+\s+balls?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ?.!,")
            candidate = re.sub(r"^(?:in which match|when|where)\s+", "", candidate)
            candidate = re.sub(r"\s+the$", "", candidate)
            if candidate:
                return candidate

    return None


def _resolve_player_candidate(candidate):
    if not candidate:
        return None
    return resolve_player_name_for_agent(candidate)


def _detect_deterministic_tool(question: str):
    """Return (tool_name, args) for high-confidence question shapes."""
    q = _normalize(question)

    # --------------------------------------------------------
    # Specific score: X runs off Y balls.
    # --------------------------------------------------------
    score_match = re.search(
        r"\b(\d+)\s+runs?\s*(?:off|from|in)\s*(\d+)\s+balls?\b",
        q,
    )
    if not score_match:
        score_match = re.search(
            r"\b(\d+)\s+off\s+(\d+)\s+balls?\b",
            q,
        )

    if score_match:
        candidate = _extract_player_candidate(q)
        resolved = _resolve_player_candidate(candidate)
        if resolved:
            return (
                "get_player_match_score",
                {
                    "player": resolved,
                    "target_runs": int(score_match.group(1)),
                    "target_balls": int(score_match.group(2)),
                },
            )

    # --------------------------------------------------------
    # Player team history/count.
    # --------------------------------------------------------
    if (
        "how many" in q
        and "team" in q
        and "play" in q
    ):
        candidate = _extract_player_candidate(q)
        resolved = _resolve_player_candidate(candidate)
        if resolved:
            return "get_player_teams", {"player": resolved}

    # --------------------------------------------------------
    # Specific bowler statistics.
    # --------------------------------------------------------
    if (
        "bowling stats" in q
        or "bowling statistics" in q
        or "bowling figures" in q
        or "bowling record" in q
    ):
        candidate = _extract_player_candidate(q)
        resolved = _resolve_player_candidate(candidate)
        if resolved:
            return "get_specific_bowler_stats", {"bowler": resolved}

    return None


def _compatibility_status(question: str, tool_name: str):
    """Return (allowed, reason). Never execute an obvious wrong tool."""
    q = _normalize(question)

    # Specific score questions MUST use the exact score+balls tool.
    if re.search(r"\b\d+\s+runs?\s*(?:off|from|in)\s*\d+\s+balls?\b", q) or re.search(
        r"\b\d+\s+off\s+\d+\s+balls?\b", q
    ):
        allowed = tool_name == "get_player_match_score"
        return allowed, "A runs+balls question requires get_player_match_score."

    # Player team history.
    if "how many" in q and "team" in q and "play" in q:
        allowed = tool_name == "get_player_teams"
        return allowed, "A player-team-count question requires get_player_teams."

    # Specific bowler stats.
    if (
        "bowling stats" in q
        or "bowling statistics" in q
        or "bowling figures" in q
        or "bowling record" in q
    ):
        allowed = tool_name == "get_specific_bowler_stats"
        return allowed, "Specific bowler statistics require get_specific_bowler_stats."

    # Explicit ranking of wicket takers.
    if (
        "most wickets" in q
        or "most wicket" in q
        or "top wicket" in q
        or "top wicket-takers" in q
    ):
        allowed = tool_name == "get_bowler_stats"
        return allowed, "A wicket-taker ranking requires get_bowler_stats."

    # Player highest score, not a particular score/balls query.
    if "highest score" in q or "highest individual score" in q:
        allowed = tool_name == "get_player_highest_score"
        return allowed, "A player's highest score requires get_player_highest_score."

    # Top career runs.
    if (
        "most runs" in q
        or "top run scorer" in q
        or "top run scorers" in q
        or "highest run scorer" in q
    ):
        allowed = tool_name == "get_top_run_scorers"
        return allowed, "A run-scoring ranking requires get_top_run_scorers."

    # Toss decision statistic.
    if "toss" in q and (
        "bat" in q or "field" in q or "win rate" in q or "winning" in q
    ):
        allowed = tool_name == "get_toss_decision_stats"
        return allowed, "Toss decision statistics require get_toss_decision_stats."

    return True, "No deterministic incompatibility detected."


def _format_tool_result(tool_name: str, result) -> str:
    """Format every supported tool result deterministically; no LLM rewriting."""
    if isinstance(result, dict) and "error" in result:
        return str(result["error"])

    if tool_name == "get_team_win_percentage":
        if isinstance(result, list) and result:
            top = result[0]
            top_line = (
                f"{top.get('team')} has the highest win percentage in IPL: "
                f"{top.get('win_pct')}% ({top.get('wins')} wins from {top.get('matches')} matches)."
            )
            if len(result) > 1:
                rest = "; ".join(
                    f"{i + 2}. {r.get('team')} ({r.get('win_pct')}%)"
                    for i, r in enumerate(result[1:5])
                )
                return f"{top_line} Next highest: {rest}."
            return top_line
        return f"Team win percentage data: {result}"

    if tool_name == "get_head_to_head":
        if isinstance(result, dict):
            total = result.get("total_matches", 0)
            if total == 0:
                return (
                    f"No matches found between '{result.get('team_a')}' and "
                    f"'{result.get('team_b')}' in the dataset."
                )
            return (
                f"Head-to-head record: {result.get('team_a')} has won "
                f"{result.get('team_a_wins')} matches, while {result.get('team_b')} "
                f"has won {result.get('team_b_wins')} matches "
                f"(out of {total} meetings)."
            )

    if tool_name == "get_highest_team_score":
        return (
            f"The highest team total in IPL history is {result.get('score')} runs, "
            f"scored by {result.get('team')} against {result.get('opponent')} "
            f"in the {result.get('season')} season."
            if isinstance(result, dict)
            else f"Highest team score data: {result}"
        )

    if tool_name == "get_top_run_scorers":
        if isinstance(result, list) and result:
            lines = "; ".join(
                f"{i + 1}. {r.get('player')} ({r.get('total_runs')} runs)"
                for i, r in enumerate(result)
            )
            return f"Top run scorers in IPL history: {lines}."
        return f"Top run scorers data: {result}"

    if tool_name == "get_player_highest_score":
        if isinstance(result, dict):
            return (
                f"{result.get('player')}'s highest individual IPL score is "
                f"{result.get('highest_score', result.get('score', 'N/A'))} "
                f"against {result.get('opponent', 'their opponent')}."
            )

    if tool_name == "get_player_match_score":
        if isinstance(result, dict):
            if not result.get("found"):
                return result.get("message", "No matching IPL innings was found in the dataset.")

            matches = result.get("matches", [])
            if len(matches) == 1:
                m = matches[0]
                details = (
                    f"{result.get('player')} scored {result.get('runs')} runs off "
                    f"{result.get('balls')} balls against {m.get('opponent')} "
                    f"in IPL {m.get('season')} (match ID {m.get('match_id')})."
                )
                if m.get("date"):
                    details += f" Date: {m.get('date')}."
                if m.get("venue"):
                    details += f" Venue: {m.get('venue')}."
                return details

            lines = "; ".join(
                f"{m.get('season')}: vs {m.get('opponent')} "
                f"(match ID {m.get('match_id')})"
                for m in matches
            )
            return (
                f"{result.get('player')} scored {result.get('runs')} off "
                f"{result.get('balls')} in these IPL innings: {lines}."
            )

    if tool_name == "get_player_teams":
        if isinstance(result, dict):
            return (
                f"{result.get('player')} played for {result.get('team_count')} "
                f"IPL teams: {', '.join(result.get('teams', []))}."
            )

    if tool_name == "get_best_strike_rate":
        if isinstance(result, list) and result:
            lines = "; ".join(
                f"{i + 1}. {r.get('striker')} ({r.get('strike_rate')} SR)"
                for i, r in enumerate(result)
            )
            return f"Best IPL strike rates (min. 500 balls faced): {lines}."
        return f"Strike rate data: {result}"

    if tool_name == "get_bowler_stats":
        if isinstance(result, list) and result:
            lines = "; ".join(
                f"{i + 1}. {r.get('bowler')} ({r.get('wickets')} wickets, econ {r.get('economy')})"
                for i, r in enumerate(result)
            )
            return f"Top IPL wicket-takers: {lines}."
        return f"Bowler stats data: {result}"

    if tool_name == "get_specific_bowler_stats":
        if isinstance(result, dict):
            return (
                f"{result.get('bowler')} bowling stats: "
                f"{result.get('wickets')} wickets, "
                f"{result.get('runs_conceded')} runs conceded, "
                f"{result.get('legal_balls')} legal balls, "
                f"economy rate {result.get('economy')}."
            )

    if tool_name == "get_venue_stats":
        if isinstance(result, list) and result:
            lines = "; ".join(
                f"{i + 1}. {r.get('venue')} (avg {r.get('avg_score')} runs)"
                for i, r in enumerate(result)
            )
            return f"Highest-scoring IPL venues: {lines}."
        return f"Venue stats data: {result}"

    if tool_name == "get_most_sixes":
        if isinstance(result, list) and result:
            lines = "; ".join(
                f"{i + 1}. {r.get('player')} ({r.get('sixes')} sixes)"
                for i, r in enumerate(result)
            )
            return f"Most career sixes in IPL: {lines}."
        return f"Sixes data: {result}"

    if tool_name == "get_best_batting_average":
        if isinstance(result, list) and result:
            lines = "; ".join(
                f"{i + 1}. {r.get('player')} (avg {r.get('average')})"
                for i, r in enumerate(result)
            )
            return f"Best career batting averages in IPL: {lines}."
        return f"Batting average data: {result}"

    if tool_name == "get_toss_decision_stats":
        if isinstance(result, list):
            bat = next((x for x in result if str(x.get("decision", "")).lower() == "bat"), {})
            field = next((x for x in result if str(x.get("decision", "")).lower() == "field"), {})
            return (
                f"When the toss winner chose to bat, they won "
                f"{bat.get('toss_winner_win_rate_pct', 'N/A')}% of the time, "
                f"compared to {field.get('toss_winner_win_rate_pct', 'N/A')}% "
                f"when choosing to field."
            )

    if isinstance(result, dict):
        return ", ".join(f"{k}: {v}" for k, v in result.items())
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return ", ".join(f"{k}: {v}" for k, v in result[0].items())
    return str(result)


class DataAgent:
    """Safe IPL data agent with deterministic routing and a compatibility gate."""

    def __init__(self, llm):
        self.tools = _build_data_tools()
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = llm.bind_tools(self.tools)

    def _run_tool(self, tool_name, args):
        selected_tool = self.tool_map.get(tool_name)
        if selected_tool is None:
            return "I couldn't find a matching IPL data tool for that question."

        try:
            result = selected_tool.invoke(args)
        except Exception as e:
            return f"That data tool failed to run: {e}"

        return _format_tool_result(tool_name, result)

    def invoke(self, payload):
        question = str(payload["messages"][0]["content"])

        # ----------------------------------------------------
        # Layer 1: exact deterministic intents.
        # ----------------------------------------------------
        deterministic = _detect_deterministic_tool(question)
        if deterministic is not None:
            tool_name, args = deterministic
            answer = self._run_tool(tool_name, args)
            return {"messages": [_Msg(answer)]}

        # ----------------------------------------------------
        # Layer 2: LLM selects a tool for everything else.
        # ----------------------------------------------------
        response = self.llm_with_tools.invoke(
            [HumanMessage(content=question)]
        )

        if not response.tool_calls:
            return {
                "messages": [
                    _Msg(
                        "I don't have a suitable IPL data tool for that question."
                    )
                ]
            }

        call = response.tool_calls[0]
        tool_name = call.get("name")
        args = call.get("args", {}) or {}

        # ----------------------------------------------------
        # Layer 3: never trust an LLM-selected tool blindly.
        # ----------------------------------------------------
        allowed, reason = _compatibility_status(question, tool_name)
        if not allowed:
            return {
                "messages": [
                    _Msg(
                        "I don't have a suitable tool for that exact question, "
                        "so I won't substitute a different statistic."
                    )
                ]
            }

        answer = self._run_tool(tool_name, args)
        return {"messages": [_Msg(answer)]}


def build_data_agent(llm) -> DataAgent:
    """Factory used by app.py / router.py."""
    return DataAgent(llm)