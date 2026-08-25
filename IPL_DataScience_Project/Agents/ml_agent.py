"""
ml_agent.py — IPL ML Agent.

Wraps the trained score-prediction model (SHAP-explained) plus two simple
arithmetic/heuristic tools from tools.py as LangChain tools, and builds a
LangChain create_agent() agent around them.
"""

from langchain_core.tools import tool
from langchain.agents import create_agent

from tools import (
    predict_score_data, predict_score_range_data, required_run_rate_data,
    estimate_winner_likelihood_data,
)


def _build_ml_tools():
    @tool
    def predict_final_score(current_score: int, current_wickets: int, balls_so_far: int, runs_last_24balls: int) -> dict:
        """Predicts a live cricket innings' final score and explains the top factors via SHAP."""
        return predict_score_data(current_score, current_wickets, balls_so_far, runs_last_24balls)

    @tool
    def predict_score_range(current_score: int, current_wickets: int, balls_so_far: int, runs_last_24balls: int) -> dict:
        """Predicts a final score WITH a realistic confidence range. Use this instead of
        predict_final_score when a 'range' or 'confidence' is asked for."""
        return predict_score_range_data(current_score, current_wickets, balls_so_far, runs_last_24balls)

    @tool
    def required_run_rate(target_score: int, current_score: int, balls_so_far: int) -> dict:
        """Calculates the run rate required to reach a target score from the current match
        state. Use ONLY when a target/winning score is given."""
        return required_run_rate_data(target_score, current_score, balls_so_far)

    @tool
    def estimate_winner_likelihood(team_a: str, team_b: str) -> dict:
        """Compares two teams' historical win rate and head-to-head record. This is a
        descriptive heuristic, not a validated prediction — always include the disclaimer
        field in your answer."""
        return estimate_winner_likelihood_data(team_a, team_b)

    return [predict_final_score, predict_score_range, required_run_rate, estimate_winner_likelihood]


def build_ml_agent(llm):
    """Factory used by app.py / router.py."""
    ml_tools = _build_ml_tools()
    return create_agent(
        model=llm, tools=ml_tools,
        system_prompt=(
            "You are the IPL ML Agent. You have four capabilities: predicting a live innings' "
            "final score, predicting a score WITH a confidence range, calculating the run rate "
            "required to reach a target, and comparing two teams' historical win likelihood. "
            "Choose the tool that matches exactly what was asked. NEVER answer from your own "
            "pretrained knowledge, even for facts you feel confident about. If a question is "
            "outside these four things — including any question about a specific player's "
            "individual stats — you have NO tool for it. Say so honestly. Never invent a number."
        ),
    )