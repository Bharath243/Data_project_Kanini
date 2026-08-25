"""
router.py — IPL Support Agent / Router.

The router uses deterministic high-confidence routing before invoking the
LLM. This prevents a small local model from choosing a semantically unrelated
specialist when the intent is obvious.

Fallback behavior:
- Data Agent: factual IPL statistics/history.
- ML Agent: predictions, score ranges, required run rate, win likelihood.
- RAG Agent: rules, formats, terminology.
- Support Agent: greetings and questions about this application.
"""

import re

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage


SUPPORT_SYSTEM_PROMPT = (
    "You are the IPL Assistant's Support Agent.\n\n"
    "Choose exactly ONE specialist tool when a specialist is needed.\n\n"
    "ask_data_agent = factual IPL dataset questions: player statistics, "
    "team statistics/history, match records, rankings, counts, percentages, "
    "averages, venues, toss statistics, head-to-head records, and specific "
    "historical innings.\n"
    "ask_ml_agent = predictions/calculations supported by the ML Agent: live "
    "score prediction, score range, required run rate, or comparison of two "
    "teams' win likelihood.\n"
    "ask_rag_agent = IPL rules, formats and terminology only, such as DLS, "
    "Super Over, powerplay, playoffs and Impact Player rules.\n\n"
    "CRITICAL RULES:\n"
    "1. Any factual IPL statistic, number, ranking, record, count, player/team "
    "history or historical match statistic belongs to ask_data_agent.\n"
    "2. A question about whether winning the toss or choosing bat/field affects "
    "historical win rate is a DATA question, not a prediction.\n"
    "3. Do not send factual statistics to ask_rag_agent.\n"
    "4. Do not send historical statistics to ask_ml_agent.\n"
    "5. Only answer directly without a tool for greetings or questions about "
    "this application itself.\n"
)


DEFAULT_HELP_MESSAGE = (
    "I can answer IPL statistics and historical records, predictions, and "
    "general IPL rules/terminology. For example: 'Who has scored the most "
    "runs in IPL?', 'How many teams did MS Dhoni play for?', or 'What is the DLS method?'"
)


_APP_QUESTION_EXACT_PHRASES = {
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "help",
    "what can you do", "what can this app do", "what does this app do",
    "what is this app", "what is this", "who are you", "what are you",
    "how does this work", "how do i use this", "how do i use this app",
    "how do i ask a question", "what kind of questions can i ask",
    "what questions can i ask", "what can i ask",
}

_APP_QUESTION_SUBSTRINGS = (
    "what can you do",
    "what can this app do",
    "what does this app do",
    "how do i ask",
    "how do i use this app",
    "what kind of questions",
    "what questions can i ask",
    "what is this app",
    "what can i ask you",
)


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s?]", "", str(text).strip().lower())
    return re.sub(r"\s+", " ", text)


def _looks_like_app_question(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", "", str(text).strip().lower())
    if normalized in _APP_QUESTION_EXACT_PHRASES:
        return True

    words = normalized.split()
    if words and words[0] in {"hi", "hello", "hey"} and len(words) <= 4:
        return True

    if "thank" in normalized or normalized.startswith("bye"):
        return True

    return any(phrase in normalized for phrase in _APP_QUESTION_SUBSTRINGS)


def _looks_like_data_question(question: str) -> bool:
    q = _normalize(question)

    data_patterns = (
        r"\b\d+\s+runs?\s*(?:off|from|in)\s*\d+\s+balls?\b",
        r"\b\d+\s+off\s+\d+\s+balls?\b",
        r"how many\s+(?:ipl\s+)?teams?\b",
        r"played\s+for\s+how many\s+(?:ipl\s+)?teams?\b",
        r"bowling\s+(?:stats|statistics|figures|record)\b",
        r"most\s+wickets?\b",
        r"top\s+wicket",
        r"highest\s+(?:individual\s+)?score\b",
        r"highest\s+team\s+(?:score|total)\b",
        r"most\s+runs?\b",
        r"top\s+run\s+scorers?\b",
        r"win\s+(?:percentage|rate)\b",
        r"head[- ]to[- ]head\b",
        r"toss.*(?:bat|field|win|chance|rate)",
        r"most\s+sixes?\b",
        r"strike\s+rate\b",
        r"batting\s+average\b",
        r"venue.*(?:average|score)",
        r"how many\b.*\b(?:runs|wickets|sixes|matches|balls|teams)\b",
        r"(?:record|statistics|stats)\b",
    )

    return any(re.search(pattern, q) for pattern in data_patterns)


def _looks_like_ml_question(question: str) -> bool:
    q = _normalize(question)

    ml_patterns = (
        r"\bpredict\b",
        r"\bprediction\b",
        r"\bproject(?:ed|ion)?\b.*\bscore\b",
        r"\bscore\s+range\b",
        r"\bfinal\s+score\b.*\b(?:predict|estimate)\b",
        r"\brequired\s+run\s+rate\b",
        r"\brrr\b",
        r"\bwin\s+likelihood\b",
        r"\bwin\s+probability\b",
        r"\bwho\s+is\s+more\s+likely\s+to\s+win\b",
    )

    return any(re.search(pattern, q) for pattern in ml_patterns)


def _looks_like_rag_question(question: str) -> bool:
    q = _normalize(question)

    rag_patterns = (
        r"\bwhat\s+is\s+(?:the\s+)?dls\b",
        r"\bhow\s+does\s+dls\b",
        r"\bwhat\s+is\s+a\s+super\s+over\b",
        r"\bhow\s+does\s+(?:the\s+)?super\s+over\b",
        r"\bwhat\s+is\s+(?:the\s+)?powerplay\b",
        r"\bhow\s+do\s+playoffs?\s+work\b",
        r"\bimpact\s+player\s+rule\b",
        r"\bwhat\s+happens\s+in\s+a\s+tie\b",
        r"\bipl\s+rules?\b",
        r"\bipl\s+format\b",
    )

    return any(re.search(pattern, q) for pattern in rag_patterns)


def build_router(llm, data_agent, ml_agent, rag_agent):
    """Build ask_support(question) -> (label, answer)."""

    @tool
    def ask_data_agent(question: str) -> str:
        """Routes factual IPL statistics and historical dataset questions."""
        result = data_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    @tool
    def ask_ml_agent(question: str) -> str:
        """Routes supported IPL predictions/calculations."""
        result = ml_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    @tool
    def ask_rag_agent(question: str) -> str:
        """Routes IPL rules, formats and terminology questions only."""
        result = rag_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    support_tools = [ask_data_agent, ask_ml_agent, ask_rag_agent]
    tool_labels = {
        "ask_data_agent": "Data agent",
        "ask_ml_agent": "ML agent",
        "ask_rag_agent": "RAG agent",
    }

    llm_with_tools = llm.bind_tools(support_tools)

    def ask_support(question: str):
        question = str(question).strip()

        # Deterministic app/support handling.
        if _looks_like_app_question(question):
            return "Support agent", DEFAULT_HELP_MESSAGE

        # Deterministic specialist routing for obvious intents.
        # This prevents a local LLM from choosing the wrong specialist.
        if _looks_like_data_question(question):
            try:
                answer = ask_data_agent.invoke({"question": question})
            except Exception as e:
                return "Data agent", (
                    f"Something went wrong answering that data question ({e})."
                )
            return "Data agent", str(answer)

        if _looks_like_ml_question(question):
            try:
                answer = ask_ml_agent.invoke({"question": question})
            except Exception as e:
                return "ML agent", (
                    f"Something went wrong answering that prediction question ({e})."
                )
            return "ML agent", str(answer)

        if _looks_like_rag_question(question):
            try:
                answer = ask_rag_agent.invoke({"question": question})
            except Exception as e:
                return "RAG agent", (
                    f"Something went wrong answering that IPL rules question ({e})."
                )
            return "RAG agent", str(answer)

        # Fallback to the LLM for less obvious wording.
        messages = [
            SystemMessage(content=SUPPORT_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            content = str(response.content or "").strip()
            return "Support agent", content if content else DEFAULT_HELP_MESSAGE

        call = response.tool_calls[0]
        selected = next(
            (t for t in support_tools if t.name == call.get("name")),
            None,
        )

        if selected is None:
            return "Support agent", DEFAULT_HELP_MESSAGE

        try:
            answer = selected.invoke(call.get("args", {}))
        except Exception as e:
            return "Support agent", (
                f"Something went wrong answering that question ({e}). "
                "Try rephrasing it."
            )

        return tool_labels.get(call.get("name"), "Support agent"), str(answer)

    return ask_support