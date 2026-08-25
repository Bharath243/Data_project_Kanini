"""
rag_agent.py — IPL RAG Agent.

Knowledge base = (1) hand-written rules/format/terminology docs, plus (2)
short auto-generated team/venue/leaderboard profile paragraphs built from
tools.py's already-computed, already-verified stats. RAG never computes
anything itself — it only retrieves pre-written, pre-verified text. Indexing
raw CSV rows was considered and rejected: vector search can't aggregate/sum,
so it can't answer exact numeric questions reliably — the Data Agent's Pandas
tools remain the source of truth for those.

Fix vs. the earlier version: this used to run a second LLM pass (via
create_agent) to compose an answer around the retrieved document. On the 3B
local model that pass sometimes ignored the "use ONLY the retrieved
document" instruction and answered from its own pretrained knowledge instead
— e.g. fabricating a team bio with real-sounding but wrong facts (wrong
owner, wrong stadium, invented win totals). This version never runs a second
LLM pass: it returns the literal retrieved document text, the same
deterministic-formatting fix already used by the Data Agent. It also checks
the retrieval distance and refuses to answer — rather than confidently
returning the nearest-but-irrelevant document — when nothing in the
knowledge base is actually close enough to the question (this is also what
stops an unrelated app/support question that slips through routing from
getting a wrong, off-topic "answer").

FAISS persistence: the index is saved to FAISS_INDEX_DIR and loaded from disk
on startup instead of being rebuilt on every run. Call
build_rag_agent(llm, rebuild=True) after the underlying cleaned data/stats
change, to regenerate and re-persist the index.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# L2 distance cutoff for all-MiniLM-L6-v2 embeddings: a retrieved document
# with a distance below this is treated as a genuine match; above it, treated
# as "nothing relevant found" rather than being returned anyway. This is a
# starting heuristic — if you see false "not found" answers, raise it; if
# you see off-topic answers slipping through, lower it.
RAG_DISTANCE_THRESHOLD = 1.0

RULES_DOCS = [
    {"title": "IPL Powerplay Rules",
     "content": "In IPL T20 matches, the powerplay covers the first 6 overs of each innings. "
                "During this period, a maximum of 2 fielders are allowed outside the 30-yard "
                "circle, encouraging attacking batting and higher risk-taking from the "
                "batting side."},
    {"title": "DLS Method",
     "content": "The Duckworth-Lewis-Stern (DLS) method is used to calculate a revised "
                "target score for the team batting second in a match interrupted by "
                "weather, most commonly rain. It accounts for both overs lost and wickets "
                "already in hand at the point of interruption."},
    {"title": "Super Over Rules",
     "content": "If an IPL match ends in a tie, the result is decided by a Super Over: each "
                "team bats one over, and whichever team scores more runs in that over wins. "
                "If the Super Over also ties, additional Super Overs are played until a "
                "winner is decided."},
    {"title": "IPL Playoff Format",
     "content": "The top 4 teams in the league table qualify for the playoffs. The format "
                "uses a Qualifier 1 (1st vs 2nd place), an Eliminator (3rd vs 4th place), a "
                "Qualifier 2 (loser of Qualifier 1 vs winner of the Eliminator), and finally "
                "the Final."},
    {"title": "Impact Player Rule",
     "content": "Introduced in the 2023 IPL season, the Impact Player rule allows each team "
                "to substitute one player during the match with a player from their squad, "
                "effectively giving teams a 12th playing member for tactical flexibility."},
]


# NOTE: team/venue/leaderboard profile summaries were REMOVED from this
# knowledge base. Real testing showed they caused the router to send stats
# questions here instead of to the Data Agent ~70% of the time ("highest win
# percentage", "top wicket-takers", "best strike rate", etc. all misrouted to
# RAG) — the profile text made ask_rag_agent look like a plausible answer to
# almost any stats question. RAG is now scoped strictly to rules/format/
# terminology, which has zero overlap with what the Data Agent covers.
def _load_or_build_vector_store(embeddings, rebuild=False) -> FAISS:
    if not rebuild and FAISS_INDEX_DIR.exists():
        try:
            return FAISS.load_local(
                str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            pass  # on-disk index missing/corrupt -> fall through and rebuild

    documents = [Document(page_content=d["content"], metadata={"title": d["title"]}) for d in RULES_DOCS]
    store = FAISS.from_documents(documents, embeddings)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(FAISS_INDEX_DIR))
    return store


class _Msg:
    """Minimal stand-in for a LangChain message — just needs .content, to
    match what the other specialist agents (Data/ML) return."""
    def __init__(self, content):
        self.content = content


class RagAgent:
    """Agent-like wrapper exposing .invoke({"messages": [{"role": "user", "content": q}]})
    -> {"messages": [<obj with .content>]}, matching the convention every
    specialist agent uses. Deliberately has NO second LLM pass — it returns
    retrieved text verbatim, so it can never fabricate facts on top of what's
    actually in the knowledge base."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def invoke(self, payload):
        question = payload["messages"][0]["content"]
        results = self.vector_store.similarity_search_with_score(question, k=1)

        if not results or results[0][1] > RAG_DISTANCE_THRESHOLD:
            return {"messages": [_Msg(
                "No relevant knowledge document found for this question. I can answer "
                "questions about IPL rules/format/terminology, or general team, venue, and "
                "leaderboard summaries."
            )]}

        doc, _distance = results[0]
        # Defensive .get() rather than ['title'] — protects against a stale
        # persisted index built by an older document shape lacking this key.
        title = doc.metadata.get("title", "IPL Knowledge")
        return {"messages": [_Msg(f"{title}: {doc.page_content}")]}


def build_rag_agent(llm, rebuild: bool = False) -> RagAgent:
    """Factory used by app.py / router.py. `llm` is accepted for interface
    consistency with build_data_agent/build_ml_agent but is unused — the RAG
    Agent never runs a second LLM pass (see RagAgent docstring above).

    rebuild=False (default): load the persisted FAISS index from
    FAISS_INDEX_DIR if present, otherwise build it once and save it.
    rebuild=True: force a fresh rebuild (use after cleaned knowledge/stats change).
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = _load_or_build_vector_store(embeddings, rebuild=rebuild)
    return RagAgent(vector_store)