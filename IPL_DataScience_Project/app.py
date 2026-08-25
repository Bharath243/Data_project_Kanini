"""
app.py — IPL Match Intelligence chat frontend (Streamlit).

Design: only the Support Agent is shown to the user — it auto-routes every
question to the Data Agent, ML Agent, or RAG Agent behind the scenes (see
router.py). Every assistant reply shows a small badge naming which specialist
actually answered. Past conversations are saved to a local JSON file and
listed in the left sidebar.

Run with:
    streamlit run app.py

Requires (same folder): tools.py, data_agent.py, ml_agent.py, rag_agent.py,
router.py, plus data/processed/*, models/score_model.pkl already prepared,
and Ollama running locally with llama3.2:3b pulled.
"""

import json
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import streamlit as st
from langchain_ollama import ChatOllama

from Agents.data_agent import build_data_agent
from Agents.ml_agent import build_ml_agent
from Agents.rag_agent import build_rag_agent
from Agents.router import build_router

st.set_page_config(page_title="IPL Assistant", page_icon="🏏", layout="centered")

st.markdown("""
    <style>
        footer {visibility: hidden;}
        [data-testid="stAppDeployButton"] {display: none;}
    </style>
""", unsafe_allow_html=True)

HISTORY_FILE = Path("chat_history.json")
AGENT_ICONS = {
    "Data agent": "", "ML agent": "", "RAG agent": "", "Support agent": "",
}


# ==================== Persisted chat history ====================

def load_chats():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_chats(chats):
    HISTORY_FILE.write_text(json.dumps(chats, indent=2))


if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_chats()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.messages = []


def start_new_chat():
    if st.session_state.messages:
        title = st.session_state.messages[0]["content"][:40]
        st.session_state.all_chats[st.session_state.current_chat_id] = {
            "title": title, "messages": st.session_state.messages,
        }
        save_chats(st.session_state.all_chats)
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.messages = []


def load_chat(chat_id):
    if st.session_state.messages:
        title = st.session_state.messages[0]["content"][:40]
        st.session_state.all_chats[st.session_state.current_chat_id] = {
            "title": title, "messages": st.session_state.messages,
        }
        save_chats(st.session_state.all_chats)
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.all_chats[chat_id]["messages"]


# ==================== Agents (built once, cached) ====================

@st.cache_resource(show_spinner="Setting up agents (first run only, ~30-60s)...")
def build_agents():
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    data_agent = build_data_agent(llm)
    ml_agent = build_ml_agent(llm)
    rag_agent = build_rag_agent(llm)          # loads existing FAISS index from disk if present
    ask_support = build_router(llm, data_agent, ml_agent, rag_agent)

    return ask_support


try:
    ask_support = build_agents()
    setup_error = None
except Exception as e:
    ask_support = None
    setup_error = str(e)

if setup_error:
    st.error(
        "⚠️ Couldn't set up the agents. This usually means Ollama isn't running, "
        "`llama3.2:3b` isn't pulled, or a required package/data file is missing.\n\n"
        f"**Details:** {setup_error}\n\n"
        "**Check:**\n"
        "1. Is Ollama running? Try `ollama list` in a terminal.\n"
        "2. Is the model pulled? Try `ollama pull llama3.2:3b`.\n"
        "3. Are all packages installed? See requirements below.\n"
        "4. Do `data/processed/match_info.csv`, `data/processed/all_matches_clean.csv`, "
        "`data/processed/score_features.csv`, and `models/score_model.pkl` exist relative "
        "to this folder?"
    )
    st.stop()


# ==================== Sidebar: chat history ====================

st.sidebar.title("🏏 IPL Assistant")
if st.sidebar.button("+ New chat", use_container_width=True):
    start_new_chat()
    st.rerun()

if st.sidebar.button("🗑️ Clear all history", use_container_width=True):
    st.session_state.all_chats = {}
    save_chats({})
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
for chat_id, chat in sorted(st.session_state.all_chats.items(), key=lambda x: x[0], reverse=True):
    label = chat["title"] or "Untitled chat"
    if st.sidebar.button(label, key=f"hist_{chat_id}", use_container_width=True):
        load_chat(chat_id)
        st.rerun()


# ==================== Main chat ====================

st.title("🏏 IPL Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("agent"):
            icon = AGENT_ICONS.get(msg["agent"], "")
            st.caption(f"{icon} {msg['agent']}")
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                label, answer = ask_support(prompt)
            except Exception as e:
                label, answer = "Support agent", (
                    "Sorry, something went wrong answering that question. "
                    f"({e})\n\nTry rephrasing, or check that Ollama is still running."
                )

        st.caption(f"{AGENT_ICONS.get(label, '')} {label}")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "agent": label})

    st.session_state.all_chats[st.session_state.current_chat_id] = {
        "title": st.session_state.messages[0]["content"][:40],
        "messages": st.session_state.messages,
    }
    save_chats(st.session_state.all_chats)