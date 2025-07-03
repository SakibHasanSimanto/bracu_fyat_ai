# -*- coding: utf-8 -*-
import uuid
import pickle
import faiss
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
import time

# ---------- 0.  Per‑user session ID ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

HISTORY_KEY = f"history_{st.session_state.session_id}"

# ---------- 1. Load data & models (cached) ----------
@st.cache_resource(show_spinner=False)
def load_resources():
    index = faiss.read_index("cse_index.faiss")
    with open("cse_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return index, chunks, embedder

index, chunks, embedder = load_resources()

# ---------- 2. Helper functions ----------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]          # stored in .streamlit/secrets.toml
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME   = "llama3-8b-8192"

def retrieve_context(query, k: int = 2) -> str:
    """Return the top‑k chunks concatenated as context."""
    query_vec = embedder.encode([query])
    _, I = index.search(np.array(query_vec), k)
    return "\n".join(chunks[i] for i in I[0])

def generate_answer(user_msg: str, history: list) -> str:
    context = retrieve_context(user_msg)

    system_prompt = (
        "You are FYAT AI 1.0, a helpful BRAC University CSE assistant. "
        "First, understand the user's query, then concisely answer based on the context below, "
        "fixing grammar/formatting as needed:\n\n"
        f"{context}\n\n"
        "If the context seems insufficient, answer with your pretrained knowledge but say you are doing so, "
        "and politely direct the user to https://cse.sds.bracu.ac.bd/ and https://www.bracu.ac.bd/."
    )

    messages = [{"role": "system", "content": system_prompt}]
    trimmed_history = history[-10:]  # keep last 5 turns (10 messages)

    for i in range(0, len(trimmed_history), 2):
        user_prev = trimmed_history[i][1]
        bot_prev  = trimmed_history[i + 1][1] if i + 1 < len(trimmed_history) else ""
        messages.append({"role": "user", "content": user_prev})
        messages.append({"role": "assistant", "content": bot_prev})

    messages.append({"role": "user", "content": user_msg})

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 800
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError:
        st.error("⚠️ The model could not generate a response (token limit or API issue).")
        st.stop()
    except Exception as e:
        st.error("⚠️ Unexpected error contacting the GROQ API.")
        st.exception(e)
        st.stop()

# ---------- 3. Streamlit UI ----------
st.set_page_config(page_title="FYAT AI – BRACU CSE Assistant", page_icon="🪄")

st.title("🪄 FYAT AI: BRACU CSE Knowledge Assistant")
st.caption("Powered by BAAI embeddings + Llama 3‑8B (GROQ)")

with st.expander("Disclaimer", expanded=False):
    st.markdown(
        "This tool provides information for educational purposes only and is "
        "**not** a substitute for FYAT Mentors or official university resources."
    )

# ---------- 4. Session state initialization ----------
if HISTORY_KEY not in st.session_state:
    st.session_state[HISTORY_KEY] = []

history = st.session_state[HISTORY_KEY]

# ---------- 4.5 Clear memory button ----------
if st.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state[HISTORY_KEY] = []
    st.session_state["last_sent_time"] = 0  # Reset cooldown as well
    st.experimental_rerun()

# ---------- 5. Display past messages ----------
for role, msg in history:
    st.chat_message(role).markdown(msg)

# ---------- 6. Enforce cooldown for sending messages ----------
if "last_sent_time" not in st.session_state:
    st.session_state["last_sent_time"] = 0

elapsed = time.time() - st.session_state["last_sent_time"]
cooldown_seconds = 5

if elapsed < cooldown_seconds:
    wait_time = int(cooldown_seconds - elapsed) + 1
    st.warning(f"⚠️ Seems like you are trying to spam. Refresh for further usage.")
    user_msg = None
else:
    user_msg = st.chat_input("Ask me anything about BRACU CSE…")

# ---------- 7. User input handling ----------
if user_msg:
    st.chat_message("user").markdown(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = generate_answer(user_msg, history)
            st.markdown(answer)

    # Update session history and cooldown timer
    history.append(("user", user_msg))
    history.append(("assistant", answer))
    st.session_state["last_sent_time"] = time.time()

    # Trim history to last 10 messages (5 turns)
    if len(history) > 10:
        history[:] = history[-10:]
