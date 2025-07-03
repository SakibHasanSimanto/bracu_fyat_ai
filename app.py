# -*- coding: utf-8 -*-
import pickle
import faiss
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

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

def generate_answer(user_msg: str) -> str:
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
    history = st.session_state.history[:-1] if st.session_state.history else []

    for i in range(0, len(history), 2):
        user_prev = history[i][1]
        bot_prev  = history[i + 1][1] if i + 1 < len(history) else ""
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

    except requests.exceptions.HTTPError as e:
        st.error("⚠️ The model could not generate a response. This may be due to hitting token limits or a GROQ API issue.")
        st.stop()

    except Exception as e:
        st.error("⚠️ Something went wrong while contacting the GROQ API.")
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

# ---------- 4. Session state initialisation ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- 5. Display past messages ----------
for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

# ---------- 6. User input ----------
user_msg = st.chat_input("Ask me anything about BRACU CSE…")

if user_msg:
    st.chat_message("user").markdown(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = generate_answer(user_msg)
            st.markdown(answer)

    # Update history (for display)
    st.session_state.history.append(("user", user_msg))
    st.session_state.history.append(("assistant", answer))

    # Reset history if it gets too long (to avoid token overflow)
    if len(st.session_state.history) > 10:  # 5 turns = 10 messages (user+assistant)
        st.session_state.history = st.session_state.history[-10:]


