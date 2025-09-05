# =====================================================================
# WebRAG Chatbot with Llama 3.1 (via OpenRouter + Streamlit UI)
# HuggingFace embeddings (local) + OpenRouter for LLM (streaming enabled)
# =====================================================================

import os
import time
import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =====================================================================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

if not OPENROUTER_API_KEY:
    st.error("❌ Please set OPENROUTER_API_KEY in your .env file.")
    st.stop()

# =====================================================================
# 2. CONSTANT URL (Your Website) - must have <p> tags
# =====================================================================
FIXED_URL = "https://api.duckduckgo.com/?q=your_query&format=json"   # ⚠️ replace with real webpage, not API

# =====================================================================
# 3. SCRAPE WEBSITE CONTENT
# =====================================================================
@st.cache_data
def fetch_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text.strip()
    except Exception as e:
        return ""

website_text = fetch_website_text(FIXED_URL)

# Fallback if no text found
if not website_text:
    website_text = "This is fallback content because the website had no readable text."

# =====================================================================
# 4. CREATE EMBEDDINGS + FAISS INDEX (HuggingFace local model)
# =====================================================================
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def create_faiss_index(text, chunk_size=500, chunk_overlap=50):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]
    if not chunks:
        chunks = ["No content available to index."]
    vectors = embedder.embed_documents(chunks)
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))
    return index, chunks, chunk_overlap

# ✅ Fix: unpack 3 values
index, chunks, _ = create_faiss_index(website_text)

# =====================================================================
# 5. HELPER: RETRIEVE RELEVANT CONTEXT
# =====================================================================
def retrieve_context(query, k=3):
    q_vector = embedder.embed_query(query)
    D, I = index.search(np.array([q_vector]).astype("float32"), min(k, len(chunks)))
    return [chunks[i] for i in I[0]]

# =====================================================================
# 6. LLM (Meta Llama 3.1 via OpenRouter, streaming enabled)
# =====================================================================
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_API_BASE,
    temperature=0.1,
    max_tokens=512,
    streaming=True   # 🔥 Enables token-by-token streaming
)

# =====================================================================
# 7. STREAMLIT CHAT UI
# =====================================================================
st.set_page_config(page_title="Your Chatbot", page_icon="🤖", layout="centered")

# Initialize session state
if "user_name" not in st.session_state:
    st.session_state["user_name"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []

# Ask for user name only once
if st.session_state["user_name"] is None:
    st.title("🤖 WelCome to AI Chatbot")
    name = st.text_input("Please Enter Your Name to Start:")
    if name:
        st.session_state["user_name"] = name.strip().title()
        st.rerun()
else:
    st.title("🤖 AI Chatbot")
    st.write(f"Hi **{st.session_state['user_name']}**, How can I help you today? 👋")

    # Display chat history
    for h in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(h["user"])
        with st.chat_message("assistant"):
            st.markdown(h["bot"])

    # Input box for new question
    if prompt := st.chat_input("Type Your Question..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        # ✅ Show spinner while retrieving context
        with st.spinner("🔎 Searching relevant info..."):
            context = retrieve_context(prompt, k=3)
            time.sleep(0.01)  # ensures spinner is visible briefly

        # Build conversation history
        conversation = ""
        for h in st.session_state["history"]:
            conversation += f"User: {h['user']}\nAssistant: {h['bot']}\n"
        conversation += f"User: {prompt}\nAssistant:"

        # Streaming container for assistant reply
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            # Stream response token by token
            for chunk in llm.stream(
                f"Use the following context to answer:\n\n{context}\n\n{conversation}"
            ):
                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")  # typing effect

            placeholder.markdown(full_response)  # final answer

        # Save to history
        st.session_state["history"].append({"user": prompt, "bot": full_response})
