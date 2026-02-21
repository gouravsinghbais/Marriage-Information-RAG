"""
Marriage RAG — Single File App
All-in-one: Config + RAG Pipeline + Streamlit UI
LLM: OpenRouter (open-source models)
"""

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────────
# ⚙️  CONFIGURATION
# Sensitive values (API keys) are read from st.secrets.
# - Locally:  define them in .streamlit/secrets.toml  (never commit this file)
# - On Cloud: paste the same TOML into the "Advanced settings" secrets field
# Non-sensitive defaults can be overridden via environment variables.
# ─────────────────────────────────────────────────

# Sensitive — must be in st.secrets (or env as fallback for CI/testing)
OPENROUTER_API_KEY  = st.secrets.get("OPENROUTER_API_KEY",  os.getenv("OPENROUTER_API_KEY", ""))
OPENROUTER_BASE_URL = st.secrets.get("OPENROUTER_BASE_URL", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
LLM_MODEL           = st.secrets.get("LLM_MODEL",           os.getenv("LLM_MODEL",           "z-ai/glm-4.5-air:free"))

# Non-sensitive defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH      = os.getenv("CHROMA_PATH",     "data/chroma_db")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "marriage_docs")


# ─────────────────────────────────────────────────
# 🤖  RAG SERVICE  (cached across reruns)
# ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_rag_service():
    os.makedirs(CHROMA_PATH, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    llm = ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=LLM_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful wedding planning assistant.
Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say so clearly and politely.

Context:
{context}

Question: {question}"""
    )

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return vector_store, chain


def ingest_file(vector_store, uploaded_file) -> int:
    """Persist an uploaded file, chunk it, and add to the vector store."""
    tmp_dir  = "data/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)

    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            docs = PyPDFLoader(tmp_path).load()
        else:
            docs = TextLoader(tmp_path).load()

        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)

        if splits:
            vector_store.add_documents(splits)

        return len(splits)
    finally:
        os.remove(tmp_path)


# ─────────────────────────────────────────────────
# 🎨  PAGE CONFIG & CSS
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Marriage Planner AI",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

* { font-family: 'Outfit', sans-serif !important; }

/* ── Background ── */
.stApp {
    background-image:
        radial-gradient(at 0%   0%, hsla(253,16%,7%,1)  0, transparent 55%),
        radial-gradient(at 50%  0%, hsla(225,39%,30%,1) 0, transparent 55%),
        radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 55%);
    min-height: 100vh;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #FF6B6B, #FF8E53);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 10px rgba(255,107,107,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(255,107,107,0.35);
}

/* ── File uploader ── */
.stFileUploader > div > div {
    background: rgba(255,255,255,0.05);
    border: 1.5px dashed rgba(255,255,255,0.25);
    border-radius: 12px;
}
.stFileUploader > div > div:hover {
    border-color: #FF6B6B;
    background: rgba(255,107,107,0.06);
}
.stFileUploader label { color: #d1d5db !important; }

/* ── Upload card ── */
.upload-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2rem;
    margin: 0.5rem 0 2rem;
}
.upload-card h3 { color: #f9fafb !important; }
.upload-card p  { color: #9ca3af !important; }

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}
div[data-testid="stChatMessage"][data-author="user"] {
    background: linear-gradient(135deg, #FF6B6B, #FF8E53) !important;
}
div[data-testid="stChatMessage"][data-author="user"] p,
div[data-testid="stChatMessage"][data-author="user"] div,
div[data-testid="stChatMessage"][data-author="user"] li {
    color: #fff !important;
}
div[data-testid="stChatMessage"][data-author="assistant"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb;
}
div[data-testid="stChatMessage"][data-author="assistant"] p,
div[data-testid="stChatMessage"][data-author="assistant"] div,
div[data-testid="stChatMessage"][data-author="assistant"] li,
div[data-testid="stChatMessage"][data-author="assistant"] strong {
    color: #1f2937 !important;
}

/* ── Chat input ── */
.stChatInput > div > div {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 24px;
}
.stChatInput > div > div:focus-within {
    border-color: #FF6B6B;
    box-shadow: 0 0 0 2px rgba(255,107,107,0.2);
}
div[data-testid="stBottom"] > div,
div[data-testid="stBottom"] { background: transparent !important; }

/* ── Title ── */
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(to right, #FF6B6B, #FF8E53);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.3rem;
}
.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* ── Status badge ── */
.model-badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #d1d5db;
    text-align: center;
}

/* ── Progress bar ── */
.stProgress > div > div > div { background: #FF6B6B; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* ── Blinking cursor animation ── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
.blink-cursor {
    display: inline-block;
    width: 10px;
    height: 1.1em;
    background: #FF6B6B;
    border-radius: 2px;
    vertical-align: text-bottom;
    animation: blink 0.85s step-start infinite;
    margin-left: 2px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# 🔀  SESSION STATE
# ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False


# ─────────────────────────────────────────────────
# 🚀  LOAD RAG SERVICE
# ─────────────────────────────────────────────────
vector_store, chain = load_rag_service()


# ─────────────────────────────────────────────────
# 🖥️  HEADER
# ─────────────────────────────────────────────────
st.markdown('<div class="main-title">💍 Harshit & Shreya\'s Wedding Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Our intelligent companion for a perfectly organized celebration</div>',
    unsafe_allow_html=True,
)

# ── Top action bar ──────────────────────────────
col_gap, col_upload = st.columns([7, 2])

with col_upload:
    if st.button("📂 Upload Documents", use_container_width=True):
        st.session_state.show_upload = not st.session_state.show_upload


# ─────────────────────────────────────────────────
# 📂  UPLOAD PANEL (toggle)
# ─────────────────────────────────────────────────
if st.session_state.show_upload:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Upload Wedding Documents")
    st.markdown("Supports PDF and TXT — contracts, itineraries, guest lists, venue details, etc.")

    uploaded_files = st.file_uploader(
        "Drop your files here",
        accept_multiple_files=True,
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    btn_col, _ = st.columns([2, 6])
    with btn_col:
        process_clicked = st.button("⚡ Process & Index", use_container_width=True)

    if process_clicked:
        if uploaded_files:
            progress    = st.progress(0)
            status      = st.empty()
            total_chunks = 0

            for i, file in enumerate(uploaded_files):
                status.info(f"Processing **{file.name}**…")
                try:
                    n = ingest_file(vector_store, file)
                    total_chunks += n
                    status.success(f"✅ {file.name} → {n} chunks indexed")
                except Exception as e:
                    status.error(f"❌ {file.name}: {e}")
                progress.progress((i + 1) / len(uploaded_files))

            st.success(
                f"🎉 Done! {len(uploaded_files)} file(s) processed · {total_chunks} chunks indexed."
            )
            st.session_state.show_upload = False
            st.rerun()
        else:
            st.warning("Please select at least one file first.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────
# 💬  CHAT INTERFACE
# ─────────────────────────────────────────────────

# Render existing messages
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Empty-state hint
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center;color:#6b7280;padding:4rem 0;">
            <p style="font-size:3rem;margin:0">💬</p>
            <p style="font-size:1.1rem;color:#9ca3af;margin-top:1rem;">
                Ask anything about our wedding — dates, venues, ceremonies, guests…
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Chat input
if prompt := st.chat_input("Ask about the wedding details…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        answer_container = st.empty()
        full_answer = ""

        # Show blinking cursor while waiting for the full response
        answer_container.markdown(
            '<span class="blink-cursor"></span>', unsafe_allow_html=True
        )

        try:
            full_answer = chain.invoke(prompt)
            answer_container.markdown(full_answer)  # replace cursor with answer
        except Exception as e:
            full_answer = f"⚠️ Error: {e}"
            answer_container.markdown(full_answer)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
