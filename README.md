# 💍 AI Wedding Planner Assistant

A production-ready Retrieval Augmented Generation (RAG) system for managing wedding documents and queries.
Built with **LangChain**, **ChromaDB**, and **Streamlit** — all in a single file.

## 🚀 Features

- **Document Ingestion**: Upload PDF contracts, itineraries, and guest lists (hidden behind a toggle button).
- **Intelligent Querying**: Ask natural language questions about your wedding details.
- **Local Privacy**: Uses local vector storage (ChromaDB) and local LLMs (Ollama) by default.
- **Configurable LLM**: Supports both Ollama (local) and OpenRouter (cloud, open-source models).
- **Modern UI**: Glassmorphism-inspired dark theme with smooth gradients.

## 🛠 Tech Stack

| Layer       | Technology                                      |
|-------------|-------------------------------------------------|
| UI          | Streamlit                                       |
| RAG Engine  | LangChain                                       |
| Vector DB   | ChromaDB (Local Persisted)                      |
| Embeddings  | Sentence Transformers (all-MiniLM-L6-v2)        |
| LLM         | Ollama (Llama 3) or OpenRouter (any model)      |

## 📋 Prerequisites

1. **Python 3.10+**
2. **Ollama** (for local LLM inference):
   - Install from [ollama.com](https://ollama.com/)
   - Pull a model: `ollama pull llama3`

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the project root (optional — sensible defaults are built-in):

```env
# LLM Provider: "ollama" (default) or "openrouter"
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434

# For OpenRouter:
# LLM_PROVIDER=openrouter
# OPENROUTER_API_KEY=your_key_here
# LLM_MODEL=meta-llama/llama-3-70b-instruct
```

## 🏃 Running

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 📂 Project Structure

```
├── app.py             # ✅ All-in-one: Config + RAG + UI
├── data/
│   └── chroma_db/     # Vector database (auto-created)
└── requirements.txt
```

## 💡 Usage

1. Click **"📂 Upload Documents"** to expand the upload panel.
2. Drop your PDF or TXT wedding documents and click **"⚡ Process & Index"**.
3. Once indexed, type your questions in the chat box!
