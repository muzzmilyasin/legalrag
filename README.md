# ⚖️ Legal RAG Assistant

A history-aware, hybrid-search RAG (Retrieval-Augmented Generation) system for analyzing legal contracts. Upload one or more PDFs, ask questions in plain language, and get expert-level answers — with the assistant remembering the full context of your conversation.

---

## Features

- **Multi-PDF support** — Upload and query countless PDFs; each upload accumulates into a shared knowledge base
- **Hybrid search (BM25 + Vector)** — Combines semantic vector search with keyword-based BM25 retrieval using Reciprocal Rank Fusion (70% vector / 30% BM25) for more accurate retrieval
- **Semantic chunking** — Documents are split using embedding-based semantic similarity, not just character count
- **History-aware retrieval** — Follow-up questions are automatically rewritten into standalone searchable queries using chat history
- **Legal analyst persona** — The LLM is prompted to flag risky, unfair, or dangerous clauses and explain them in simple language
- **Dark-themed web UI** — Clean Flask + HTML interface, no frontend framework required
- **CLI mode** — Includes a terminal-based chatbot for quick testing
- **Batch ingestion** — Standalone pipeline to pre-ingest PDFs into a persistent ChromaDB

---

## Project Structure

```
legal/
├── web_app.py                  # Flask backend — main web application
├── hybrid_search_retrievel.py  # Standalone hybrid search script (BM25 + Vector + EnsembleRetriever)
├── history_aware_rag.py        # CLI chatbot (terminal mode, uses persistent ChromaDB)
├── ingestion_pipeline.py       # Batch PDF ingestion into persistent ChromaDB
├── static/
│   ├── index.html              # Frontend web UI
│   └── favicon.png             # App icon
├── docs/                       # Put PDFs here for batch ingestion
├── db/                         # ChromaDB vector store (auto-created by ingestion pipeline)
├── requirements.txt
└── .env                        # API keys (not committed)
```

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace — `BAAI/bge-base-en` |
| Vector Store | ChromaDB (in-memory for web app, persistent for CLI) |
| Hybrid Retrieval | BM25 (`langchain-community`) + Vector, fused via `EnsembleRetriever` |
| PDF Parsing | PyMuPDF (`fitz`) |
| Semantic Chunking | `langchain-experimental` SemanticChunker |
| Web Framework | Flask |
| Orchestration | LangChain Core, LangChain Classic |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/legal-rag.git
cd legal-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install rank_bm25  # required for BM25Retriever
```

### 4. Add your API key

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

---

## Running the Web App

```bash
python web_app.py
```

Open your browser at **http://localhost:5000**

**How to use:**
1. Click **"Upload PDF"** in the sidebar and select a legal contract
2. Click **"Process PDF"** — the document will be semantically chunked, embedded, and indexed into both ChromaDB and BM25
3. Upload more PDFs if needed — they accumulate into the same knowledge base
4. Type your question in the chat and press **Send**

---

## Running the CLI Chatbot

For quick terminal-based testing (requires pre-ingested documents):

```bash
# Step 1: Put your PDFs in the docs/ folder, then run:
python ingestion_pipeline.py

# Step 2: Start the chatbot
python history_aware_rag.py
```

---

## Running the Hybrid Search Script

To test hybrid search in isolation (requires pre-ingested ChromaDB):

```bash
python hybrid_search_retrievel.py
```

---

## How It Works

```
User Question
      │
      ▼
[History-Aware Rewriter]       ← rewrites follow-up questions into standalone queries
      │
      ▼
[Hybrid Retriever]
  ├── [ChromaDB Vector Search]  ← semantic similarity (top-10 chunks, weight: 0.7)
  └── [BM25 Keyword Search]     ← keyword matching (top-10 chunks, weight: 0.3)
      │
      ▼ Reciprocal Rank Fusion (EnsembleRetriever)
      │
      ▼
[Legal Analyst LLM]            ← answers with context, flags risky clauses
      │
      ▼
Answer + updated chat history
```

---

## Notes

- The first run will download the HuggingFace embedding model (~430MB); subsequent runs use the cached version
- The web app stores documents in-memory — restarting the server resets the knowledge base
- For persistent storage across restarts, use the `ingestion_pipeline.py` + `history_aware_rag.py` CLI workflow
- `rank_bm25` must be installed separately as it is not included in `requirements.txt` by default
