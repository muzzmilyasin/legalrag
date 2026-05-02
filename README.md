# ⚖️ Legal RAG Assistant

A history-aware RAG (Retrieval-Augmented Generation) system for analyzing legal contracts. Upload a PDF, ask questions in plain language, and get expert-level answers — with the assistant remembering the full context of your conversation.

---

## Features

- **PDF ingestion** — Upload any legal contract PDF directly from the browser
- **Semantic chunking** — Documents are split using embedding-based semantic similarity, not just character count
- **History-aware retrieval** — Follow-up questions are automatically rewritten into standalone searchable queries
- **Legal analyst persona** — The LLM is prompted to flag risky, unfair, or dangerous clauses and explain them in simple language
- **Dark-themed web UI** — Clean Flask + HTML interface, no frontend framework required
- **CLI mode** — Includes a terminal-based chatbot for quick testing

---

## Project Structure

```
legal/
├── web_app.py              # Flask backend (main app)
├── history_aware_rag.py    # CLI chatbot (terminal mode)
├── ingestion_pipeline.py   # Batch PDF ingestion script
├── static/
│   ├── index.html          # Frontend web UI
│   └── favicon.png         # App icon
├── docs/                   # (optional) Put PDFs here for batch ingestion
├── db/                     # ChromaDB vector store (auto-created)
├── requirements.txt
└── .env                    # API keys (not committed)
```

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace — `BAAI/bge-base-en` |
| Vector Store | ChromaDB (local) |
| PDF Parsing | PyMuPDF (`fitz`) |
| Semantic Chunking | `langchain-experimental` SemanticChunker |
| Web Framework | Flask |
| Orchestration | LangChain Core |

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
2. Click **"Process PDF"** — the document will be chunked and indexed
3. Type your question in the chat and press **Send**

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

## How It Works

```
User Question
      │
      ▼
[History-Aware Rewriter]  ← rewrites follow-up questions into standalone queries
      │
      ▼
[ChromaDB Retriever]      ← finds top-6 semantically similar chunks
      │
      ▼
[Legal Analyst LLM]       ← answers with context, flags risky clauses
      │
      ▼
Answer + updated chat history
```

---

## Notes

- The first run will download the HuggingFace embedding model (~430MB), subsequent runs use the cached version
- ChromaDB data is stored locally in `db/cdb/` — delete this folder to reset the vector store
- The web app processes the PDF in-memory; no files are saved to disk permanently
