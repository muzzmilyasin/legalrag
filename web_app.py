import os
import tempfile
import fitz
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

app = Flask(__name__, static_folder="static")

# Global state
embedding_model = None
db = None
llm = None
bm25_retriever = None
chat_history = []


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return embedding_model


def get_llm():
    global llm
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2
        )
    return llm


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/favicon.png")
def favicon():
    return send_from_directory("static", "favicon.png")


@app.route("/upload", methods=["POST"])
def upload():
    global db, bm25_retriever, chat_history

    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf"]

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    # Extract text with PyMuPDF
    pdf = fitz.open(tmp_path)
    documents = [
        Document(
            page_content=page.get_text(),
            metadata={"source": file.filename, "page": i}
        )
        for i, page in enumerate(pdf)
    ]
    pdf.close()
    os.unlink(tmp_path)

    # Chunk and embed
    embeddings = get_embedding_model()
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embedding_model()
    if db is None:
        # First upload — create a fresh collection
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        # Subsequent uploads — add to the existing collection
        db.add_documents(chunks)

    # Rebuild BM25 from ALL stored texts (all PDFs uploaded so far)
    stored = db.get(include=["documents", "metadatas"])
    bm25_retriever = BM25Retriever.from_texts(
        texts=stored["documents"],
        metadatas=stored["metadatas"]
    )
    bm25_retriever.k = 10

    return jsonify({"message": f"Processed {len(chunks)} chunks from {file.filename}. Total chunks in DB: {len(stored['documents'])}"})


@app.route("/ask", methods=["POST"])
def ask():
    global db, bm25_retriever, chat_history

    if db is None:
        return jsonify({"error": "Please upload a PDF first"}), 400

    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty question"}), 400

    model = get_llm()

    # Rewrite question using chat history
    if chat_history:
        messages = [
            SystemMessage(content="You are a legal assistant. Rewrite the new question to be standalone and searchable. Return only the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New Question: {query}")
        ]
        search_query = model.invoke(messages).content.strip()
    else:
        search_query = query

    # Hybrid retrieval: vector + BM25 ensemble
    vector_retriever = db.as_retriever(search_kwargs={"k": 10})
    if bm25_retriever is not None:
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
    else:
        retriever = vector_retriever
    docs = retriever.invoke(search_query)
    context = "\n\n".join([f"Document:\n{doc.page_content}" for doc in docs])

    answer_messages = [
        SystemMessage(content=(
            "You are an expert legal analyst reviewing contracts on behalf of the user.\n"
            "Your job is to:\n"
            "- Answer greetings from the user, but politely refuse if asked for personal information.\n"
            "- Identify and explain any clauses that could be risky, unfair, or dangerous to the Borrower.\n"
            "- Answer questions using the provided contract excerpts.\n"
            "- If a clause sounds legal but is actually harmful, flag it and explain why.\n"
            "- Tell the term number and name if available.\n"
            "- IMPORTANT: Format your response beautifully using markdown. Use **bold text** for key terms, use bullet points, and write in short, readable paragraphs. Do NOT output a single massive paragraph.\n"
            "- Explain in simple language.\n"
            "- Keep it brief.\n"
            "- If the answer is not in the context, say: 'This information is not available in the provided documents.'\n\n"
            "Contract context:\n" + context
        )),
        HumanMessage(content=search_query)
    ]
    answer = model.invoke(answer_messages).content

    # Save to history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    return jsonify({"answer": answer})


@app.route("/reset", methods=["POST"])
def reset():
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared"})


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=False, port=5000)
