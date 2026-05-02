import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load documents from the specified directory."""

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} not found.")
    
    loader = DirectoryLoader(path=docs_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in {docs_path}")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}: ")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents


def get_embedding_model():
    """Initialize and return the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def chunk_text(documents, embedding_model):
    """Chunk text semantically using embeddings."""

    text_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",  # splits where similarity drops most
        breakpoint_threshold_amount=70           
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\nSemantic chunking: {len(documents)} doc(s) → {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Length: {len(chunk.page_content)} characters")
        print(f"  Preview: {chunk.page_content[:100]}...")

    return chunks


def embedding_chunks(chunks, embedding_model, persist_directory="db/cdb"):
    """Store chunks in ChromaDB using the provided embedding model."""

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("\nDocuments embedded and stored in ChromaDB successfully")

    return vector_store


def main():

    # load the files
    documents = load_documents(docs_path="docs")

    # single embedding model shared across chunker + vector store
    embedding_model = get_embedding_model()

    # semantic chunking
    chunks = chunk_text(documents, embedding_model)

    # embed and store
    vector_store = embedding_chunks(chunks, embedding_model)


if __name__ == "__main__":
    main()