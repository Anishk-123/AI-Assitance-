"""RAG pipeline: load -> chunk -> embed -> store in Chroma -> retrieve."""
import os
import re
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from app.config import CHROMA_DIR

COLLECTION = "documents_groq"


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=None,
        persist_directory=CHROMA_DIR,
    )


def load_document(path: str, document_id: str | None = None):
    """Step 1: Document loading (PDF, DOCX, TXT/MD)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = os.path.basename(path)
        if document_id:
            d.metadata["document_id"] = document_id
    return docs


def chunk_documents(docs):
    """Step 2: Chunking."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(docs)


def ingest(path: str, document_id: str | None = None) -> int:
    """Steps 3-4: Embed chunks and store them in the vector DB."""
    chunks = chunk_documents(load_document(path, document_id))
    get_vectorstore().add_documents(chunks)
    return len(chunks)


def reset_store():
    try:
        get_vectorstore().delete_collection()
    except Exception:
        pass


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def get_retriever(k: int = 4, document_id: str | None = None):
    """Combine semantic and exact-keyword retrieval within the requested document scope."""
    vectorstore = get_vectorstore()
    search_kwargs = {"k": k}
    if document_id:
        search_kwargs["filter"] = {"document_id": document_id}
    semantic_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    where = {"document_id": document_id} if document_id else None
    stored = vectorstore.get(where=where, include=["documents", "metadatas"]) if where else (
        vectorstore.get(include=["documents", "metadatas"])
    )
    keyword_documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(stored["documents"], stored["metadatas"])
    ]
    if not keyword_documents:
        return semantic_retriever

    keyword_retriever = BM25Retriever.from_documents(
        keyword_documents, k=k, preprocess_func=_tokenize
    )
    return EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.5, 0.5],
    )
