"""FastAPI + LangServe API.

Run:  uvicorn app.api:app --reload --port 8000
Docs: http://localhost:8000/docs      LangServe playground: http://localhost:8000/rag/playground
"""
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from groq import RateLimitError
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from app.config import UPLOAD_DIR
from app.rag import ingest, reset_store
from app.graph import ask

app = FastAPI(title="AI Research & Document Intelligence Assistant")
AI_QUOTA_ERROR = (
    "Groq API rate or quota limit reached. Check your Groq account and model limits, then retry."
)


class AskRequest(BaseModel):
    question: str
    document_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, os.path.basename(file.filename))
    document_id = uuid.uuid4().hex
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        n = ingest(path, document_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RateLimitError as e:
        raise HTTPException(status_code=503, detail=AI_QUOTA_ERROR) from e
    return {"filename": file.filename, "document_id": document_id, "chunks_indexed": n}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    try:
        return ask(req.question, req.document_id)
    except RateLimitError as e:
        raise HTTPException(status_code=503, detail=AI_QUOTA_ERROR) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/reset")
def reset():
    reset_store()
    return {"status": "vector store cleared"}


# LangServe route (adds /rag/invoke, /rag/stream, /rag/playground ...)
def _rag(x):
    q = x["question"] if isinstance(x, dict) else x.question
    document_id = x.get("document_id") if isinstance(x, dict) else x.document_id
    return ask(q, document_id)


add_routes(app, RunnableLambda(_rag).with_types(input_type=AskRequest, output_type=dict), path="/rag")
