# AI Research & Document Intelligence Assistant

Upload a research paper or document from your device (PDF / DOCX / TXT / MD), ask questions about it, and get **grounded answers with sources**.
Built as the GenAI capstone project.

## Features
| Requirement | Where |
|---|---|
| RAG (load, chunk, embed, Chroma vector DB, retrieve) | `app/rag.py` |
| LangChain (loaders, splitters, prompts, LLM, embeddings) | `app/rag.py`, `app/graph.py` |
| LangGraph (state, nodes, conditional edges, retry) | `app/graph.py` |
| Prompt engineering (strict, context-only, citations) | `ANSWER_PROMPT` in `app/graph.py` |
| Hallucination reduction (LLM grounding check + retry + fallback) | `check_grounding` node |
| LangSmith tracing | enabled via `.env` |
| FastAPI + LangServe | `app/api.py` |
| Streamlit UI | `streamlit_app.py` |
| Evaluation (dataset + metrics) | `evaluation/` |

## Architecture
```
Streamlit UI  --->  FastAPI (/upload, /ask, /rag via LangServe)
                         |
                    LangGraph workflow
   retrieve -> generate -> check_grounding --grounded--> answer
       |                        |  not grounded
       | no docs                +--> retry generate (stricter prompt) --> fallback if still bad
       +--> fallback ("not found in documents")
                         |
              Chroma vector DB (local embeddings)
```

## Setup & Run (step by step)

1. **Install Python 3.10, 3.11 or 3.12** and open a terminal in this folder.
2. **Create a virtual environment**
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Add your keys**: copy `.env.example` to `.env` and set `GROQ_API_KEY` using a key from https://console.groq.com/keys.
   Chroma uses a local embedding model; the first run downloads it. `LANGSMITH_API_KEY` is optional for tracing.
5. **Start the API** (Terminal 1): `uvicorn app.api:app --reload --port 8000`
   - Swagger docs: http://localhost:8000/docs
   - LangServe playground: http://localhost:8000/rag/playground
6. **Start the UI** (Terminal 2, venv activated): `streamlit run streamlit_app.py`
   - Opens at http://localhost:8501
7. **Use it**: upload a research paper or `data/acme_handbook.txt` in the sidebar and click *Index document*. Questions default to the latest upload; choose *All documents* to search the full index.
8. **Run evaluation** (Terminal 3, API need not be running): `python -m evaluation.evaluate`
   Results are printed and saved to `evaluation/results.json`.

## Evaluation
`evaluation/eval_dataset.json` has 13 questions (10 answerable, 3 not in the document).
Metrics: answer accuracy, refusal accuracy (hallucination test), retrieval hit rate, groundedness rate.

## Project structure
```
app/            config, rag pipeline, langgraph workflow, fastapi+langserve
data/           sample document
evaluation/     dataset + evaluate.py
screenshots/    add your screenshots here
streamlit_app.py
```

## Tech stack
Python, LangChain, LangGraph, LangSmith, LangServe, FastAPI, ChromaDB, Groq, Streamlit
