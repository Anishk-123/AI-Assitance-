# Project Report: AI Research & Document Intelligence Assistant

**Name:** ______  **Batch:** genai-fast-track  **Date:** ______

## 1. Problem Statement
LLMs can hallucinate. This project lets users upload documents and ask questions, returning answers grounded in the document with cited sources.

## 2. Architecture
(Paste the diagram from README.md and a screenshot of the LangGraph flow.)

## 3. Components
- **RAG pipeline:** loaders (PDF/DOCX/TXT) -> RecursiveCharacterTextSplitter (800 chars, 120 overlap) -> local Chroma embeddings -> top-4 retrieval; Groq generates and checks answers.
- **LangGraph workflow:** nodes retrieve, generate, check_grounding, fallback; conditional edges for empty retrieval and failed grounding (one retry).
- **Prompt engineering:** context-only rules, exact "not found" reply, citations.
- **Hallucination reduction:** LLM judge verifies each answer against context; unsupported answers are regenerated or replaced by a safe fallback.
- **API:** FastAPI endpoints `/upload`, `/ask`, `/reset`, plus LangServe route `/rag`.
- **UI:** Streamlit chat with upload, grounded badge and source viewer.
- **LangSmith:** every run is traced (add trace screenshots).

## 4. Evaluation
Dataset: 13 questions. Paste the summary from `evaluation/results.json`:

| Metric | Score |
|---|---|
| Answer accuracy | |
| Refusal accuracy | |
| Retrieval hit rate | |
| Groundedness rate | |

## 5. Screenshots
(Streamlit UI, FastAPI /docs, LangServe playground, LangSmith traces, evaluation output.)

## 6. Challenges and Learnings
(Write 3 to 4 points in your own words.)

## 7. Future Improvements
Hybrid search, reranking, multi-document filters, conversation memory, RAGAS metrics.
