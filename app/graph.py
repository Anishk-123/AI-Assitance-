"""LangGraph workflow with answer generation + hallucination (grounding) check.

retrieve -> (no docs?) -> fallback
         -> generate -> check_grounding -> grounded?      -> END
                                        -> not grounded   -> generate (1 retry, stricter)
                                        -> still bad      -> fallback
"""
import os
from functools import lru_cache
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from app.config import GROQ_MODEL
from app.rag import get_retriever

NO_ANSWER = "I could not find this information in the uploaded documents."
MAX_ATTEMPTS = 2


class State(TypedDict, total=False):
    question: str
    document_id: str
    docs: List[Document]
    answer: str
    grounded: bool
    reason: str
    attempts: int
    fallback: bool


class GroundingResult(BaseModel):
    grounded: bool = Field(description="True only if every claim in the answer is supported by the context")
    reason: str = Field(description="One short sentence explaining the verdict")


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env and restart the API.")
    return ChatGroq(model=GROQ_MODEL, temperature=0)

# ---------- Prompt engineering ----------
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful document assistant.\n"
     "Rules:\n"
     "1. Answer ONLY using the context below. Do not use outside knowledge.\n"
     "2. If the context does not contain the answer, reply exactly: \"{no_answer}\"\n"
     "3. Be concise and cite sources as [1], [2] matching the context numbers.\n"
     "{extra}\n\nContext:\n{context}"),
    ("human", "{question}"),
])

GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker. Given a CONTEXT and an ANSWER, decide if the answer "
     "is fully supported by the context. An answer saying the information was not found "
     "is considered grounded."),
    ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}"),
])


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source', '?')}) {d.page_content}"
        for i, d in enumerate(docs, 1)
    )


# ---------- Graph nodes ----------
def retrieve(state: State) -> State:
    docs = get_retriever(document_id=state.get("document_id")).invoke(state["question"])
    return {"docs": docs, "attempts": 0}


def generate(state: State) -> State:
    attempts = state.get("attempts", 0)
    extra = "" if attempts == 0 else (
        "IMPORTANT: Your previous answer contained unsupported claims. "
        "Only state facts that appear word-for-word in the context."
    )
    msg = ANSWER_PROMPT.format_messages(
        no_answer=NO_ANSWER, extra=extra,
        context=format_context(state["docs"]), question=state["question"],
    )
    answer = get_llm().invoke(msg).content.strip()
    return {"answer": answer, "attempts": attempts + 1}


def check_grounding(state: State) -> State:
    """Hallucination reduction: an LLM judge verifies the answer against the context."""
    checker = get_llm().with_structured_output(GroundingResult)
    result = checker.invoke(GROUNDING_PROMPT.format_messages(
        context=format_context(state["docs"]),
        question=state["question"], answer=state["answer"],
    ))
    return {
        "grounded": result.grounded,
        "reason": result.reason,
        "fallback": state["answer"].strip() == NO_ANSWER,
    }


def fallback(state: State) -> State:
    if state.get("docs"):
        reason = (
            "No supporting passage for this question was found in the selected document. "
            "Check that the correct module or paper is uploaded, or try a more specific question."
        )
    else:
        reason = (
            "No document passages were retrieved. Upload the document you want to ask about "
            "and make sure it finishes indexing."
        )
    return {"answer": NO_ANSWER, "grounded": True, "fallback": True, "reason": reason}


# ---------- Conditional edges ----------
def after_retrieve(state: State) -> str:
    return "generate" if state["docs"] else "fallback"


def after_check(state: State) -> str:
    if state["grounded"]:
        return "end"
    return "generate" if state["attempts"] < MAX_ATTEMPTS else "fallback"


def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("check_grounding", check_grounding)
    g.add_node("fallback", fallback)
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", after_retrieve,
                            {"generate": "generate", "fallback": "fallback"})
    g.add_edge("generate", "check_grounding")
    g.add_conditional_edges("check_grounding", after_check,
                            {"end": END, "generate": "generate", "fallback": "fallback"})
    g.add_edge("fallback", END)
    return g.compile()


rag_graph = build_graph()


def ask(question: str, document_id: str | None = None) -> dict:
    out = rag_graph.invoke({"question": question, "document_id": document_id})
    sources, seen = [], set()
    for d in out.get("docs", []):
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": d.metadata.get("source"),
                            "page": d.metadata.get("page"),
                            "snippet": d.page_content[:200]})
    return {
        "answer": out["answer"],
        "grounded": out.get("grounded", True),
        "fallback": out.get("fallback", False),
        "reason": out.get("reason", ""),
        "sources": [] if out.get("fallback") else sources,
    }
