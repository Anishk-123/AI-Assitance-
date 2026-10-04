"""Evaluation script.

Metrics:
  - Answer accuracy       : expected keyword appears in the answer
  - Refusal accuracy      : out-of-document questions are correctly refused (no hallucination)
  - Retrieval hit rate    : retrieved chunks contain the expected keyword
  - Groundedness rate     : % of answers passed by the grounding checker

Run from project root:  python -m evaluation.evaluate
All runs are traced to LangSmith automatically if LANGSMITH_* is set in .env
"""
import json
import os
from app.rag import reset_store, ingest, get_retriever
from app.graph import ask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "data", "acme_handbook.txt")
DATASET = os.path.join(ROOT, "evaluation", "eval_dataset.json")
OUT = os.path.join(ROOT, "evaluation", "results.json")


def main():
    print("Indexing sample document...")
    reset_store()
    ingest(DOC)
    data = json.load(open(DATASET, encoding="utf-8"))

    rows = []
    for item in data:
        res = ask(item["question"])
        ans = res["answer"].lower()
        if item["expect_refusal"]:
            correct = res["fallback"]
            hit = None
        else:
            correct = any(k.lower() in ans for k in item["expected_any"])
            chunks = " ".join(d.page_content.lower() for d in get_retriever().invoke(item["question"]))
            hit = any(k.lower() in chunks for k in item["expected_any"])
        rows.append({"question": item["question"], "answer": res["answer"],
                     "correct": correct, "retrieval_hit": hit, "grounded": res["grounded"],
                     "refusal_case": item["expect_refusal"]})
        print(f"{'PASS' if correct else 'FAIL'} | {item['question']}")

    answerable = [r for r in rows if not r["refusal_case"]]
    refusals = [r for r in rows if r["refusal_case"]]
    summary = {
        "answer_accuracy": sum(r["correct"] for r in answerable) / len(answerable),
        "refusal_accuracy": sum(r["correct"] for r in refusals) / len(refusals),
        "retrieval_hit_rate": sum(bool(r["retrieval_hit"]) for r in answerable) / len(answerable),
        "groundedness_rate": sum(r["grounded"] for r in rows) / len(rows),
        "total_questions": len(rows),
    }
    json.dump({"summary": summary, "details": rows}, open(OUT, "w", encoding="utf-8"), indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
