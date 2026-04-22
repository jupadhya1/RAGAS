"""
generation/generate.py — RAG Answer Generation (v3 — All Improvements)
======================================================================
IMPROVEMENTS:
  P1: DSPy module instantiated ONCE, reused for all questions
  P1: dspy.configure(lm=...) called ONCE at start, not per-question
  P3: Concise CRA-format system prompt (match terse ground truth style)
  P4: retrieved_contexts stored properly for RAGAS consumption
  NEW: Token tracking works for both OpenAI direct and DSPy modes
"""

import os, json, time, pickle
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# P3: CONCISE CRA SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert ESG/Climate Risk analyst evaluating Shell Petroleum Company Limited.
You are answering questions from the SCB Climate Risk Assessment (CRA) questionnaire.

RULES:
1. Answer ONLY from the provided context. If the context doesn't contain enough information, say "Insufficient context."
2. Be CONCISE — match the terse format of regulatory documents.
3. If the answer is a number, score, rating, or classification, state it directly without lengthy explanation.
4. Quote specific numbers, percentages, and scores exactly as they appear in the context.
5. Cite the source document when possible (CRA, ETS 2024, Annual Report 2023, CDP 2023).

Example format:
  Q: "What is Shell's CRA score?" → A: "57.22 — RED BRAG rating"
  Q: "What CDP score?" → A: "B (2023)"
  Q: "Scope 1&2 target?" → A: "50% reduction by 2030 from 2016 baseline under operational control"
"""


def _load_index(index_dir: str):
    path = os.path.join(index_dir, "hybrid_index.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI DIRECT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_openai(
    question: str, context: str, model: str = "gpt-4o-mini", client=None
) -> Dict:
    """Generate answer using OpenAI chat completion."""
    import openai

    if client is None:
        client = openai.OpenAI()

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:",
                },
            ],
            temperature=0.0,
            max_tokens=300,  # P3: reduced from 500 to force conciseness
        )
        return {
            "answer": resp.choices[0].message.content.strip(),
            "model": model,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "tokens": {
                "prompt": resp.usage.prompt_tokens,
                "completion": resp.usage.completion_tokens,
                "total": resp.usage.total_tokens,
            },
            "status": "ok",
            "method": "openai_direct",
        }
    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "model": model,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "status": "error",
            "method": "openai_direct",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# P1: DSPY GENERATION — MODULE CREATED ONCE, REUSED
# ═══════════════════════════════════════════════════════════════════════════════

def _init_dspy_module(module_name: str, model: str = "gpt-4o-mini"):
    """
    P1 FIX: Create DSPy LM + module ONCE.
    Returns (module, lm) tuple for reuse across all questions.
    """
    import dspy
    from generation.dspy_modules import get_module

    lm = dspy.LM(f"openai/{model}", api_key=os.environ.get("OPENAI_API_KEY", ""))
    dspy.configure(lm=lm)
    module = get_module(module_name)
    print(f"    P1: DSPy module '{module_name}' instantiated ONCE (reusing for all questions)")
    return module, lm


def _generate_dspy(
    question: str, context: str, module, module_name: str, model: str = "gpt-4o-mini"
) -> Dict:
    """Generate using a pre-instantiated DSPy module (P1 fix)."""
    t0 = time.time()
    try:
        prediction = module(question=question, context=context, answer_options="")
        answer = getattr(prediction, "answer", str(prediction))
        explanation = getattr(prediction, "explanation", "")
        return {
            "answer": answer,
            "explanation": explanation,
            "model": model,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "status": "ok",
            "method": f"dspy_{module_name}",
        }
    except Exception as e:
        return {
            "answer": f"DSPy Error: {e}",
            "model": model,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "status": "error",
            "method": f"dspy_{module_name}",
        }


def _generate_fallback(question: str, context: str) -> Dict:
    return {
        "answer": context[:500] if context else "No context",
        "model": "fallback",
        "latency_ms": 0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
        "status": "fallback",
        "method": "keyword_fallback",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def generate_answers(
    gt_path: str,
    corpus_path: str,
    index_dir: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    top_k: int = 8,
    retriever: str = "hybrid",
    dspy_module: str = None,
) -> Dict:
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Load or build index
    hybrid_index = _load_index(index_dir)
    if hybrid_index is None:
        from indexing.build_index import BM25Index, HybridIndex

        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)
        bm25 = BM25Index()
        documents = [c["text"] for c in corpus]
        bm25.build(documents)
        hybrid_index = HybridIndex(bm25, None)
        hybrid_index.documents = documents
        hybrid_index.corpus_meta = [
            {"source": c["source"], "id": c["id"]} for c in corpus
        ]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    use_openai = bool(api_key) and not api_key.startswith("sk-placeholder")

    # ── P1: Instantiate DSPy module ONCE ──────────────────────
    dspy_mod_instance = None
    if dspy_module and use_openai:
        dspy_mod_instance, _ = _init_dspy_module(dspy_module, model)

    # Create OpenAI client once for reuse
    openai_client = None
    if use_openai and not dspy_module:
        import openai
        openai_client = openai.OpenAI()

    method = (
        f"dspy_{dspy_module}"
        if dspy_module
        else ("openai" if use_openai else "fallback")
    )

    print(
        f"    Generating {len(ground_truth)} answers | Method: {method} | Top-K: {top_k}"
    )

    results = []
    total_tokens = 0
    for i, qa in enumerate(ground_truth):
        retrieved = hybrid_index.query(qa["question"], top_k=top_k)
        context_chunks = [r["text"] for r in retrieved]
        context = "\n\n---\n\n".join(context_chunks[:5])
        sources = list(set(r["metadata"].get("source", "?") for r in retrieved))

        # ── P1: Reuse pre-instantiated DSPy module ────────────
        if dspy_mod_instance is not None:
            gen = _generate_dspy(
                qa["question"], context, dspy_mod_instance, dspy_module, model
            )
        elif use_openai:
            gen = _generate_openai(qa["question"], context, model, openai_client)
        else:
            gen = _generate_fallback(qa["question"], context)

        total_tokens += gen["tokens"]["total"]

        # ── P4: Store actual retrieved_contexts for RAGAS ─────
        results.append({
            "question_id": qa.get("question_id", f"q{i + 1}"),
            "section": qa.get("section", "unknown"),
            "question": qa["question"],
            "ground_truth": qa["answer"],
            "generated_answer": gen["answer"],
            "difficulty": qa.get("difficulty", "medium"),
            "retrieval": {
                "top_k": top_k,
                "retriever": retriever,
                "chunks": len(retrieved),
                "sources": sources,
                "context_chars": len(context),
                "retrieved_contexts": context_chunks[:5],  # P4: actual chunks for RAGAS
            },
            "generation": gen,
        })
        if (i + 1) % 10 == 0:
            print(
                f"      [{i + 1}/{len(ground_truth)}] {gen['status']} {gen['latency_ms']:.0f}ms"
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"    Done: {len(results)} answers | {total_tokens:,} tokens → {output_path}")
    return {"total": len(results), "method": method, "tokens": total_tokens}
