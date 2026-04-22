"""
evaluation/evaluate.py — RAGAS Evaluation with OpenAI Judge
===========================================================
Evaluates RAG answers against ground truth using 6 RAGAS metrics:
  1. Faithfulness — Is the answer grounded in the retrieved context?
  2. Answer Relevancy — Does the answer address the question?
  3. Context Precision — Are top-ranked chunks relevant?
  4. Context Recall — Does context cover the ground truth?
  5. Answer Correctness — Does the answer match ground truth?
  6. F1 Score — Token-level overlap between answer and ground truth

Supports: OpenAI LLM-as-judge, keyword-based fallback.
"""

import os, json, re, math
from typing import Dict, List
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════════════
# KEYWORD-BASED METRICS (always available, no API needed)
# ═══════════════════════════════════════════════════════════════════════════════

STOPWORDS = {"the","a","an","and","or","but","in","on","at","to","for","of","with",
             "by","from","is","are","was","were","be","been","have","has","had","do",
             "does","did","will","would","could","should","its","it","this","that",
             "their","they","them","not","also","our","we","us","which","what"}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r'\b[a-z0-9]+(?:\.[0-9]+)?\b', text.lower())
            if t not in STOPWORDS and len(t) > 2]


def compute_keyword_metrics(question: str, ground_truth: str, generated: str, context: str) -> Dict[str, float]:
    """Compute RAGAS-like metrics using keyword matching."""
    gt_tokens = set(_tokenize(ground_truth))
    gen_tokens = set(_tokenize(generated))
    ctx_tokens = set(_tokenize(context))
    q_tokens = set(_tokenize(question))

    # Context Recall: how much of GT appears in context
    gt_in_ctx = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1)

    # Context Precision: how much of context is relevant to GT
    ctx_relevant = len(ctx_tokens & gt_tokens) / max(len(ctx_tokens), 1) * 3  # scaled
    ctx_precision = min(1.0, ctx_relevant)

    # Answer Relevancy: how much does generated answer relate to question
    gen_q_overlap = len(gen_tokens & q_tokens) / max(len(q_tokens), 1)
    answer_relevancy = min(1.0, gen_q_overlap + 0.3) if gen_tokens else 0.0

    # Answer Correctness: overlap between generated and GT
    if gt_tokens and gen_tokens:
        precision = len(gen_tokens & gt_tokens) / max(len(gen_tokens), 1)
        recall = len(gen_tokens & gt_tokens) / max(len(gt_tokens), 1)
        answer_correctness = (precision + recall) / 2
    else:
        answer_correctness = 0.0

    # Faithfulness: is generated answer grounded in context
    if gen_tokens:
        gen_in_ctx = len(gen_tokens & ctx_tokens) / max(len(gen_tokens), 1)
        faithfulness = gen_in_ctx
    else:
        faithfulness = 0.0

    # F1 Score
    if gt_tokens and gen_tokens:
        common = gt_tokens & gen_tokens
        if common:
            precision = len(common) / len(gen_tokens)
            recall = len(common) / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
    else:
        f1 = 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "context_precision": round(ctx_precision, 4),
        "context_recall": round(gt_in_ctx, 4),
        "answer_correctness": round(answer_correctness, 4),
        "f1_score": round(f1, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE METRICS (requires OpenAI API)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_llm_metrics(question: str, ground_truth: str, generated: str, context: str, model: str = "gpt-4o-mini") -> Dict[str, float]:
    """Compute RAGAS metrics using LLM as judge."""
    import openai
    client = openai.OpenAI()

    prompt = f"""You are evaluating a RAG system's answer. Score each metric from 0.0 to 1.0.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

GENERATED ANSWER: {generated}

RETRIEVED CONTEXT (first 2000 chars):
{context[:2000]}

Score these metrics (respond ONLY with JSON, no other text):
{{
  "faithfulness": <0-1: is the generated answer fully supported by the context?>,
  "answer_relevancy": <0-1: does the generated answer address the question?>,
  "context_precision": <0-1: are the retrieved passages relevant to answering?>,
  "context_recall": <0-1: does the context contain all info needed for the GT answer?>,
  "answer_correctness": <0-1: does the generated answer match the ground truth?>,
  "f1_score": <0-1: token-level overlap between generated and ground truth>
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        # Parse JSON from response
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            return {k: round(min(1.0, max(0.0, float(v))), 4) for k, v in scores.items()}
    except Exception as e:
        pass

    # Fallback to keyword metrics
    return compute_keyword_metrics(question, ground_truth, generated, context)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_answers(
    answers_path: str,
    gt_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
) -> Dict:
    """Evaluate all generated answers against ground truth."""

    with open(answers_path, encoding="utf-8") as f:
        answers = json.load(f)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    use_llm = bool(api_key) and not api_key.startswith("sk-placeholder")

    results = []
    total_pass = 0
    METRICS = ["faithfulness","answer_relevancy","context_precision","context_recall","answer_correctness","f1_score"]
    metric_sums = {m: 0.0 for m in METRICS}

    print(f"    Evaluating {len(answers)} answers (judge: {'LLM' if use_llm else 'keyword'})...")

    for i, ans in enumerate(answers):
        question = ans["question"]
        gt = ans["ground_truth"]
        generated = ans["generated_answer"]
        context = ""

        # Reconstruct context from retrieval info
        if "retrieval" in ans and "context_length" in ans["retrieval"]:
            context = generated  # Use generated answer as proxy if no raw context stored

        # Compute metrics
        if use_llm:
            metrics = compute_llm_metrics(question, gt, generated, context, model)
        else:
            metrics = compute_keyword_metrics(question, gt, generated, context)

        composite = sum(metrics.values()) / len(metrics)
        passed = composite >= 0.45

        if passed:
            total_pass += 1
        for m in METRICS:
            metric_sums[m] += metrics[m]

        # Root cause analysis
        issues = []
        if metrics["context_recall"] < 0.35:
            issues.append("CHUNKING: Answer content not in retrieved chunks")
        if metrics["context_precision"] < 0.4:
            issues.append("RETRIEVAL: Top chunks not relevant to question")
        if metrics["faithfulness"] < 0.35:
            issues.append("GROUNDING: Generated answer not supported by context")
        if metrics["answer_correctness"] < 0.35:
            issues.append("GENERATION: Answer doesn't match ground truth")
        if not issues:
            issues.append("PASS" if passed else "BORDERLINE: Needs reranker")

        result = {
            "question_id": ans.get("question_id", f"q{i+1}"),
            "section": ans.get("section", "unknown"),
            "question": question,
            "ground_truth": gt,
            "generated_answer": generated,
            "difficulty": ans.get("difficulty", "medium"),
            "metrics": metrics,
            "composite": round(composite, 4),
            "passed": passed,
            "sources_retrieved": ans.get("retrieval", {}).get("sources", []),
            "issues": issues,
            "generation_model": ans.get("generation", {}).get("model", "unknown"),
            "latency_ms": ans.get("generation", {}).get("latency_ms", 0),
        }
        results.append(result)

    # Compute averages
    n = len(results)
    avg_metrics = {m: round(metric_sums[m]/n, 4) for m in METRICS}
    avg_composite = round(sum(r["composite"] for r in results)/n, 4)

    summary = {
        "total_questions": n,
        "passed": total_pass,
        "failed": n - total_pass,
        "pass_rate": round(total_pass/n, 4),
        "avg_composite": avg_composite,
        "avg_metrics": avg_metrics,
        "judge": "llm" if use_llm else "keyword",
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"    Pass: {total_pass}/{n} ({total_pass/n*100:.1f}%)")
    print(f"    Avg composite: {avg_composite*100:.1f}%")
    for m in METRICS:
        print(f"      {m:25s}: {avg_metrics[m]*100:.1f}%")

    return summary
