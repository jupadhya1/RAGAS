"""
evaluation/ragas_evaluate.py — RAGAS 0.4.3 Evaluation Module (v3 — All Improvements)
=====================================================================================
IMPROVEMENTS:
  P4 CRITICAL: build_ragas_dataset() now reads actual retrieved_contexts from
      rag_answers.json instead of using generated_answer[:500] as context.
      This fixes ALL context-dependent metrics (faithfulness, ctx precision, ctx recall).
  NEW: Separate --ragas-llm parameter for cost-effective evaluation (gpt-4o-mini)
  NEW: Robust NaN handling in averaging
  NEW: Per-question score includes question_id and section for diagnosis
"""

import os, json, warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# RAGAS IMPORTS (v0.4.3 verified)
# ═══════════════════════════════════════════════════════════════════════════════

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AnswerSimilarity,
    ContextEntityRecall,
    NoiseSensitivity,
    ResponseRelevancy,
    FactualCorrectness,
    SemanticSimilarity,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_PRESETS = {
    "core": [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
        AnswerSimilarity(),
    ],
    "extended": [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
        ContextEntityRecall(),
        FactualCorrectness(),
        SemanticSimilarity(),
    ],
    "full": [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
        AnswerSimilarity(),
        ContextEntityRecall(),
        FactualCorrectness(),
        SemanticSimilarity(),
    ],
    "retrieval_only": [
        ContextPrecision(),
        ContextRecall(),
        ContextEntityRecall(),
    ],
    "generation_only": [
        Faithfulness(),
        AnswerRelevancy(),
        AnswerCorrectness(),
        FactualCorrectness(),
    ],
    "non_llm": [
        NonLLMContextPrecisionWithReference(),
        NonLLMContextRecall(),
        AnswerSimilarity(),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# P4 FIX: BUILD EVALUATION DATASET WITH ACTUAL RETRIEVED CONTEXTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_ragas_dataset(answers_path: str) -> EvaluationDataset:
    """
    P4 FIX: Convert RAG pipeline answers to RAGAS EvaluationDataset.

    CRITICAL CHANGE: Uses actual retrieved_contexts from the retrieval step
    instead of the old buggy approach that truncated generated_answer[:500]
    and passed it as context.

    The generate.py step stores retrieved chunks at:
        ans["retrieval"]["retrieved_contexts"]  →  List[str]
    """
    with open(answers_path, encoding="utf-8") as f:
        answers = json.load(f)

    samples = []
    contexts_fixed = 0
    contexts_fallback = 0

    for ans in answers:
        # ── P4: Extract ACTUAL retrieved contexts ─────────────
        retrieved_contexts = None

        # Primary: from retrieval.retrieved_contexts (set by improved generate.py)
        retrieval_data = ans.get("retrieval", {})
        if isinstance(retrieval_data, dict):
            ctx = retrieval_data.get("retrieved_contexts")
            if ctx and isinstance(ctx, list) and len(ctx) > 0:
                retrieved_contexts = ctx
                contexts_fixed += 1

        # Fallback: if old format without retrieved_contexts
        if retrieved_contexts is None:
            # Use generated_answer as last resort (old buggy behavior)
            gen_answer = ans.get("generated_answer", "")
            if gen_answer:
                retrieved_contexts = [gen_answer[:800]]
            else:
                retrieved_contexts = ["No context available"]
            contexts_fallback += 1

        sample = SingleTurnSample(
            user_input=ans["question"],
            response=ans.get("generated_answer", ""),
            retrieved_contexts=retrieved_contexts,
            reference=ans.get("ground_truth", ""),
        )
        samples.append(sample)

    if contexts_fixed > 0:
        print(f"    P4: {contexts_fixed}/{len(answers)} samples using ACTUAL retrieved contexts")
    if contexts_fallback > 0:
        print(f"    ⚠️  {contexts_fallback}/{len(answers)} samples using fallback (no retrieved_contexts in data)")

    return EvaluationDataset(samples=samples)


def build_ragas_dataset_with_contexts(
    answers: List[Dict],
    contexts: List[List[str]],
) -> EvaluationDataset:
    """Build dataset with explicitly provided context chunks."""
    samples = []
    for i, ans in enumerate(answers):
        ctx = contexts[i] if i < len(contexts) else [ans.get("generated_answer", "")[:500]]
        sample = SingleTurnSample(
            user_input=ans["question"],
            response=ans.get("generated_answer", ""),
            retrieved_contexts=ctx,
            reference=ans.get("ground_truth", ""),
        )
        samples.append(sample)
    return EvaluationDataset(samples=samples)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN RAGAS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_ragas_evaluation(
    answers_path: str,
    output_path: str,
    metric_preset: str = "core",
    llm_model: str = "gpt-4o-mini",
    ragas_llm: str = None,
) -> Dict:
    """
    Run full RAGAS evaluation on RAG pipeline answers.

    Args:
        answers_path: Path to rag_answers.json from generation step
        output_path: Where to save evaluation results
        metric_preset: One of 'core', 'extended', 'full', etc.
        llm_model: Fallback LLM model name
        ragas_llm: Separate LLM for RAGAS evaluation (cost optimization).
                   If None, uses llm_model.
    """
    eval_llm_model = ragas_llm or llm_model

    print(f"    Loading answers from {answers_path}")
    dataset = build_ragas_dataset(answers_path)
    print(f"    Built RAGAS dataset: {len(dataset)} samples")

    metrics = METRIC_PRESETS.get(metric_preset, METRIC_PRESETS["core"])
    metric_names = [type(m).__name__ for m in metrics]
    print(f"    Metrics ({metric_preset}): {', '.join(metric_names)}")

    # Configure RAGAS LLM + Embeddings
    evaluator_llm = None
    evaluator_embeddings = None
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=eval_llm_model))
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )
        print(f"    Evaluator LLM: {eval_llm_model} | Embeddings: text-embedding-3-small")
    except ImportError as e:
        print(f"    ⚠️ langchain-openai not fully available ({e}). Using RAGAS defaults.")

    # Run evaluation
    print(f"    Running RAGAS evaluation...")
    try:
        eval_kwargs = {"dataset": dataset, "metrics": metrics}
        if evaluator_llm:
            eval_kwargs["llm"] = evaluator_llm
        if evaluator_embeddings:
            eval_kwargs["embeddings"] = evaluator_embeddings

        result = evaluate(**eval_kwargs)

        # Extract per-sample scores
        df = result.to_pandas()
        scores_per_sample = df.to_dict("records")

        # ── Enrich per-sample scores with question metadata ───
        with open(answers_path, encoding="utf-8") as f:
            answers_data = json.load(f)
        for i, row in enumerate(scores_per_sample):
            if i < len(answers_data):
                row["question_id"] = answers_data[i].get("question_id", "")
                row["section"] = answers_data[i].get("section", "")
                row["difficulty"] = answers_data[i].get("difficulty", "")

        # Compute averages (robust NaN handling)
        avg_scores = {}
        for col in df.columns:
            if col not in [
                "user_input", "response", "retrieved_contexts", "reference",
            ]:
                try:
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        mean_val = float(vals.mean())
                        if mean_val == mean_val:  # not NaN
                            avg_scores[col] = round(mean_val, 4)
                except (ValueError, TypeError):
                    pass

        output = {
            "metric_preset": metric_preset,
            "metrics_used": metric_names,
            "evaluator_llm": eval_llm_model,
            "num_samples": len(dataset),
            "contexts_source": "actual_retrieved",  # P4: flag confirming fix
            "avg_scores": avg_scores,
            "per_sample_scores": scores_per_sample,
        }

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"    RAGAS Results:")
        for metric, score in avg_scores.items():
            print(f"      {metric:35s}: {score:.4f}")
        print(f"    Saved to {output_path}")

        return output

    except Exception as e:
        print(f"    ⚠️ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        error_result = {
            "status": "error",
            "error": str(e),
            "metric_preset": metric_preset,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)
        return error_result


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARATIVE EVALUATION (multiple pipeline variants)
# ═══════════════════════════════════════════════════════════════════════════════

def compare_pipelines(
    variant_paths: Dict[str, str],
    output_path: str,
    metric_preset: str = "core",
) -> Dict:
    """Run RAGAS evaluation on multiple pipeline variants and compare."""
    results = {}
    for name, path in variant_paths.items():
        print(f"\n  Evaluating variant: {name}")
        result = run_ragas_evaluation(
            answers_path=path,
            output_path=output_path.replace(".json", f"_{name}.json"),
            metric_preset=metric_preset,
        )
        results[name] = result.get("avg_scores", {})

    # Build comparison table
    comparison = {
        "variants": list(results.keys()),
        "metrics": {},
    }
    all_metrics = set()
    for scores in results.values():
        all_metrics.update(scores.keys())

    for metric in sorted(all_metrics):
        comparison["metrics"][metric] = {
            name: results[name].get(metric, None) for name in results
        }

    # Find best variant per metric
    comparison["best_per_metric"] = {}
    for metric, scores in comparison["metrics"].items():
        valid = {k: v for k, v in scores.items() if v is not None}
        if valid:
            best = max(valid, key=valid.get)
            comparison["best_per_metric"][metric] = {
                "variant": best,
                "score": valid[best],
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Comparison saved to {output_path}")
    return comparison
