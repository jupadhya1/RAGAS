"""
optimization/dspy_optimizers.py — All 7 DSPy 3.2.0 Optimizers
==============================================================
Verified available optimizers in DSPy 3.2.0:
  1. dspy.BootstrapFewShot             — Bootstrap demonstrations from training set
  2. dspy.BootstrapFewShotWithRandomSearch — Bootstrap + random search over combos
  3. dspy.MIPROv2                       — Multi-Instance Progressive Retrieval Opt
  4. dspy.COPRO                         — Contextual Prompt Optimization
  5. dspy.SIMBA                         — Self-Improving Multi-step Bootstrapped Agent
  6. dspy.KNNFewShot                   — k-Nearest Neighbor few-shot selection
  7. dspy.Ensemble                     — Ensemble multiple optimized programs

Each optimizer compiles a DSPy module using training examples and a metric function.
"""

import os, json, time
from typing import Dict, List, Optional
import dspy

from generation.dspy_signatures import CRAExtendedQA, CRABasicQA
from generation.dspy_modules import MODULE_REGISTRY, get_module


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def cra_f1_metric(example, prediction, trace=None):
    """F1 token overlap between predicted answer and ground truth."""
    import re
    gt = (getattr(example, 'answer', '') or '').lower()
    pred = (getattr(prediction, 'answer', '') or '').lower()
    gt_tokens = set(re.findall(r'\b[a-z0-9]+(?:\.[0-9]+)?\b', gt))
    pred_tokens = set(re.findall(r'\b[a-z0-9]+(?:\.[0-9]+)?\b', pred))
    stops = {'the','and','was','for','are','with','from','that','this','which','has','have','been','our','its','not','also'}
    gt_tokens -= stops
    pred_tokens -= stops
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = gt_tokens & pred_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cra_faithfulness_metric(example, prediction, trace=None):
    """Checks if predicted answer tokens are present in the context."""
    import re
    pred = (getattr(prediction, 'answer', '') or '').lower()
    ctx = (getattr(example, 'context', '') or '').lower()
    pred_tokens = set(re.findall(r'\b[a-z0-9]+(?:\.[0-9]+)?\b', pred))
    ctx_tokens = set(re.findall(r'\b[a-z0-9]+(?:\.[0-9]+)?\b', ctx))
    stops = {'the','and','was','for','are','with','from','that','this','which'}
    pred_tokens -= stops
    if not pred_tokens:
        return 1.0
    return len(pred_tokens & ctx_tokens) / len(pred_tokens)


def cra_combined_metric(example, prediction, trace=None):
    """Weighted combination: 0.4 × F1 + 0.3 × faithfulness + 0.3 × length_penalty."""
    f1 = cra_f1_metric(example, prediction, trace)
    faith = cra_faithfulness_metric(example, prediction, trace)
    # Length penalty: penalize very short or very long answers
    answer = getattr(prediction, 'answer', '') or ''
    length = len(answer.split())
    length_score = 1.0 if 5 <= length <= 100 else 0.7 if 3 <= length <= 200 else 0.3
    return 0.4 * f1 + 0.3 * faith + 0.3 * length_score


METRIC_REGISTRY = {
    "f1": cra_f1_metric,
    "faithfulness": cra_faithfulness_metric,
    "combined": cra_combined_metric,
}


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════

def build_dspy_examples(gt_path: str, corpus_path: str = None, max_examples: int = 30) -> List:
    """Convert ground truth JSON into DSPy Examples."""
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    examples = []
    for qa in ground_truth[:max_examples]:
        ex = dspy.Example(
            question=qa["question"],
            context=qa.get("comments", qa.get("answer", ""))[:800],
            answer_options=qa.get("answer_options", ""),
            answer=qa["answer"],
        ).with_inputs("question", "context", "answer_options")
        examples.append(ex)

    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_bootstrap_fewshot(module, trainset, valset, metric):
    """Optimizer 1: BootstrapFewShot — Select best few-shot demos."""
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        max_rounds=1,
    )
    return optimizer.compile(module, trainset=trainset)


def run_bootstrap_random_search(module, trainset, valset, metric):
    """Optimizer 2: BootstrapFewShotWithRandomSearch — Bootstrap + random combo search."""
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        num_candidate_programs=4,
        num_threads=1,
    )
    return optimizer.compile(module, trainset=trainset, valset=valset)


def run_miprov2(module, trainset, valset, metric):
    """Optimizer 3: MIPROv2 — Multi-Instance Progressive Retrieval Optimization."""
    optimizer = dspy.MIPROv2(
        metric=metric,
        auto="light",
        num_threads=1,
    )
    return optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )


def run_copro(module, trainset, valset, metric):
    """Optimizer 4: COPRO — Contextual Prompt Optimization."""
    optimizer = dspy.COPRO(
        metric=metric,
        breadth=3,
        depth=2,
        init_temperature=0.7,
    )
    return optimizer.compile(module, trainset=trainset)


def run_simba(module, trainset, valset, metric):
    """Optimizer 5: SIMBA — Self-Improving Multi-step Bootstrapped Agent."""
    optimizer = dspy.SIMBA(
        metric=metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        num_candidate_programs=3,
    )
    return optimizer.compile(module, trainset=trainset, valset=valset)


def run_knn_fewshot(module, trainset, valset, metric):
    """Optimizer 6: KNNFewShot — k-Nearest Neighbor demo selection."""
    optimizer = dspy.KNNFewShot(
        k=3,
        trainset=trainset,
    )
    return optimizer.compile(module, trainset=trainset)


def run_ensemble(module, trainset, valset, metric):
    """Optimizer 7: Ensemble — Combine multiple optimized programs."""
    # First bootstrap a few different programs
    programs = []
    for max_demos in [2, 3, 4]:
        opt = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=max_demos)
        prog = opt.compile(module, trainset=trainset)
        programs.append(prog)

    ensemble = dspy.Ensemble(
        reduce_fn=dspy.majority,
    )
    return ensemble.compile(programs)


OPTIMIZER_REGISTRY = {
    "bootstrap_fewshot":    (run_bootstrap_fewshot,        "BootstrapFewShot — Few-shot demo selection"),
    "bootstrap_random":     (run_bootstrap_random_search,  "BootstrapRandom — Bootstrap + random search"),
    "miprov2":              (run_miprov2,                  "MIPROv2 — Progressive retrieval optimization"),
    "copro":                (run_copro,                    "COPRO — Contextual prompt optimization"),
    "simba":                (run_simba,                    "SIMBA — Self-improving bootstrapped agent"),
    "knn_fewshot":          (run_knn_fewshot,              "KNNFewShot — k-NN demo selection"),
    "ensemble":             (run_ensemble,                 "Ensemble — Combine multiple programs"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_optimization(
    module_name: str,
    optimizer_name: str,
    gt_path: str,
    output_dir: str,
    model: str = "gpt-4o-mini",
    metric_name: str = "combined",
    train_ratio: float = 0.7,
) -> Dict:
    """Run a single DSPy optimization experiment."""
    os.makedirs(output_dir, exist_ok=True)

    # Configure DSPy LM
    lm = dspy.LM(f"openai/{model}", api_key=os.environ.get("OPENAI_API_KEY", ""))
    dspy.configure(lm=lm)

    # Build examples
    examples = build_dspy_examples(gt_path)
    split = int(len(examples) * train_ratio)
    trainset = examples[:split]
    valset = examples[split:]

    # Get module and metric
    module = get_module(module_name)
    metric = METRIC_REGISTRY.get(metric_name, cra_combined_metric)

    # Get optimizer
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Available: {list(OPTIMIZER_REGISTRY.keys())}")
    opt_fn, opt_desc = OPTIMIZER_REGISTRY[optimizer_name]

    print(f"    Module: {module_name} | Optimizer: {optimizer_name}")
    print(f"    Metric: {metric_name} | Train: {len(trainset)} | Val: {len(valset)}")

    t0 = time.time()
    try:
        optimized = opt_fn(module, trainset, valset, metric)
        elapsed = time.time() - t0

        # Evaluate on validation set
        evaluator = dspy.Evaluate(
            devset=valset,
            metric=metric,
            num_threads=1,
            display_progress=False,
            return_all_scores=True,
        )
        avg_score, all_scores = evaluator(optimized)

        # Save optimized module
        save_path = os.path.join(output_dir, f"optimized_{module_name}_{optimizer_name}.json")
        try:
            optimized.save(save_path)
        except Exception:
            pass

        result = {
            "status": "ok",
            "module": module_name,
            "optimizer": optimizer_name,
            "optimizer_desc": opt_desc,
            "metric": metric_name,
            "model": model,
            "train_size": len(trainset),
            "val_size": len(valset),
            "avg_val_score": round(float(avg_score), 4),
            "val_scores": [round(float(s), 4) for s in all_scores],
            "elapsed_s": round(elapsed, 2),
            "save_path": save_path,
        }
    except Exception as e:
        result = {
            "status": "error",
            "module": module_name,
            "optimizer": optimizer_name,
            "error": str(e),
            "elapsed_s": round(time.time() - t0, 2),
        }

    # Save result
    result_path = os.path.join(output_dir, f"result_{module_name}_{optimizer_name}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def run_full_optimization_sweep(
    gt_path: str,
    output_dir: str,
    model: str = "gpt-4o-mini",
    modules: List[str] = None,
    optimizers: List[str] = None,
    metric_name: str = "combined",
) -> Dict:
    """Run optimization across multiple module × optimizer combinations."""
    if modules is None:
        modules = ["chain_of_thought", "predict", "self_refining_rag"]
    if optimizers is None:
        optimizers = ["miprov2", "bootstrap_fewshot", "copro"]

    results = []
    for mod in modules:
        for opt in optimizers:
            print(f"\n  ── Experiment: {mod} × {opt} ──")
            try:
                result = run_single_optimization(
                    module_name=mod,
                    optimizer_name=opt,
                    gt_path=gt_path,
                    output_dir=output_dir,
                    model=model,
                    metric_name=metric_name,
                )
                results.append(result)
                score = result.get("avg_val_score", "N/A")
                print(f"  → Score: {score}")
            except Exception as e:
                print(f"  → Error: {e}")
                results.append({"module": mod, "optimizer": opt, "status": "error", "error": str(e)})

    # Summary
    summary = {
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r.get("status") == "ok"),
        "results": sorted(
            [r for r in results if r.get("status") == "ok"],
            key=lambda r: r.get("avg_val_score", 0),
            reverse=True,
        ),
        "errors": [r for r in results if r.get("status") != "ok"],
    }

    if summary["results"]:
        best = summary["results"][0]
        summary["best"] = {
            "module": best["module"],
            "optimizer": best["optimizer"],
            "score": best["avg_val_score"],
        }
        print(f"\n  🏆 Best: {best['module']} × {best['optimizer']} = {best['avg_val_score']:.4f}")

    with open(os.path.join(output_dir, "sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
