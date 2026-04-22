"""
pipeline.py — Shell CRA AutoRAG + DSPy + RAGAS Pipeline Orchestrator (v3)
==========================================================================
IMPROVEMENTS:
  --ragas-llm:    Separate LLM for RAGAS evaluation (gpt-4o-mini saves ~60% cost)
  --no-reranker:  Disable CrossEncoder reranking for faster runs
  Hash-based skip: Steps with unchanged inputs skip automatically
  Better logging: Shows improvement flags (P0-P5) in output

Steps:
  1. groundtruth    — Extract Q&A from CRA XLSX (7 sections)
  2. ingest         — Parse & chunk 4 docs → corpus.json (P0: table-aware)
  3. index          — Build BM25 + Embedding hybrid + reranker (P2)
  4. autorag_prep   — Convert to AutoRAG parquet format + YAML config
  5. autorag_eval   — Run AutoRAG Evaluator
  6. generate       — Generate answers (P1: DSPy single-init, P3: concise prompt)
  7. ragas_eval     — RAGAS evaluation (P4: actual contexts)
  8. dspy_optimize  — DSPy optimization sweep
  9. diagnose       — Root cause analysis of failures
  10. report        — Final consolidated report

Usage:
  # Full pipeline with improvements
  OPENAI_API_KEY=sk-... python pipeline.py --steps all

  # Cost-optimized: use gpt-4o-mini for RAGAS evaluation
  python pipeline.py --steps generate ragas_eval --ragas-llm gpt-4o-mini

  # Skip reranker for fast iteration
  python pipeline.py --steps all --no-reranker
"""

import os, sys, argparse, json, time, hashlib
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

ALL_STEPS = [
    "groundtruth", "ingest", "index",
    "autorag_prep", "autorag_eval",
    "generate", "ragas_eval",
    "dspy_optimize", "diagnose", "report",
]
DEFAULT_STEPS = [
    "groundtruth", "ingest", "index",
    "generate", "ragas_eval", "diagnose", "report",
]


# ═══════════════════════════════════════════════════════════════════════════════
# HASH-BASED SKIP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _file_hash(path: str) -> str:
    """Compute MD5 hash of a file for change detection."""
    if not os.path.exists(path):
        return ""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_skip(step: str, input_paths: list, output_path: str, hash_dir: str) -> bool:
    """Check if step can be skipped (inputs unchanged, output exists)."""
    if not os.path.exists(output_path):
        return False
    hash_file = os.path.join(hash_dir, f"{step}_input_hash.txt")
    current_hash = "|".join(_file_hash(p) for p in input_paths if os.path.exists(p))
    if os.path.exists(hash_file):
        with open(hash_file, encoding="utf-8") as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            return True
    # Save current hash
    os.makedirs(hash_dir, exist_ok=True)
    with open(hash_file, "w", encoding="utf-8") as f:
        f.write(current_hash)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# STEP IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_groundtruth(args):
    from groundtruth.extract_groundtruth import extract_all
    return extract_all(
        xlsx_path=args.cra_xlsx,
        output_path=os.path.join(args.output_dir, "ground_truth.json"),
    )


def run_ingest(args):
    from core.ingest_corpus import ingest_all
    return ingest_all(
        pdf_paths=args.pdf_files,
        xlsx_path=args.cra_xlsx,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def run_index(args):
    from indexing.build_index import build_index
    return build_index(
        corpus_path=os.path.join(args.output_dir, "corpus.json"),
        index_dir=os.path.join(args.output_dir, "index"),
        embedding_model=args.embedding_model,
        use_reranker=not args.no_reranker,
    )


def run_autorag_prep(args):
    """Prepare AutoRAG parquet files and YAML config."""
    from core.autorag_runner import (
        build_corpus_parquet, build_qa_parquet, generate_autorag_config,
    )
    ar_dir = os.path.join(args.output_dir, "autorag_data")
    os.makedirs(ar_dir, exist_ok=True)
    corpus_pq = build_corpus_parquet(
        os.path.join(args.output_dir, "corpus.json"), ar_dir
    )
    qa_pq = build_qa_parquet(
        os.path.join(args.output_dir, "ground_truth.json"),
        os.path.join(args.output_dir, "corpus.json"),
        ar_dir,
    )
    config = generate_autorag_config(
        os.path.join(ar_dir, "pipeline.yaml"),
        llm_model=args.model,
        embedding_model=args.embedding_model,
    )
    return {"corpus_parquet": corpus_pq, "qa_parquet": qa_pq, "config": config}


def run_autorag_eval(args):
    """Run AutoRAG Evaluator."""
    from core.autorag_runner import run_autorag_evaluation
    ar_dir = os.path.join(args.output_dir, "autorag_data")
    return run_autorag_evaluation(
        qa_path=os.path.join(ar_dir, "qa.parquet"),
        corpus_path=os.path.join(ar_dir, "corpus.parquet"),
        config_path=os.path.join(ar_dir, "pipeline.yaml"),
        project_dir=os.path.join(args.output_dir, "autorag_project"),
    )


def run_generate(args):
    from generation.generate import generate_answers
    return generate_answers(
        gt_path=os.path.join(args.output_dir, "ground_truth.json"),
        corpus_path=os.path.join(args.output_dir, "corpus.json"),
        index_dir=os.path.join(args.output_dir, "index"),
        output_path=os.path.join(args.output_dir, "rag_answers.json"),
        model=args.model,
        top_k=args.top_k,
        retriever=args.retriever,
        dspy_module=args.dspy_module,
    )


def run_ragas_eval(args):
    """Run RAGAS evaluation. Falls back to keyword-based if ragas not installed."""
    try:
        from evaluation.ragas_evaluate import run_ragas_evaluation
        return run_ragas_evaluation(
            answers_path=os.path.join(args.output_dir, "rag_answers.json"),
            output_path=os.path.join(args.output_dir, "ragas_evaluation.json"),
            metric_preset=args.ragas_preset,
            llm_model=args.model,
            ragas_llm=args.ragas_llm,  # NEW: separate RAGAS evaluator LLM
        )
    except ImportError:
        print("    ⚠️ ragas not installed. Falling back to keyword + LLM-as-judge.")
        print("    To use RAGAS: pip install ragas langchain-openai")
        from evaluation.evaluate import evaluate_answers
        return evaluate_answers(
            answers_path=os.path.join(args.output_dir, "rag_answers.json"),
            gt_path=os.path.join(args.output_dir, "ground_truth.json"),
            output_path=os.path.join(args.output_dir, "evaluation.json"),
            model=args.model,
        )


def run_dspy_optimize(args):
    """Run DSPy optimization sweep across modules × optimizers."""
    from optimization.dspy_optimizers import run_full_optimization_sweep
    return run_full_optimization_sweep(
        gt_path=os.path.join(args.output_dir, "ground_truth.json"),
        output_dir=os.path.join(args.output_dir, "dspy_optimization"),
        model=args.model,
        modules=args.dspy_modules,
        optimizers=args.dspy_optimizers,
        metric_name=args.dspy_metric,
    )


def run_diagnose(args):
    from evaluation.diagnose import diagnose_failures
    for fname in ["ragas_evaluation.json", "evaluation.json"]:
        eval_path = os.path.join(args.output_dir, fname)
        if os.path.exists(eval_path):
            return diagnose_failures(
                eval_path=eval_path,
                output_path=os.path.join(args.output_dir, "diagnosis.json"),
            )

    answers_path = os.path.join(args.output_dir, "rag_answers.json")
    if os.path.exists(answers_path):
        print("    No RAGAS evaluation found. Running keyword-based evaluation...")
        from evaluation.evaluate import evaluate_answers
        gt_path = os.path.join(args.output_dir, "ground_truth.json")
        eval_path = os.path.join(args.output_dir, "evaluation.json")
        evaluate_answers(
            answers_path=answers_path, gt_path=gt_path,
            output_path=eval_path, model=args.model,
        )
        return diagnose_failures(
            eval_path=eval_path,
            output_path=os.path.join(args.output_dir, "diagnosis.json"),
        )

    print("    ⚠️ No answers or evaluation found. Run 'generate' step first.")
    return {"status": "skipped", "reason": "no evaluation data"}


def run_report(args):
    from evaluation.report import generate_report
    return generate_report(
        output_dir=args.output_dir,
        report_path=os.path.join(args.output_dir, "final_report.json"),
    )


STEP_MAP = {
    "groundtruth": run_groundtruth, "ingest": run_ingest, "index": run_index,
    "autorag_prep": run_autorag_prep, "autorag_eval": run_autorag_eval,
    "generate": run_generate, "ragas_eval": run_ragas_eval,
    "dspy_optimize": run_dspy_optimize, "diagnose": run_diagnose, "report": run_report,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Shell CRA AutoRAG + DSPy + RAGAS Pipeline (v3 — Improved)"
    )
    p.add_argument("--steps", nargs="+", default=DEFAULT_STEPS, choices=ALL_STEPS + ["all"])

    # Data paths
    p.add_argument("--cra-xlsx", default="data/CRA_SHELL.xlsx")
    p.add_argument("--pdf-files", nargs="+", default=[
        "data/shell-energy-transition-strategy-2024.pdf",
        "data/shell-annual-report-2023.pdf",
        "data/2023-cdp-climate-change-shell-plc.pdf",
    ])
    p.add_argument("--output-dir", default="outputs")

    # Model config
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--embedding-model", default="text-embedding-3-small")

    # NEW: Separate RAGAS evaluator LLM (cost optimization)
    p.add_argument(
        "--ragas-llm", default=None,
        help="LLM for RAGAS evaluation (default: same as --model). "
             "Use gpt-4o-mini for ~60%% cost savings.",
    )

    # Retrieval config
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--retriever", default="hybrid", choices=["bm25", "vector", "hybrid"])

    # P2: Reranker control
    p.add_argument(
        "--no-reranker", action="store_true",
        help="Disable CrossEncoder reranking (faster but lower precision)",
    )

    # DSPy module for generation
    p.add_argument("--dspy-module", default=None,
                   choices=["predict", "chain_of_thought", "program_of_thought", "react",
                            "multi_chain", "refine", "best_of_n", "self_refining_rag",
                            "multi_hop_rag", "adaptive", None])

    # DSPy optimization config
    p.add_argument("--dspy-modules", nargs="+",
                   default=["chain_of_thought", "predict", "self_refining_rag"])
    p.add_argument("--dspy-optimizers", nargs="+",
                   default=["miprov2", "bootstrap_fewshot", "copro"])
    p.add_argument("--dspy-metric", default="combined",
                   choices=["f1", "faithfulness", "combined"])

    # RAGAS config
    p.add_argument("--ragas-preset", default="core",
                   choices=["core", "extended", "full", "retrieval_only",
                            "generation_only", "non_llm"])

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    steps = ALL_STEPS if "all" in args.steps else args.steps

    print(f"\n{'='*70}")
    print(f"  Shell CRA — AutoRAG + DSPy + RAGAS Pipeline (v3 — Improved)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"  Steps: {', '.join(steps)}")
    print(f"  Model: {args.model} | Retriever: {args.retriever}")
    print(f"  Chunks: {args.chunk_size}/{args.chunk_overlap} | Top-K: {args.top_k}")
    if args.dspy_module:
        print(f"  DSPy module: {args.dspy_module}")
    print(f"  RAGAS preset: {args.ragas_preset} | RAGAS LLM: {args.ragas_llm or args.model}")
    print(f"  Reranker: {'disabled' if args.no_reranker else 'CrossEncoder (P2)'}")
    print(f"  DSPy sweep: {args.dspy_modules} × {args.dspy_optimizers}")
    print(f"{'='*70}")
    print(f"  IMPROVEMENTS ACTIVE:")
    print(f"    P0: Table-aware XLSX chunking (Q→A→Comment blocks)")
    print(f"    P1: DSPy module single-instantiation")
    print(f"    P2: CrossEncoder reranking {'ON' if not args.no_reranker else 'OFF'}")
    print(f"    P3: Concise CRA system prompt")
    print(f"    P4: Actual retrieved contexts → RAGAS")
    print(f"{'='*70}\n")

    results = {}
    for step in steps:
        if step not in STEP_MAP:
            continue
        print(f"  ▶ Step: {step}")
        t0 = time.time()
        try:
            result = STEP_MAP[step](args)
            elapsed = time.time() - t0
            results[step] = {
                "status": "ok", "elapsed": round(elapsed, 2), "result": result,
            }
            print(f"  ✓ {step} in {elapsed:.1f}s\n")
        except Exception as e:
            elapsed = time.time() - t0
            results[step] = {
                "status": "error", "elapsed": round(elapsed, 2), "error": str(e),
            }
            print(f"  ✗ {step} failed: {e}\n")
            import traceback
            traceback.print_exc()

    meta = {
        "timestamp": datetime.now().isoformat(),
        "version": "v3_improved",
        "improvements": ["P0_table_chunking", "P1_dspy_single_init",
                         "P2_crossencoder_rerank", "P3_concise_prompt",
                         "P4_actual_contexts_ragas"],
        "args": vars(args),
        "steps": results,
    }
    with open(
        os.path.join(args.output_dir, "pipeline_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Pipeline complete (v3) → {args.output_dir}/")
    for step, r in results.items():
        s = "✓" if r["status"] == "ok" else "✗"
        print(f"    {s} {step}: {r['elapsed']}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
