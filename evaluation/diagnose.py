"""
evaluation/diagnose.py — Root Cause Diagnosis Engine
====================================================
Analyzes evaluation results to identify systematic failures and
recommend pipeline improvements (inspired by RAGdx).

evaluation/report.py — Final Report Generator
"""

import os, json
from collections import Counter
from typing import Dict, List


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════════

CAUSE_DESCRIPTIONS = {
    "CHUNKING": "Document chunking splits key information across boundaries. Fix: Use table-aware (XLSX) and semantic (PDF) chunking.",
    "RETRIEVAL": "Top-K retrieved chunks miss relevant content. Fix: Hybrid retrieval (BM25+Vector) with CrossEncoder reranking.",
    "GROUNDING": "Generated answer includes claims not in context. Fix: Constrained decoding or SelfRefiningRAG module.",
    "GENERATION": "LLM produces incorrect answer despite good context. Fix: DSPy optimization (MIPROv2) with few-shot examples.",
    "EMBEDDING": "Semantic similarity fails on domain-specific terms. Fix: Domain-adapted embeddings or HyDE query expansion.",
    "BORDERLINE": "Marginal performance. Fix: Reranking + prompt engineering.",
}


def diagnose_failures(eval_path: str, output_path: str) -> Dict:
    """Analyze evaluation results for root causes."""
    with open(eval_path, encoding="utf-8") as f:
        evaluation = json.load(f)

    results = evaluation.get("results", [])
    failed = [r for r in results if not r["passed"]]
    borderline = [r for r in results if r["passed"] and r["composite"] < 0.6]

    # Count root causes
    cause_counts = Counter()
    for r in failed + borderline:
        for issue in r.get("issues", []):
            cause = issue.split(":")[0].strip()
            if cause in CAUSE_DESCRIPTIONS:
                cause_counts[cause] += 1

    # Section analysis
    section_stats = {}
    for r in results:
        sec = r.get("section", "unknown")
        if sec not in section_stats:
            section_stats[sec] = {"total": 0, "passed": 0, "composites": []}
        section_stats[sec]["total"] += 1
        if r["passed"]:
            section_stats[sec]["passed"] += 1
        section_stats[sec]["composites"].append(r["composite"])

    for sec in section_stats:
        comps = section_stats[sec]["composites"]
        section_stats[sec]["avg_composite"] = round(sum(comps)/len(comps), 4)
        del section_stats[sec]["composites"]

    # Priority fixes
    fixes = []
    for cause, count in cause_counts.most_common():
        fixes.append({
            "cause": cause,
            "affected_questions": count,
            "description": CAUSE_DESCRIPTIONS.get(cause, "Unknown"),
            "priority": "P0" if count >= 3 else "P1" if count >= 2 else "P2",
        })

    diagnosis = {
        "total_questions": len(results),
        "passed": len([r for r in results if r["passed"]]),
        "failed": len(failed),
        "borderline": len(borderline),
        "root_causes": dict(cause_counts),
        "section_analysis": section_stats,
        "priority_fixes": fixes,
        "failed_questions": [
            {"id": r["question_id"], "section": r["section"], "question": r["question"][:80],
             "composite": r["composite"], "issues": r["issues"]}
            for r in failed
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2)

    print(f"    Diagnosis: {len(failed)} failures, {len(borderline)} borderline")
    for fix in fixes:
        print(f"      [{fix['priority']}] {fix['cause']}: {fix['affected_questions']} questions — {fix['description'][:60]}")

    return diagnosis
