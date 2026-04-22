"""
evaluation/report.py — Final Report Generator
==============================================
"""
import os, json
from datetime import datetime
from typing import Dict

def generate_report(output_dir: str, report_path: str) -> Dict:
    """Generate consolidated final report."""
    report = {"generated_at": datetime.now().isoformat(), "sections": {}}

    for fname in ["pipeline_meta.json","ground_truth.json","evaluation.json","ragas_evaluation.json","diagnosis.json","rag_answers.json"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, encoding="utf-8") as f:
                report["sections"][fname.replace(".json","")] = json.load(f)

    # Add optimization results if available
    for subdir in ["optimization", "dspy_optimization"]:
        opt_dir = os.path.join(output_dir, subdir)
        if os.path.exists(opt_dir):
            for fname in os.listdir(opt_dir):
                if fname.endswith(".json"):
                    with open(os.path.join(opt_dir, fname), encoding="utf-8") as f:
                        report["sections"][f"{subdir}_{fname}"] = json.load(f)

    out_dir = os.path.dirname(report_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"    Report generated → {report_path}")
    return {"path": report_path, "sections": list(report["sections"].keys())}
