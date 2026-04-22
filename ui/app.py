"""
ui/app.py — Streamlit Dashboard for Shell CRA AutoRAG + DSPy Pipeline
=====================================================================
Launch: streamlit run ui/app.py
"""

import os, sys, json
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

st.set_page_config(page_title="Shell CRA — AutoRAG + DSPy Lab", layout="wide", page_icon="⚗️")

def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

gt_data = load_json(os.path.join(OUTPUT_DIR, "ground_truth.json"))
answers_data = load_json(os.path.join(OUTPUT_DIR, "rag_answers.json"))
ragas_data = load_json(os.path.join(OUTPUT_DIR, "ragas_evaluation.json"))
eval_data = load_json(os.path.join(OUTPUT_DIR, "evaluation.json"))
diag_data = load_json(os.path.join(OUTPUT_DIR, "diagnosis.json"))
meta_data = load_json(os.path.join(OUTPUT_DIR, "pipeline_meta.json"))
report_data = load_json(os.path.join(OUTPUT_DIR, "final_report.json"))

# Use RAGAS data if available, else fallback evaluation
active_eval = ragas_data if ragas_data and "avg_scores" in ragas_data else eval_data

METRICS = ["faithfulness","answer_relevancy","context_precision","context_recall","answer_correctness","answer_similarity"]

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚗️ Shell CRA Lab")
    st.caption("AutoRAG + DSPy + RAGAS Pipeline")
    st.divider()

    if meta_data:
        st.caption(f"Last run: {meta_data.get('timestamp','N/A')[:19]}")
        args = meta_data.get("args", {})
        st.caption(f"Model: {args.get('model', 'N/A')}")
        st.caption(f"Retriever: {args.get('retriever', 'N/A')}")
        st.caption(f"Chunks: {args.get('chunk_size', '?')}/{args.get('chunk_overlap', '?')}")
        st.caption(f"Top-K: {args.get('top_k', 'N/A')}")

    st.divider()
    st.subheader("Run Pipeline")
    st.code("python pipeline.py --steps all --model gpt-4o", language="powershell")
    st.caption(f"Output: {OUTPUT_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📋 Ground Truth", "✅ RAGAS Evaluation",
    "🔍 Answers", "⚠️ Diagnosis", "🧠 DSPy Optimization"
])

# ─── TAB 1: OVERVIEW ──────────────────────────────────────────

with tab1:
    st.header("Pipeline Overview")

    if meta_data:
        steps = meta_data.get("steps", {})
        col1, col2, col3 = st.columns(3)
        ok_count = sum(1 for s in steps.values() if s["status"] == "ok")
        fail_count = sum(1 for s in steps.values() if s["status"] != "ok")
        total_time = sum(s.get("elapsed", 0) for s in steps.values())
        col1.metric("Steps Passed", f"{ok_count}/{len(steps)}")
        col2.metric("Total Time", f"{total_time:.0f}s")
        col3.metric("Model", meta_data.get("args", {}).get("model", "N/A"))

        st.subheader("Step Results")
        for step, result in steps.items():
            status = "✅" if result["status"] == "ok" else "❌"
            st.write(f"{status} **{step}** — {result['elapsed']}s")
    else:
        st.info("No pipeline results yet. Run the pipeline first.")

    if active_eval and "avg_scores" in active_eval:
        st.subheader("RAGAS Scores")
        cols = st.columns(len(active_eval["avg_scores"]))
        for i, (metric, score) in enumerate(active_eval["avg_scores"].items()):
            cols[i % len(cols)].metric(
                metric.replace("_", " ").title(),
                f"{score*100:.1f}%"
            )

    if gt_data:
        st.subheader("Knowledge Base")
        col1, col2, col3 = st.columns(3)
        col1.metric("Ground Truth Q&A", len(gt_data))
        if answers_data:
            col2.metric("Generated Answers", len(answers_data))
            tokens = sum(a.get("generation", {}).get("tokens", {}).get("total", 0) for a in answers_data)
            col3.metric("Total Tokens", f"{tokens:,}")

# ─── TAB 2: GROUND TRUTH ─────────────────────────────────────

with tab2:
    st.header("Ground Truth Q&A Pairs")

    if gt_data:
        sections = sorted(set(q.get("section", "unknown") for q in gt_data))
        sel_section = st.selectbox("Filter by Section", ["All"] + sections)
        filtered = gt_data if sel_section == "All" else [q for q in gt_data if q.get("section") == sel_section]
        st.write(f"Showing {len(filtered)} questions")

        for q in filtered:
            diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(q.get("difficulty", ""), "⚪")
            with st.expander(f"{diff_color} **{q.get('question_id', '')}** — {q['question'][:80]}"):
                st.write(f"**Section:** {q.get('section', '')}")
                st.write(f"**Difficulty:** {q.get('difficulty', '')}")
                st.success(f"**Answer:** {q['answer']}")
                if q.get("comments"):
                    st.info(f"**Comments:** {q['comments'][:500]}")
    else:
        st.info("Run 'groundtruth' step first.")

# ─── TAB 3: RAGAS EVALUATION ─────────────────────────────────

with tab3:
    st.header("RAGAS Evaluation Results")

    if active_eval and "avg_scores" in active_eval:
        avg = active_eval["avg_scores"]

        # Summary bar
        cols = st.columns(min(6, len(avg)))
        for i, (m, v) in enumerate(avg.items()):
            color = "normal" if v >= 0.6 else "off" if v >= 0.4 else "inverse"
            cols[i % len(cols)].metric(m.replace("_", " ").title(), f"{v*100:.1f}%")

        st.divider()

        # Per-sample breakdown
        samples = active_eval.get("per_sample_scores", [])
        if samples:
            st.subheader(f"Per-Question Scores ({len(samples)} questions)")
            for i, s in enumerate(samples):
                question = s.get("user_input", f"Question {i+1}")
                scores = {k: v for k, v in s.items() if k not in ["user_input", "response", "retrieved_contexts", "reference"] and v is not None}
                if scores:
                    avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / max(len(scores), 1)
                    status = "✅" if avg_score >= 0.5 else "⚠️" if avg_score >= 0.3 else "❌"
                    with st.expander(f"{status} Q{i+1}: {question[:70]}... — avg {avg_score*100:.0f}%"):
                        mcols = st.columns(min(6, len(scores)))
                        for j, (mk, mv) in enumerate(scores.items()):
                            if isinstance(mv, (int, float)):
                                mcols[j % len(mcols)].metric(mk.split("_")[0], f"{mv*100:.0f}%")
                        st.write(f"**Response:** {s.get('response', '')[:300]}")
                        st.write(f"**Reference:** {s.get('reference', '')[:300]}")
    else:
        st.info("Run 'ragas_eval' step first.")

# ─── TAB 4: ANSWERS ──────────────────────────────────────────

with tab4:
    st.header("Generated Answers")

    if answers_data:
        st.write(f"Total: {len(answers_data)} answers")
        for i, ans in enumerate(answers_data):
            gen = ans.get("generation", {})
            latency = gen.get("latency_ms", 0)
            method = gen.get("method", "unknown")
            with st.expander(f"**{ans.get('question_id', f'Q{i+1}')}** [{ans.get('section', '')}] — {latency:.0f}ms"):
                st.write(f"**Question:** {ans['question']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Ground Truth:**\n{ans.get('ground_truth', 'N/A')}")
                with col2:
                    st.info(f"**Generated ({method}):**\n{ans.get('generated_answer', 'N/A')[:500]}")

                ret = ans.get("retrieval", {})
                st.caption(f"Sources: {', '.join(ret.get('sources', []))} | Chunks: {ret.get('chunks', 0)} | Context: {ret.get('context_chars', 0):,} chars")
    else:
        st.info("Run 'generate' step first.")

# ─── TAB 5: DIAGNOSIS ────────────────────────────────────────

with tab5:
    st.header("Root Cause Diagnosis")

    if diag_data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", diag_data.get("total_questions", 0))
        col2.metric("Passed", diag_data.get("passed", 0))
        col3.metric("Failed", diag_data.get("failed", 0))
        col4.metric("Borderline", diag_data.get("borderline", 0))

        causes = diag_data.get("root_causes", {})
        if causes:
            st.subheader("Root Causes")
            for cause, count in causes.items():
                st.write(f"**{cause}**: {count} questions")

        fixes = diag_data.get("priority_fixes", [])
        if fixes:
            st.subheader("Priority Fixes")
            for fix in fixes:
                pri = {"P0": "🔴", "P1": "🟡", "P2": "🟢"}.get(fix.get("priority", ""), "⚪")
                st.write(f"{pri} **[{fix['priority']}] {fix['cause']}** — {fix['affected_questions']} questions")
                st.caption(fix.get("description", ""))

        sections = diag_data.get("section_analysis", {})
        if sections:
            st.subheader("Section Analysis")
            for sec, stats in sections.items():
                pct = stats.get("passed", 0) / max(stats.get("total", 1), 1) * 100
                st.write(f"**{sec}**: {stats.get('passed', 0)}/{stats.get('total', 0)} ({pct:.0f}%) — avg {stats.get('avg_composite', 0)*100:.1f}%")
    else:
        st.info("Run 'diagnose' step first.")

# ─── TAB 6: DSPy OPTIMIZATION ────────────────────────────────

with tab6:
    st.header("DSPy Optimization Results")

    opt_dir = os.path.join(OUTPUT_DIR, "dspy_optimization")
    if os.path.exists(opt_dir):
        # Load sweep summary
        sweep = load_json(os.path.join(opt_dir, "sweep_summary.json"))
        if sweep:
            col1, col2 = st.columns(2)
            col1.metric("Total Experiments", sweep.get("total_experiments", 0))
            col2.metric("Successful", sweep.get("successful", 0))

            best = sweep.get("best", {})
            if best:
                st.success(f"🏆 **Best: {best.get('module', '?')} × {best.get('optimizer', '?')} = {best.get('score', 0)*100:.1f}%**")

            results = sweep.get("results", [])
            if results:
                st.subheader("Experiment Results (sorted by score)")
                for r in results:
                    with st.expander(f"**{r.get('module', '?')} × {r.get('optimizer', '?')}** — {r.get('avg_val_score', 0)*100:.1f}%"):
                        st.write(f"Train: {r.get('train_size', '?')} | Val: {r.get('val_size', '?')} | Time: {r.get('elapsed_s', 0):.1f}s")
                        if r.get("val_scores"):
                            st.write(f"Val scores: {[f'{s*100:.1f}%' for s in r['val_scores']]}")

        # Also show individual result files
        for fname in sorted(os.listdir(opt_dir)):
            if fname.startswith("result_") and fname.endswith(".json"):
                data = load_json(os.path.join(opt_dir, fname))
                if data and data.get("status") == "ok":
                    with st.expander(f"📄 {fname}"):
                        st.json(data)
    else:
        st.info("Run 'dspy_optimize' step first.")

    st.divider()
    st.subheader("Available DSPy Modules")
    modules = {
        "predict": "dspy.Predict — Direct single-pass",
        "chain_of_thought": "dspy.ChainOfThought — Step-by-step reasoning",
        "program_of_thought": "dspy.ProgramOfThought — Code-based computation",
        "react": "dspy.ReAct — Reason + Act agent",
        "multi_chain": "dspy.MultiChainComparison — N candidates, pick best",
        "refine": "dspy.Refine — Iterative self-refinement",
        "best_of_n": "dspy.BestOfN — Sample N, return highest",
        "self_refining_rag": "Custom: Generate → Judge → Refine",
        "multi_hop_rag": "Custom: Decompose → Sub-answer → Synthesize",
        "adaptive": "Custom: Routes by question category",
    }
    for name, desc in modules.items():
        st.write(f"`{name}` — {desc}")
