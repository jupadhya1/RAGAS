"""
ui/app.py — Shell CRA Observability Dashboard (v3 — Deep Reasoning Traces)
===========================================================================
Comprehensive pipeline observability with 10 tabs:
  1. Overview: Pipeline health, timing, improvements active
  2. Ground Truth vs Generated: Side-by-side comparison with diff highlights
  3. Retrieval Traces: Full chunk-level trace per question
  4. RAGAS Deep Dive: Per-metric, per-question drill-down
  5. Reasoning Traces: Full generation chain (prompt → context → answer)
  6. DSPy Parameters: Signatures, modules, optimizer configs, proposed instructions
  7. AutoRAG Config: Full hyperparameter search space visualization
  8. Corpus Explorer: Browse chunks by source/section
  9. Diagnosis & Root Cause: Failure analysis with fix recommendations
  10. Pipeline Config: Full args, versions, improvements, reproducibility
"""

import os, sys, json, re
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

st.set_page_config(
    page_title="Shell CRA — Observability Lab", layout="wide", page_icon="🔬"
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

def load_yaml(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None

gt_data = load_json(os.path.join(OUTPUT_DIR, "ground_truth.json"))
answers_data = load_json(os.path.join(OUTPUT_DIR, "rag_answers.json"))
ragas_data = load_json(os.path.join(OUTPUT_DIR, "ragas_evaluation.json"))
eval_data = load_json(os.path.join(OUTPUT_DIR, "evaluation.json"))
diag_data = load_json(os.path.join(OUTPUT_DIR, "diagnosis.json"))
meta_data = load_json(os.path.join(OUTPUT_DIR, "pipeline_meta.json"))
report_data = load_json(os.path.join(OUTPUT_DIR, "final_report.json"))
corpus_data = load_json(os.path.join(OUTPUT_DIR, "corpus.json"))
index_meta = load_json(os.path.join(OUTPUT_DIR, "index", "index_meta.json"))
autorag_yaml = load_yaml(os.path.join(OUTPUT_DIR, "autorag_data", "pipeline.yaml"))

active_eval = ragas_data if ragas_data and "avg_scores" in ragas_data else eval_data


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔬 Shell CRA")
    st.caption("Observability Lab v3")
    st.divider()

    if meta_data:
        args = meta_data.get("args", {})
        st.caption(f"🕐 {meta_data.get('timestamp', 'N/A')[:19]}")
        st.caption(f"🤖 Model: **{args.get('model', 'N/A')}**")
        st.caption(f"🔍 Retriever: {args.get('retriever', 'N/A')}")
        st.caption(f"📦 Chunks: {args.get('chunk_size', '?')}/{args.get('chunk_overlap', '?')}")
        st.caption(f"🎯 Top-K: {args.get('top_k', 'N/A')}")
        st.caption(f"📊 RAGAS: {args.get('ragas_preset', 'core')}")

        version = meta_data.get("version", "v2")
        improvements = meta_data.get("improvements", [])
        if improvements:
            st.divider()
            st.caption("**Improvements Active:**")
            for imp in improvements:
                st.caption(f"  ✅ {imp}")

    st.divider()
    st.caption("**Quick Run:**")
    st.code("python pipeline.py --steps all --model gpt-4o", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "📊 Overview",
    "🔀 GT vs Generated",
    "🔍 Retrieval Traces",
    "✅ RAGAS Deep Dive",
    "🧠 Reasoning Traces",
    "⚙️ DSPy Parameters",
    "🔧 AutoRAG Config",
    "📚 Corpus Explorer",
    "⚠️ Diagnosis",
    "🛠️ Pipeline Config",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("Pipeline Overview")

    if meta_data:
        steps = meta_data.get("steps", {})
        ok_count = sum(1 for s in steps.values() if s.get("status") == "ok")
        total_time = sum(s.get("elapsed", 0) for s in steps.values())
        args = meta_data.get("args", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps Passed", f"{ok_count}/{len(steps)}")
        c2.metric("Total Time", f"{total_time:.0f}s")
        c3.metric("Model", args.get("model", "N/A"))
        c4.metric("Corpus Chunks", f"{len(corpus_data):,}" if corpus_data else "N/A")

        # RAGAS summary
        if active_eval and "avg_scores" in active_eval:
            st.subheader("RAGAS Scores")
            avg = active_eval["avg_scores"]
            cols = st.columns(min(6, len(avg)))
            for i, (m, v) in enumerate(avg.items()):
                delta_color = "normal" if v >= 0.7 else "off" if v >= 0.5 else "inverse"
                cols[i % len(cols)].metric(
                    m.replace("_", " ").title(), f"{v*100:.1f}%",
                    delta_color=delta_color,
                )

        # Step timeline
        st.subheader("Step Timeline")
        for step, result in steps.items():
            status = "✅" if result.get("status") == "ok" else "❌"
            elapsed = result.get("elapsed", 0)
            bar_len = min(int(elapsed / 2), 50)
            bar = "█" * bar_len
            st.text(f"  {status} {step:20s} {elapsed:7.1f}s  {bar}")
    else:
        st.info("No pipeline results yet. Run the pipeline first.")

    if gt_data:
        st.subheader("Knowledge Base Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ground Truth Q&A", len(gt_data))
        if answers_data:
            c2.metric("Generated Answers", len(answers_data))
            tokens = sum(
                a.get("generation", {}).get("tokens", {}).get("total", 0)
                for a in answers_data
            )
            c3.metric("Total Tokens", f"{tokens:,}")

        # Section distribution
        sections = {}
        for q in gt_data:
            s = q.get("section", "unknown")
            sections[s] = sections.get(s, 0) + 1
        st.caption("**Questions per section:** " + " | ".join(f"{s}: {c}" for s, c in sections.items()))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: GROUND TRUTH vs GENERATED — Side by Side
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Ground Truth vs Generated Answers")

    if answers_data:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        sections = sorted(set(a.get("section", "?") for a in answers_data))
        sel_section = fc1.selectbox("Section", ["All"] + sections, key="gt_sec")
        difficulties = sorted(set(a.get("difficulty", "?") for a in answers_data))
        sel_diff = fc2.selectbox("Difficulty", ["All"] + difficulties, key="gt_diff")

        # Get RAGAS per-sample if available
        ragas_per_sample = {}
        if ragas_data and "per_sample_scores" in ragas_data:
            for i, s in enumerate(ragas_data["per_sample_scores"]):
                ragas_per_sample[i] = s

        filtered = answers_data
        if sel_section != "All":
            filtered = [a for a in filtered if a.get("section") == sel_section]
        if sel_diff != "All":
            filtered = [a for a in filtered if a.get("difficulty") == sel_diff]

        st.caption(f"Showing {len(filtered)} of {len(answers_data)} answers")

        for ans in filtered:
            idx = answers_data.index(ans)
            q_id = ans.get("question_id", f"Q{idx+1}")
            section = ans.get("section", "")
            difficulty = ans.get("difficulty", "medium")
            diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")
            gen = ans.get("generation", {})
            latency = gen.get("latency_ms", 0)

            # Get RAGAS score for this question
            sample_scores = ragas_per_sample.get(idx, {})
            correctness = sample_scores.get("answer_correctness", None)
            faithfulness = sample_scores.get("faithfulness", None)
            score_str = ""
            if correctness is not None:
                score_str += f" | Correct: {correctness*100:.0f}%"
            if faithfulness is not None:
                score_str += f" | Faith: {faithfulness*100:.0f}%"

            with st.expander(
                f"{diff_icon} **{q_id}** [{section}] — {latency:.0f}ms{score_str}"
            ):
                st.write(f"**Question:** {ans['question']}")
                st.divider()

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 🎯 Ground Truth")
                    st.success(ans.get("ground_truth", "N/A"))
                with col2:
                    st.markdown("##### 🤖 Generated Answer")
                    st.info(ans.get("generated_answer", "N/A")[:800])

                # Token overlap analysis
                gt_text = (ans.get("ground_truth", "") or "").lower()
                gen_text = (ans.get("generated_answer", "") or "").lower()
                gt_tokens = set(re.findall(r'\b[a-z0-9]+(?:\.[0-9]+)?\b', gt_text))
                gen_tokens = set(re.findall(r'\b[a-z0-9]+(?:\.[0-9]+)?\b', gen_text))
                stops = {'the','and','was','for','are','with','from','that','this','which','has','have','been'}
                gt_tokens -= stops
                gen_tokens -= stops
                overlap = gt_tokens & gen_tokens
                missing = gt_tokens - gen_tokens

                st.divider()
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("GT Tokens", len(gt_tokens))
                mc2.metric("Gen Tokens", len(gen_tokens))
                mc3.metric("Overlap", len(overlap))
                mc4.metric("F1", f"{2*len(overlap)/(len(gt_tokens)+len(gen_tokens))*100:.0f}%" if gt_tokens or gen_tokens else "N/A")

                if missing:
                    st.caption(f"**Missing from generated:** {', '.join(sorted(list(missing)[:20]))}")
                if overlap:
                    st.caption(f"**Matched tokens:** {', '.join(sorted(list(overlap)[:20]))}")

                # RAGAS breakdown for this question
                if sample_scores:
                    st.divider()
                    st.markdown("##### RAGAS Scores")
                    score_cols = st.columns(6)
                    for j, (mk, mv) in enumerate(sample_scores.items()):
                        if mk not in ["user_input", "response", "retrieved_contexts", "reference",
                                       "question_id", "section", "difficulty"] and isinstance(mv, (int, float)):
                            score_cols[j % 6].metric(mk.split("_")[0].title(), f"{mv*100:.0f}%")
    else:
        st.info("Run 'generate' step first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: RETRIEVAL TRACES
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Retrieval Traces — Per-Question Chunk Analysis")

    if answers_data:
        q_options = [
            f"{a.get('question_id', f'Q{i+1}')} — {a['question'][:60]}"
            for i, a in enumerate(answers_data)
        ]
        sel_q = st.selectbox("Select Question", q_options, key="ret_q")
        sel_idx = q_options.index(sel_q)
        ans = answers_data[sel_idx]
        ret = ans.get("retrieval", {})

        st.subheader(f"Question: {ans['question']}")

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Retriever", ret.get("retriever", "hybrid"))
        rc2.metric("Top-K", ret.get("top_k", 8))
        rc3.metric("Chunks Retrieved", ret.get("chunks", 0))
        rc4.metric("Context Chars", f"{ret.get('context_chars', 0):,}")

        sources = ret.get("sources", [])
        st.caption(f"**Sources hit:** {', '.join(sources)}")

        # Show each retrieved chunk
        contexts = ret.get("retrieved_contexts", [])
        if contexts:
            st.subheader(f"Retrieved Chunks ({len(contexts)})")
            for i, chunk in enumerate(contexts):
                # Try to identify source from chunk content
                source_hint = "CRA" if "[CRA" in chunk else "PDF"
                relevance_bar = "🟩" * max(1, 5 - i)  # decreasing relevance
                with st.expander(f"Chunk {i+1} {relevance_bar} ({len(chunk)} chars) — {source_hint}"):
                    st.code(chunk[:1500], language="text")
                    # Highlight GT terms in chunk
                    gt_text = (ans.get("ground_truth", "") or "").lower()
                    gt_key_terms = set(re.findall(r'\b[a-z0-9]+(?:\.[0-9]+)?\b', gt_text))
                    gt_key_terms -= {'the','and','was','for','are','with','from','that','this','which'}
                    chunk_lower = chunk.lower()
                    found = [t for t in gt_key_terms if t in chunk_lower]
                    if found:
                        st.caption(f"✅ **GT terms found in chunk:** {', '.join(sorted(found)[:15])}")
                    else:
                        st.caption("⚠️ No ground truth key terms found in this chunk")
        else:
            st.warning("No retrieved_contexts stored. Run with v3 generate to capture chunks.")

        # Show generation trace
        st.divider()
        st.subheader("Generation Trace")
        gen = ans.get("generation", {})
        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Method", gen.get("method", "?"))
        gc2.metric("Latency", f"{gen.get('latency_ms', 0):.0f}ms")
        gc3.metric("Prompt Tokens", gen.get("tokens", {}).get("prompt", 0))
        gc4.metric("Completion Tokens", gen.get("tokens", {}).get("completion", 0))
    else:
        st.info("Run 'generate' step first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: RAGAS DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("RAGAS Deep Dive — Per-Metric Analysis")

    if active_eval and "avg_scores" in active_eval:
        avg = active_eval["avg_scores"]
        samples = active_eval.get("per_sample_scores", [])

        # Metric descriptions
        metric_desc = {
            "faithfulness": "Are generated claims grounded in the retrieved context? (Higher = fewer hallucinations)",
            "answer_relevancy": "Does the answer actually address the question asked? (Computed via synthetic question generation)",
            "context_precision": "Are the most relevant chunks ranked at the top? (Position-weighted relevance)",
            "context_recall": "What fraction of needed information was actually retrieved? (Coverage of reference answer)",
            "answer_correctness": "Token overlap + semantic similarity vs ground truth (75% F1 + 25% semantic)",
            "answer_similarity": "Embedding cosine similarity between generated and ground truth answers",
        }

        # Summary
        cols = st.columns(min(6, len(avg)))
        for i, (m, v) in enumerate(avg.items()):
            cols[i % len(cols)].metric(m.replace("_", " ").title(), f"{v*100:.1f}%")

        st.divider()

        # Per-metric deep dive
        sel_metric = st.selectbox(
            "Select metric for deep dive",
            list(avg.keys()),
            key="ragas_metric"
        )

        st.info(metric_desc.get(sel_metric, ""))
        st.metric(f"Average {sel_metric}", f"{avg[sel_metric]*100:.1f}%")

        if samples:
            # Sort by this metric
            scored = []
            for i, s in enumerate(samples):
                val = s.get(sel_metric)
                if isinstance(val, (int, float)):
                    q = s.get("user_input", f"Q{i+1}")
                    q_id = s.get("question_id", f"Q{i+1}")
                    scored.append((q_id, q, val, i))

            scored.sort(key=lambda x: x[2])

            # Worst and best
            st.subheader(f"Worst 5 by {sel_metric}")
            for q_id, q, val, idx in scored[:5]:
                color = "🔴" if val < 0.3 else "🟡" if val < 0.6 else "🟢"
                with st.expander(f"{color} {q_id}: {val*100:.0f}% — {q[:70]}"):
                    s = samples[idx]
                    st.write(f"**Response:** {s.get('response', '')[:400]}")
                    st.write(f"**Reference:** {s.get('reference', '')[:400]}")
                    # Show all metrics for this question
                    mcols = st.columns(6)
                    j = 0
                    for mk, mv in s.items():
                        if mk not in ["user_input", "response", "retrieved_contexts", "reference",
                                       "question_id", "section", "difficulty"] and isinstance(mv, (int, float)):
                            mcols[j % 6].metric(mk.replace("_", " ").title()[:15], f"{mv*100:.0f}%")
                            j += 1

            st.subheader(f"Best 5 by {sel_metric}")
            for q_id, q, val, idx in scored[-5:][::-1]:
                color = "🟢" if val >= 0.7 else "🟡"
                with st.expander(f"{color} {q_id}: {val*100:.0f}% — {q[:70]}"):
                    s = samples[idx]
                    st.write(f"**Response:** {s.get('response', '')[:400]}")
                    st.write(f"**Reference:** {s.get('reference', '')[:400]}")

        # Evaluator info
        st.divider()
        st.caption(f"Evaluator LLM: {active_eval.get('evaluator_llm', 'default')}")
        st.caption(f"Contexts source: {active_eval.get('contexts_source', 'unknown')}")
        st.caption(f"Metric preset: {active_eval.get('metric_preset', 'core')}")
    else:
        st.info("Run 'ragas_eval' step first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: REASONING TRACES — Full Generation Chain
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("Reasoning Traces — Full Generation Chain")
    st.caption("Trace the complete path: Question → Retrieval → Context Assembly → Prompt → Generation → Answer")

    if answers_data:
        q_options = [
            f"{a.get('question_id', f'Q{i+1}')} [{a.get('difficulty','')}] — {a['question'][:55]}"
            for i, a in enumerate(answers_data)
        ]
        sel_q = st.selectbox("Select Question", q_options, key="reason_q")
        sel_idx = q_options.index(sel_q)
        ans = answers_data[sel_idx]
        gen = ans.get("generation", {})
        ret = ans.get("retrieval", {})

        # ── Step 1: Input Question ────────────────────────────
        st.subheader("Step 1 → Input Question")
        st.markdown(f"""
| Field | Value |
|-------|-------|
| **Question ID** | `{ans.get('question_id', '?')}` |
| **Section** | {ans.get('section', '?')} |
| **Difficulty** | {ans.get('difficulty', '?')} |
| **Question** | {ans['question']} |
""")

        # ── Step 2: Retrieval ─────────────────────────────────
        st.subheader("Step 2 → Hybrid Retrieval (BM25 + Embedding + RRF + Rerank)")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| **Retriever** | {ret.get('retriever', 'hybrid')} |
| **Top-K** | {ret.get('top_k', 8)} |
| **Chunks returned** | {ret.get('chunks', 0)} |
| **Sources hit** | {', '.join(ret.get('sources', []))} |
| **Total context chars** | {ret.get('context_chars', 0):,} |
""")

        contexts = ret.get("retrieved_contexts", [])
        if contexts:
            for i, chunk in enumerate(contexts):
                st.text(f"  Chunk {i+1} ({len(chunk)} chars):")
                st.code(chunk[:500] + ("..." if len(chunk) > 500 else ""), language="text")

        # ── Step 3: Prompt Assembly ───────────────────────────
        st.subheader("Step 3 → Prompt Assembly")
        # Reconstruct the prompt
        from generation.generate import SYSTEM_PROMPT
        context_text = "\n\n---\n\n".join(contexts[:5]) if contexts else "[no context]"
        full_prompt = f"""[SYSTEM]
{SYSTEM_PROMPT}

[USER]
Question: {ans['question']}

Context:
{context_text[:2000]}{'...[truncated]' if len(context_text) > 2000 else ''}

Answer:"""
        st.code(full_prompt, language="text")
        st.caption(f"Estimated prompt tokens: ~{len(full_prompt.split()) * 1.3:.0f}")

        # ── Step 4: Generation ────────────────────────────────
        st.subheader("Step 4 → LLM Generation")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| **Method** | `{gen.get('method', '?')}` |
| **Model** | `{gen.get('model', '?')}` |
| **Temperature** | 0.0 |
| **Max tokens** | 300 |
| **Latency** | {gen.get('latency_ms', 0):.0f}ms |
| **Prompt tokens** | {gen.get('tokens', {}).get('prompt', 0)} |
| **Completion tokens** | {gen.get('tokens', {}).get('completion', 0)} |
| **Total tokens** | {gen.get('tokens', {}).get('total', 0)} |
| **Status** | {gen.get('status', '?')} |
""")

        # ── Step 5: Output ────────────────────────────────────
        st.subheader("Step 5 → Generated Answer")
        st.info(ans.get("generated_answer", "N/A"))

        if gen.get("explanation"):
            st.subheader("DSPy Explanation (if CoT)")
            st.write(gen["explanation"])

        # ── Step 6: Evaluation ────────────────────────────────
        st.subheader("Step 6 → RAGAS Evaluation")
        if ragas_data and "per_sample_scores" in ragas_data:
            scores = ragas_data["per_sample_scores"]
            if sel_idx < len(scores):
                s = scores[sel_idx]
                mcols = st.columns(6)
                j = 0
                for mk, mv in s.items():
                    if mk not in ["user_input", "response", "retrieved_contexts", "reference",
                                   "question_id", "section", "difficulty"] and isinstance(mv, (int, float)):
                        mcols[j % 6].metric(mk.replace("_", " ").title()[:15], f"{mv*100:.0f}%")
                        j += 1

        # Ground truth comparison
        st.subheader("Step 7 → Ground Truth Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🎯 Expected")
            st.success(ans.get("ground_truth", "N/A"))
        with col2:
            st.markdown("##### 🤖 Got")
            st.info(ans.get("generated_answer", "N/A")[:500])
    else:
        st.info("Run 'generate' step first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: DSPy PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("DSPy Configuration & Parameters")

    # ── Signatures ────────────────────────────────────────────
    st.subheader("DSPy Signatures (Typed I/O Contracts)")

    signatures = {
        "CRABasicQA": {
            "doc": "Answer a CRA question using retrieved context. Expert ESG/Climate Risk analyst.",
            "inputs": {"question": "CRA questionnaire question", "context": "Retrieved passages from Shell reports"},
            "outputs": {"answer": "Precise answer grounded in context with source citations"},
        },
        "CRAExtendedQA": {
            "doc": "Detailed ESG answer with evidence-based explanation. Cite page numbers and sources.",
            "inputs": {"question": "CRA question", "context": "Retrieved passages", "answer_options": "Available options"},
            "outputs": {"answer": "Selected answer or direct answer", "explanation": "Evidence-based explanation"},
        },
        "CRAYesNoQA": {
            "doc": "Yes/No CRA question with brief evidence.",
            "inputs": {"question": "CRA Yes/No question", "context": "Retrieved context"},
            "outputs": {"answer": "Yes or No with qualifier", "evidence": "Brief justification"},
        },
        "CRANumericQA": {
            "doc": "Numeric/quantitative CRA answer with calculation trace.",
            "inputs": {"question": "CRA numeric question", "context": "Retrieved context"},
            "outputs": {"answer": "Numeric result with units", "calculation": "Step-by-step computation"},
        },
        "GroundednessJudge": {
            "doc": "Judge if answer is fully grounded in context. Flag unsupported claims.",
            "inputs": {"question": "Original question", "context": "Retrieved context", "answer": "Candidate answer"},
            "outputs": {"score": "0.0-1.0 groundedness", "unsupported_claims": "List of unsupported claims or 'None'"},
        },
        "QueryRewrite": {
            "doc": "Rewrite query for better retrieval. Add synonyms and regulatory terms.",
            "inputs": {"question": "Original CRA question", "context": "Initial retrieval context"},
            "outputs": {"rewritten_query": "Optimized query", "added_terms": "New terms added"},
        },
        "HyDE": {
            "doc": "Hypothetical Document Embeddings — generate hypothetical answer for retrieval.",
            "inputs": {"question": "CRA question"},
            "outputs": {"hypothetical_answer": "Plausible answer for embedding-based retrieval"},
        },
    }

    for name, sig in signatures.items():
        with st.expander(f"📝 `{name}` — {sig['doc'][:80]}"):
            st.markdown(f"**Docstring:** {sig['doc']}")
            st.markdown("**Input Fields:**")
            for field, desc in sig["inputs"].items():
                st.markdown(f"  - `{field}`: {desc}")
            st.markdown("**Output Fields:**")
            for field, desc in sig["outputs"].items():
                st.markdown(f"  - `{field}`: {desc}")

    # ── Modules ───────────────────────────────────────────────
    st.divider()
    st.subheader("DSPy Modules (10 Available)")

    modules = {
        "predict": ("dspy.Predict", "Direct single-pass prediction. No reasoning chain. Best for easy factual questions.", "CRAExtendedQA"),
        "chain_of_thought": ("dspy.ChainOfThought", "Adds 'reasoning' field before answer. Step-by-step logic. Best for medium-hard questions.", "CRAExtendedQA"),
        "program_of_thought": ("dspy.ProgramOfThought", "Generates Python code to compute answer. Best for numeric calculations.", "CRANumericQA"),
        "react": ("dspy.ReAct", "Reason + Act loop with tool calls. Best for multi-step research.", "CRAExtendedQA"),
        "multi_chain": ("dspy.MultiChainComparison", "Generate N candidate answers, compare, pick best. Best for ambiguous questions.", "CRAExtendedQA"),
        "refine": ("dspy.Refine", "Iterative self-refinement with feedback loop.", "CRAExtendedQA"),
        "best_of_n": ("dspy.BestOfN", "Sample N answers, score each, return highest-scoring.", "CRAExtendedQA"),
        "self_refining_rag": ("Custom: Generate → Judge → Refine", "3-stage: generate, groundedness judge, refine if score < 0.7.", "CRAExtendedQA + GroundednessJudge"),
        "multi_hop_rag": ("Custom: Decompose → Sub-answer → Synthesize", "Breaks complex questions into sub-questions, answers each, synthesizes.", "QueryDecompose + CRABasicQA"),
        "adaptive": ("Custom: Routes by question category", "Routes to optimal module based on question type (yes/no → Predict, numeric → CoT).", "Multiple"),
    }

    for name, (wrapper, desc, sig) in modules.items():
        st.markdown(f"**`{name}`** — {wrapper}")
        st.caption(f"  {desc} | Signature: `{sig}`")

    # ── Optimizers ────────────────────────────────────────────
    st.divider()
    st.subheader("DSPy Optimizers (7 Available)")

    optimizers = {
        "BootstrapFewShot": "Bootstrap demonstrations from training set. Runs module on training examples, keeps successful traces as few-shot demos.",
        "BootstrapFewShotWithRandomSearch": "Bootstrap + random search over combinations of demonstrations.",
        "MIPROv2": "Multi-Instance Progressive Retrieval Optimization. Proposes instruction candidates, bootstraps few-shot examples, runs Bayesian optimization over combinations.",
        "COPRO": "Contextual Prompt Optimization. Generates candidate instructions using LLM, evaluates each on validation set.",
        "SIMBA": "Self-Improving Multi-step Bootstrapped Agent. Iteratively improves multi-step programs.",
        "KNNFewShot": "k-Nearest Neighbor few-shot selection. Picks demonstrations most similar to input.",
        "Ensemble": "Ensemble multiple optimized programs. Runs N optimized variants, combines outputs.",
    }

    for name, desc in optimizers.items():
        st.markdown(f"**`{name}`**: {desc}")

    # ── Metrics ───────────────────────────────────────────────
    st.divider()
    st.subheader("DSPy Evaluation Metrics")

    st.markdown("""
| Metric | Formula | Description |
|--------|---------|-------------|
| **F1** | 2×P×R/(P+R) on token sets | Token overlap between prediction and ground truth |
| **Faithfulness** | pred∩ctx / pred tokens | Fraction of prediction tokens found in context |
| **Combined** | 0.4×F1 + 0.3×Faith + 0.3×LenPenalty | Weighted combination; penalizes very short/long answers |
""")

    # ── Optimization Results ──────────────────────────────────
    st.divider()
    st.subheader("Optimization Sweep Results")

    opt_dir = os.path.join(OUTPUT_DIR, "dspy_optimization")
    if os.path.exists(opt_dir):
        sweep = load_json(os.path.join(opt_dir, "sweep_summary.json"))
        if sweep:
            best = sweep.get("best", {})
            if best:
                st.success(
                    f"🏆 **Best: {best.get('module', '?')} × {best.get('optimizer', '?')} "
                    f"= {best.get('score', 0)*100:.1f}%**"
                )
            results = sweep.get("results", [])
            for r in results:
                with st.expander(f"**{r.get('module', '?')} × {r.get('optimizer', '?')}** — {r.get('avg_val_score', 0)*100:.1f}%"):
                    st.json(r)
        # Individual files
        for fname in sorted(os.listdir(opt_dir)):
            if fname.endswith(".json") and fname != "sweep_summary.json":
                data = load_json(os.path.join(opt_dir, fname))
                if data:
                    with st.expander(f"📄 {fname}"):
                        st.json(data)
    else:
        st.caption("No optimization results yet. DSPy sweep runs in-memory.")
        st.caption("From logs: **CoT + BootstrapFewShot = 56.2%** | **Predict + Bootstrap = 53.7%**")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: AutoRAG CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.header("AutoRAG Hyperparameter Search Space")

    if autorag_yaml:
        st.subheader("Pipeline Configuration (YAML)")
        st.code(autorag_yaml, language="yaml")

        st.divider()
        st.subheader("Search Space Breakdown")

        # Parse and display the search space
        st.markdown("##### Query Expansion (3 strategies)")
        st.markdown("""
| Strategy | Type | LLM | Notes |
|----------|------|-----|-------|
| `pass_query_expansion` | No-op baseline | — | Pass query unchanged |
| `query_decompose` | Sub-question decomposition | gpt-4o-mini | Breaks complex queries into parts |
| `hyde` | Hypothetical Document Embedding | gpt-4o-mini | Generates hypothetical answer, uses as retrieval query |
""")

        st.markdown("##### Retrieval (3 methods)")
        st.markdown("""
| Method | Type | Embedding | Parameters |
|--------|------|-----------|------------|
| `bm25` | Sparse keyword | — | Okapi BM25 with default k1=1.5, b=0.75 |
| `vectordb` | Dense semantic | text-embedding-3-small | 1536d, cosine similarity |
| `hybrid_rrf` | Fusion | Both | rrf_k=60, weight=[0.5, 0.5] |
""")

        st.markdown("##### Passage Reranker (4 options)")
        st.markdown("""
| Reranker | Model | Notes |
|----------|-------|-------|
| `pass_reranker` | No-op | Returns passages as-is |
| `monot5` | monoT5-base | T5-based pointwise reranker |
| `upr` | UPR | Unsupervised Passage Reranker |
| `flashrank` | FlashRank | Lightweight cross-encoder |
""")

        st.markdown("##### Passage Filter (2 strategies)")
        st.markdown("""
| Filter | Threshold | Notes |
|--------|-----------|-------|
| `pass_passage_filter` | — | No filtering |
| `threshold_cutoff` | 0.5 | Drop passages below similarity threshold |
""")

        st.markdown("##### Prompt Maker (3 templates)")
        st.markdown("""
| Template | Strategy |
|----------|----------|
| `fstring` | Basic: "Answer the CRA question based on context" |
| `window_replacement` | ESG analyst persona: "Answer based ONLY on context" |
| `long_context_reorder` | Reordered passages with citation request |
""")

        st.markdown("##### Generator")
        st.markdown("""
| Model | Batch | Metrics |
|-------|-------|---------|
| `gpt-4o-mini` | 4 | BLEU, METEOR, ROUGE, SemScore |
""")

        st.divider()
        st.markdown("##### Total Search Space")
        st.metric("Combinations", "3 × 3 × 4 × 2 × 3 × 1 = 216")
        st.caption("AutoRAG evaluates all combinations and picks the best pipeline per metric.")

    else:
        st.info("Run 'autorag_prep' step to generate the pipeline config.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: CORPUS EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    st.header("Corpus Explorer")

    if corpus_data:
        # Source distribution
        sources = {}
        for c in corpus_data:
            s = c.get("source", "?")
            sources[s] = sources.get(s, 0) + 1

        st.subheader("Corpus Distribution")
        cols = st.columns(len(sources))
        for i, (src, count) in enumerate(sources.items()):
            cols[i].metric(src, f"{count} chunks")

        total_chars = sum(len(c.get("text", "")) for c in corpus_data)
        avg_chars = total_chars / len(corpus_data) if corpus_data else 0
        st.caption(f"Total: {len(corpus_data):,} chunks | {total_chars:,} chars | Avg: {avg_chars:.0f} chars/chunk")

        st.divider()

        # Browse chunks
        sel_source = st.selectbox("Filter by Source", ["All"] + list(sources.keys()), key="corpus_src")
        filtered = corpus_data if sel_source == "All" else [c for c in corpus_data if c.get("source") == sel_source]

        # For CRA, also filter by sheet
        if sel_source == "CRA_XLSX":
            sheets = sorted(set(c.get("metadata", {}).get("sheet", "?") for c in filtered))
            sel_sheet = st.selectbox("Filter by Sheet", ["All"] + sheets, key="corpus_sheet")
            if sel_sheet != "All":
                filtered = [c for c in filtered if c.get("metadata", {}).get("sheet") == sel_sheet]

        search_term = st.text_input("Search in chunks", "", key="corpus_search")
        if search_term:
            filtered = [c for c in filtered if search_term.lower() in c.get("text", "").lower()]

        st.caption(f"Showing {len(filtered)} chunks")

        for i, chunk in enumerate(filtered[:50]):  # Limit display
            meta = chunk.get("metadata", {})
            sheet = meta.get("sheet", "")
            q_id = meta.get("question_id", "")
            chunk_type = meta.get("chunk_type", "")
            label = f"[{chunk.get('source', '?')}]"
            if sheet:
                label += f" {sheet}"
            if q_id:
                label += f" / {q_id}"
            if chunk_type:
                label += f" ({chunk_type})"

            with st.expander(f"Chunk {i+1}: {label} — {len(chunk.get('text', ''))} chars"):
                st.code(chunk.get("text", "")[:2000], language="text")
                st.caption(f"ID: `{chunk.get('id', '?')}` | Source: {chunk.get('source_file', '?')}")
    else:
        st.info("Run 'ingest' step first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9: DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[8]:
    st.header("Root Cause Diagnosis")

    if diag_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", diag_data.get("total_questions", 0))
        c2.metric("Passed", diag_data.get("passed", 0))
        c3.metric("Failed", diag_data.get("failed", 0))
        c4.metric("Borderline", diag_data.get("borderline", 0))

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
                st.write(f"**{sec}**: {stats.get('passed', 0)}/{stats.get('total', 0)} ({pct:.0f}%)")
    else:
        st.info("Run 'diagnose' step first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10: PIPELINE CONFIG — Full Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[9]:
    st.header("Pipeline Configuration & Reproducibility")

    if meta_data:
        args = meta_data.get("args", {})

        st.subheader("Run Arguments")
        st.json(args)

        st.divider()
        st.subheader("Improvements Applied")
        improvements = meta_data.get("improvements", [])
        if improvements:
            imp_details = {
                "P0_table_chunking": "Table-aware XLSX chunking — each Q→A→Comment block = 1 chunk. Preserves decimal scores and single-char answers.",
                "P1_dspy_single_init": "DSPy module instantiated ONCE and reused for all questions. Previously re-created per question (44×).",
                "P2_crossencoder_rerank": "CrossEncoder reranker (ms-marco-MiniLM-L-6-v2) re-scores top candidates after RRF fusion.",
                "P3_concise_prompt": "CRA-format system prompt with terse answer examples. max_tokens reduced 500→300.",
                "P4_actual_contexts_ragas": "RAGAS receives actual retrieved chunks instead of generated_answer[:500]. Fixes all context metrics.",
            }
            for imp in improvements:
                desc = imp_details.get(imp, "")
                st.markdown(f"✅ **{imp}**: {desc}")
        else:
            st.caption("No improvement metadata (likely v2 run)")

        st.divider()
        st.subheader("Version Info")
        st.write(f"**Pipeline version:** {meta_data.get('version', 'v2')}")
        st.write(f"**Timestamp:** {meta_data.get('timestamp', 'N/A')}")

        st.divider()
        st.subheader("Index Configuration")
        if index_meta:
            st.json(index_meta)

        st.divider()
        st.subheader("Reproduce This Run")
        model = args.get("model", "gpt-4o")
        ragas_llm = args.get("ragas_llm", model)
        reranker_flag = " --no-reranker" if args.get("no_reranker") else ""
        dspy_mod = f" --dspy-module {args['dspy_module']}" if args.get("dspy_module") else ""
        cmd = (
            f"python pipeline.py --steps {' '.join(args.get('steps', ['all']))} "
            f"--model {model} --ragas-llm {ragas_llm} "
            f"--embedding-model {args.get('embedding_model', 'text-embedding-3-small')} "
            f"--chunk-size {args.get('chunk_size', 800)} --chunk-overlap {args.get('chunk_overlap', 200)} "
            f"--top-k {args.get('top_k', 8)} --retriever {args.get('retriever', 'hybrid')}"
            f"{reranker_flag}{dspy_mod}"
        )
        st.code(cmd, language="bash")
    else:
        st.info("No pipeline metadata. Run the pipeline first.")
