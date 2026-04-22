"""
core/autorag_runner.py — AutoRAG Integration
=============================================
Uses the actual AutoRAG package API to run full pipeline evaluation.

AutoRAG node types (verified from package):
  query_expansion:  pass_query_expansion, query_decompose, hyde, multi_query_expansion
  retrieval:        bm25 (lexical), vectordb (semantic), hybrid_rrf, hybrid_cc
  passage_reranker: pass_reranker, monot5, cohere, upr, colbert, sentence_transformer,
                    flashrank, flag_embedding, rankgpt, jina, voyageai
  passage_filter:   pass_passage_filter, threshold_cutoff, percentile_cutoff,
                    similarity_threshold_cutoff, recency
  passage_compressor: pass_compressor, tree_summarize, refine, longllmlingua
  passage_augmenter: pass_passage_augmenter, prev_next_augmenter
  prompt_maker:     fstring, chat_fstring, window_replacement, long_context_reorder
  generator:        openai_llm, vllm_api, llama_index_llm

AutoRAG requires:
  - corpus.parquet (doc_id, contents, metadata)
  - qa.parquet (qid, query, retrieval_gt, generation_gt)
  - pipeline.yaml (node configuration)
"""

import os, json, hashlib
from typing import Dict, List, Optional

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARE AUTORAG DATA FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

def build_corpus_parquet(corpus_path: str, output_dir: str) -> str:
    """Convert corpus.json → corpus.parquet for AutoRAG."""
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    records = []
    for chunk in corpus:
        records.append({
            "doc_id": chunk["id"],
            "contents": chunk["text"],
            "metadata": json.dumps({
                "source": chunk.get("source", ""),
                "source_file": chunk.get("source_file", ""),
            }),
        })

    df = pd.DataFrame(records)
    parquet_path = os.path.join(output_dir, "corpus.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"    corpus.parquet: {len(df)} rows → {parquet_path}")
    return parquet_path


def build_qa_parquet(gt_path: str, corpus_path: str, output_dir: str) -> str:
    """Convert ground_truth.json → qa.parquet for AutoRAG.

    Builds retrieval_gt by matching GT answer keywords against corpus chunks.
    """
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    # Build simple keyword-based retrieval ground truth
    records = []
    for qa in ground_truth:
        qid = qa.get("question_id", hashlib.md5(qa["question"].encode()).hexdigest()[:8])

        # Find relevant chunk IDs by keyword matching
        answer_lower = qa["answer"].lower()
        answer_words = set(answer_lower.split())
        relevant_ids = []
        for chunk in corpus:
            chunk_lower = chunk["text"].lower()
            overlap = sum(1 for w in answer_words if w in chunk_lower and len(w) > 4)
            if overlap >= 3:
                relevant_ids.append(chunk["id"])
            if len(relevant_ids) >= 5:
                break

        records.append({
            "qid": qid,
            "query": qa["question"],
            "retrieval_gt": [relevant_ids] if relevant_ids else [[]],
            "generation_gt": [qa["answer"]],
        })

    df = pd.DataFrame(records)
    parquet_path = os.path.join(output_dir, "qa.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"    qa.parquet: {len(df)} rows → {parquet_path}")
    return parquet_path


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE AUTORAG YAML CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_autorag_config(
    output_path: str,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
) -> str:
    """Generate AutoRAG pipeline.yaml with all node types."""

    config = f"""
node_lines:
- node_line_name: retrieve_and_generate
  nodes:
    # ─── Query Expansion ───────────────────────────────────
    - node_type: query_expansion
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
        speed_threshold: 10
      modules:
        - module_type: pass_query_expansion
        - module_type: query_decompose
          generator_module_type: openai_llm
          llm: {llm_model}
        - module_type: hyde
          generator_module_type: openai_llm
          llm: {llm_model}
          max_token: 128

    # ─── Retrieval ─────────────────────────────────────────
    - node_type: retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        speed_threshold: 10
      top_k: 10
      modules:
        - module_type: bm25
        - module_type: vectordb
          embedding_model: {embedding_model}
        - module_type: hybrid_rrf
          target_modules:
            - bm25
            - vectordb
          rrf_k: 60
          weight: [0.5, 0.5]

    # ─── Passage Reranker ──────────────────────────────────
    - node_type: passage_reranker
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
        speed_threshold: 10
      top_k: 5
      modules:
        - module_type: pass_reranker
        - module_type: monot5
        - module_type: upr
          use_progress_bar: false
        - module_type: flashrank

    # ─── Passage Filter (optional) ─────────────────────────
    - node_type: passage_filter
      strategy:
        metrics: [retrieval_f1, retrieval_precision]
      modules:
        - module_type: pass_passage_filter
        - module_type: threshold_cutoff
          threshold: 0.5

    # ─── Prompt Maker ──────────────────────────────────────
    - node_type: prompt_maker
      strategy:
        metrics: [bleu, meteor, rouge]
        speed_threshold: 10
      modules:
        - module_type: fstring
          prompt: "Answer the following CRA question based on the context.\\n\\nContext: {{retrieved_contents}}\\n\\nQuestion: {{query}}\\n\\nAnswer:"
        - module_type: window_replacement
          prompt: "You are an ESG analyst. Answer based ONLY on the context.\\n\\nContext: {{retrieved_contents}}\\n\\nQuestion: {{query}}\\n\\nAnswer:"
        - module_type: long_context_reorder
          prompt: "Context passages (reordered for relevance):\\n{{retrieved_contents}}\\n\\nQuestion: {{query}}\\n\\nProvide a precise answer with citations:"

    # ─── Generator ─────────────────────────────────────────
    - node_type: generator
      strategy:
        metrics: [bleu, meteor, rouge, sem_score]
        speed_threshold: 10
      modules:
        - module_type: openai_llm
          llm: {llm_model}
          batch: 4
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config.strip())

    print(f"    AutoRAG config → {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# RUN AUTORAG EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_autorag_evaluation(
    qa_path: str,
    corpus_path: str,
    config_path: str,
    project_dir: str,
) -> Dict:
    """Run AutoRAG Evaluator with the configured pipeline."""
    os.makedirs(project_dir, exist_ok=True)

    try:
        from autorag.evaluator import Evaluator

        evaluator = Evaluator(
            qa_data_path=qa_path,
            corpus_data_path=corpus_path,
            project_dir=project_dir,
        )

        print(f"    Running AutoRAG Evaluator...")
        print(f"    QA: {qa_path}")
        print(f"    Corpus: {corpus_path}")
        print(f"    Config: {config_path}")
        print(f"    Project: {project_dir}")

        evaluator.start_trial(config_path)

        # Load results
        result = {
            "status": "ok",
            "project_dir": project_dir,
            "config": config_path,
        }

        # Check for trial results
        trial_dirs = [d for d in os.listdir(project_dir) if d.startswith("trial")]
        if trial_dirs:
            latest_trial = sorted(trial_dirs)[-1]
            trial_path = os.path.join(project_dir, latest_trial)
            result["trial_dir"] = trial_path

            # Load summary if available
            summary_path = os.path.join(trial_path, "summary.csv")
            if os.path.exists(summary_path):
                summary_df = pd.read_csv(summary_path)
                result["summary"] = summary_df.to_dict('records')

            # Load best config
            best_path = os.path.join(trial_path, "best_config.yaml")
            if os.path.exists(best_path):
                with open(best_path, encoding="utf-8") as f:
                    result["best_config"] = f.read()

        print(f"    AutoRAG evaluation complete.")
        return result

    except Exception as e:
        print(f"    ⚠️ AutoRAG evaluation failed: {e}")
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# DEPLOY BEST AUTORAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def deploy_best_pipeline(project_dir: str) -> Optional[object]:
    """Load and deploy the best AutoRAG pipeline from trial results."""
    try:
        from autorag.deploy import Runner

        trial_dirs = [d for d in os.listdir(project_dir) if d.startswith("trial")]
        if not trial_dirs:
            print("    No trial results found.")
            return None

        latest_trial = os.path.join(project_dir, sorted(trial_dirs)[-1])
        runner = Runner.from_trial_folder(latest_trial)
        print(f"    Deployed best pipeline from {latest_trial}")
        return runner

    except Exception as e:
        print(f"    ⚠️ Deployment failed: {e}")
        return None


def query_deployed_pipeline(runner, query: str) -> Dict:
    """Query the deployed AutoRAG pipeline."""
    try:
        result = runner.run(query)
        return {
            "query": query,
            "answer": result.get("answer", ""),
            "retrieved_passages": result.get("retrieved_passages", []),
            "metadata": result.get("metadata", {}),
        }
    except Exception as e:
        return {"query": query, "error": str(e)}
