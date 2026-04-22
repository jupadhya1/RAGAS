"""
indexing/build_index.py — Hybrid Vector Store (v3 — All Improvements)
=====================================================================
IMPROVEMENTS:
  P2: CrossEncoder reranking after RRF fusion (re-scores top candidates)
  NEW: Embedding cache to disk (skip re-embedding on unchanged corpus)
  NEW: Configurable RRF k parameter
  NEW: Reranker model auto-downloads from sentence-transformers

Retrieval pipeline:
  1. BM25 (sparse) → top fetch_k candidates
  2. OpenAI embeddings (dense) → top fetch_k candidates
  3. RRF fusion → merge & rank
  4. CrossEncoder rerank → re-score top candidates (P2)
  5. Return top_k results
"""

import os, json, math, re, pickle, hashlib
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 INDEX
# ═══════════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "its", "it", "this", "that", "these",
    "those", "their", "they", "them", "there", "what", "which", "who", "whom",
    "how", "when", "where", "why", "whether", "not", "also", "such", "each",
    "other", "some", "any", "all", "both", "than", "more", "most", "very",
    "just", "about", "between", "under", "upon", "only", "much", "our", "we", "us",
}


def tokenize(text: str) -> List[str]:
    return [
        t
        for t in re.findall(r"\b[a-z0-9]+(?:\.[0-9]+)?\b", text.lower())
        if t not in STOPWORDS and len(t) > 2
    ]


class BM25Index:
    """Okapi BM25 sparse retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = Counter()
        self.doc_lens = []
        self.doc_tokens = []
        self.avg_dl = 0
        self.n_docs = 0

    def build(self, documents: List[str]):
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.n_docs = len(documents)
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / max(self.n_docs, 1)
        for tokens in self.doc_tokens:
            for t in set(tokens):
                self.doc_freqs[t] += 1

    def query(self, q: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q_tokens = tokenize(q)
        scores = []
        for i, doc_tokens in enumerate(self.doc_tokens):
            score = 0.0
            dl = self.doc_lens[i]
            tf_counter = Counter(doc_tokens)
            for qt in q_tokens:
                if qt not in self.doc_freqs:
                    continue
                df = self.doc_freqs[qt]
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
                tf = tf_counter.get(qt, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                score += idf * num / den
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING INDEX (with disk cache)
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingIndex:
    """Dense retriever using OpenAI embeddings with disk caching."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.embeddings = []
        self.dim = 0

    def build(self, documents: List[str], batch_size: int = 100, cache_dir: str = None):
        """Build embedding index. Uses disk cache if available."""
        # ── Check cache ───────────────────────────────────────
        if cache_dir:
            corpus_hash = hashlib.md5(
                "".join(d[:50] for d in documents).encode()
            ).hexdigest()[:16]
            cache_path = os.path.join(
                cache_dir, f"embeddings_{self.model.replace('/', '_')}_{corpus_hash}.npy"
            )
            if os.path.exists(cache_path):
                self.embeddings = np.load(cache_path).tolist()
                self.dim = len(self.embeddings[0]) if self.embeddings else 0
                print(f"      ✓ Loaded cached embeddings: {len(self.embeddings)} docs, dim={self.dim}")
                return

        # ── Compute embeddings ────────────────────────────────
        import openai

        client = openai.OpenAI()
        self.embeddings = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch = [d[:8000] for d in batch]
            try:
                response = client.embeddings.create(input=batch, model=self.model)
                for emb in response.data:
                    self.embeddings.append(emb.embedding)
            except Exception as e:
                print(f"      ⚠️  Embedding batch {i} failed: {e}")
                for _ in batch:
                    self.embeddings.append([0.0] * 1536)

        if self.embeddings:
            self.dim = len(self.embeddings[0])
        print(f"      Embedded {len(self.embeddings)} docs, dim={self.dim}")

        # ── Save cache ────────────────────────────────────────
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, np.array(self.embeddings, dtype=np.float32))
            print(f"      ✓ Cached embeddings to {cache_path}")

    def query(self, q: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Query by cosine similarity."""
        import openai

        client = openai.OpenAI()
        try:
            response = client.embeddings.create(input=[q[:8000]], model=self.model)
            q_emb = response.data[0].embedding
        except Exception:
            return []

        scores = []
        for i, doc_emb in enumerate(self.embeddings):
            dot = sum(a * b for a, b in zip(q_emb, doc_emb))
            norm_q = math.sqrt(sum(a * a for a in q_emb))
            norm_d = math.sqrt(sum(b * b for b in doc_emb))
            sim = dot / (norm_q * norm_d + 1e-10)
            scores.append((i, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# P2: CROSS-ENCODER RERANKER
# ═══════════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    P2: Re-scores (query, chunk) pairs using a cross-attention model.
    Falls back gracefully if sentence-transformers is not installed.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            print(f"      P2: CrossEncoder reranker loaded: {self.model_name}")
        except ImportError:
            print(
                "      ⚠️ sentence-transformers not installed. "
                "Reranking disabled. Install: pip install sentence-transformers"
            )
            self.model = None
        except Exception as e:
            print(f"      ⚠️ CrossEncoder load failed: {e}. Reranking disabled.")
            self.model = None

    def rerank(
        self, query: str, documents: List[str], top_k: int = 8
    ) -> List[Tuple[int, float]]:
        """Re-score documents and return top_k (original_index, score) pairs."""
        if self.model is None:
            # No reranking — return identity ranking
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(documents)))]

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(idx, float(score)) for idx, score in indexed[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID INDEX (RRF FUSION + RERANKING)
# ═══════════════════════════════════════════════════════════════════════════════

class HybridIndex:
    """Reciprocal Rank Fusion of BM25 + Embedding retrieval, with optional reranking."""

    def __init__(
        self,
        bm25: BM25Index,
        embedding: Optional[EmbeddingIndex] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25
        self.embedding = embedding
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.documents = []
        self.corpus_meta = []

    def query(self, q: str, top_k: int = 8) -> List[Dict]:
        """Hybrid retrieval with RRF fusion + optional CrossEncoder rerank."""
        # ── Step 1-2: Fetch candidates from both retrievers ───
        fetch_k = top_k * 3 if self.reranker and self.reranker.model else top_k * 3

        bm25_results = self.bm25.query(q, fetch_k)
        bm25_scores = {
            idx: 1.0 / (self.rrf_k + rank + 1)
            for rank, (idx, _) in enumerate(bm25_results)
        }

        emb_results = []
        emb_scores = {}
        if self.embedding and self.embedding.embeddings:
            emb_results = self.embedding.query(q, fetch_k)
            emb_scores = {
                idx: 1.0 / (self.rrf_k + rank + 1)
                for rank, (idx, _) in enumerate(emb_results)
            }

        # ── Step 3: RRF Fusion ────────────────────────────────
        all_ids = set(bm25_scores.keys()) | set(emb_scores.keys())
        fused = []
        for idx in all_ids:
            score = bm25_scores.get(idx, 0) + emb_scores.get(idx, 0)
            fused.append((idx, score))
        fused.sort(key=lambda x: -x[1])

        # ── Step 4: P2 CrossEncoder Reranking ─────────────────
        if self.reranker and self.reranker.model:
            # Take top candidates for reranking
            rerank_k = min(top_k * 3, len(fused))
            candidates = fused[:rerank_k]
            candidate_texts = [
                self.documents[idx] if idx < len(self.documents) else ""
                for idx, _ in candidates
            ]
            reranked = self.reranker.rerank(q, candidate_texts, top_k)
            # Map back to original indices
            final_indices = [
                (candidates[orig_idx][0], rerank_score)
                for orig_idx, rerank_score in reranked
            ]
        else:
            final_indices = fused[:top_k]

        # ── Step 5: Build result dicts ────────────────────────
        results = []
        for idx, score in final_indices:
            results.append({
                "chunk_id": idx,
                "text": self.documents[idx] if idx < len(self.documents) else "",
                "score": round(score, 6),
                "metadata": (
                    self.corpus_meta[idx] if idx < len(self.corpus_meta) else {}
                ),
                "bm25_rank": next(
                    (r for r, (i, _) in enumerate(bm25_results) if i == idx), -1
                ),
                "emb_rank": (
                    next(
                        (r for r, (i, _) in enumerate(emb_results) if i == idx), -1
                    )
                    if self.embedding
                    else -1
                ),
            })
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def build_index(
    corpus_path: str,
    index_dir: str,
    embedding_model: str = "text-embedding-3-small",
    use_reranker: bool = True,
) -> Dict:
    """Build hybrid retrieval index from corpus."""
    os.makedirs(index_dir, exist_ok=True)

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    documents = [c["text"] for c in corpus]
    metadata = [{"source": c["source"], "id": c["id"]} for c in corpus]

    # Build BM25
    print(f"    Building BM25 index ({len(documents)} docs)...")
    bm25 = BM25Index()
    bm25.build(documents)

    # Build Embedding index (with cache)
    emb_index = None
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk-placeholder"):
        print(f"    Building embedding index ({embedding_model})...")
        emb_index = EmbeddingIndex(model=embedding_model)
        emb_index.build(documents, cache_dir=index_dir)
    else:
        print(f"    ⚠️  No OPENAI_API_KEY — using BM25 only")

    # P2: Build CrossEncoder reranker
    reranker = None
    if use_reranker:
        print(f"    P2: Initializing CrossEncoder reranker...")
        reranker = CrossEncoderReranker()

    # Create hybrid index
    hybrid = HybridIndex(bm25, emb_index, reranker)
    hybrid.documents = documents
    hybrid.corpus_meta = metadata

    # Save index
    index_data = {
        "n_docs": len(documents),
        "has_embeddings": emb_index is not None,
        "embedding_model": embedding_model if emb_index else "none",
        "has_reranker": reranker is not None and reranker.model is not None,
    }
    with open(os.path.join(index_dir, "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    # Pickle the hybrid index for reuse
    with open(os.path.join(index_dir, "hybrid_index.pkl"), "wb") as f:
        pickle.dump(hybrid, f)

    print(
        f"    Index built: {len(documents)} docs, embeddings={'yes' if emb_index else 'no'}, "
        f"reranker={'yes' if reranker and reranker.model else 'no'}"
    )
    return index_data
