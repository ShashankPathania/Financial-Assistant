"""
Advanced Retriever
===================
Implements HyDE (Hypothetical Document Embeddings) and Multi-Query Expansion
using Groq, then searches ChromaDB child chunks and fetches associated parent
chunks for broader context.  Deduplicates results by parent_id.

Includes cross-encoder re-ranking for precision improvement.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.ingestion.chunking_engine import ChunkingEngine

load_dotenv()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Minimum bi-encoder similarity to keep a candidate
MIN_SIMILARITY_THRESHOLD = 0.30

# Minimum cross-encoder score to keep a result after re-ranking
MIN_RERANK_SCORE = 0.3


class AdvancedRetriever:
    """
    Retrieval pipeline:
       1. Multi-Query Expansion (Groq) — generate 3 rephrasings
       2. HyDE (Groq) — generate a hypothetical answer, embed it
       3. Search ChromaDB with all embeddings
       4. Score threshold filter (drop low-similarity noise)
       5. Cross-encoder re-ranking for precision
       6. Fetch parent chunks, de-duplicate, return top results
    """

    def __init__(
        self,
        chunking_engine: ChunkingEngine,
        groq_model: str = "",
        top_k: int = 10,
    ):
        self.chunking_engine = chunking_engine
        self.top_k = top_k

        # Groq client for query expansion / HyDE
        api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_client = Groq(api_key=api_key) if api_key else None
        self.groq_model = groq_model or os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

        # Embedding model (shared with chunking engine for consistency)
        self.embedder: SentenceTransformer = chunking_engine.embedder

        # Cross-encoder re-ranker for precision improvement
        logger.info("Loading cross-encoder re-ranker: %s", RERANKER_MODEL)
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
            logger.info("Cross-encoder re-ranker loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load cross-encoder: %s — re-ranking disabled.", e)
            self.reranker = None

        logger.info("AdvancedRetriever initialized (model=%s, top_k=%d)", self.groq_model, top_k)

    # ------------------------------------------------------------------ #
    #  Multi-Query Expansion
    # ------------------------------------------------------------------ #
    def _expand_queries(self, original_query: str) -> List[str]:
        """Use Groq to generate 3 alternative phrasings of the user query."""
        queries = [original_query]

        if not self.groq_client:
            logger.warning("Groq client unavailable — skipping multi-query expansion.")
            return queries

        prompt = (
            "You are a financial research assistant. Given the user query below, "
            "generate exactly 3 alternative phrasings that capture the same intent "
            "but use different keywords and perspectives. Return ONLY the 3 queries, "
            "one per line, with no numbering or extra text.\n\n"
            f"User query: {original_query}"
        )

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
                timeout=15,
            )
            text = response.choices[0].message.content.strip()
            alt_queries = [q.strip() for q in text.split("\n") if q.strip()]
            queries.extend(alt_queries[:3])
            logger.info("Multi-query expansion produced %d total queries.", len(queries))
        except Exception as e:
            logger.error("Multi-query expansion failed: %s", e)

        return queries

    # ------------------------------------------------------------------ #
    #  HyDE — Hypothetical Document Embeddings
    # ------------------------------------------------------------------ #
    def _generate_hyde_document(self, query: str) -> Optional[str]:
        """Use Groq to generate a hypothetical answer to embed."""
        if not self.groq_client:
            logger.warning("Groq client unavailable — skipping HyDE generation.")
            return None

        prompt = (
            "You are a financial analyst. Write a short, factual paragraph (3-5 sentences) "
            "that would be a perfect answer to the following question. Write as if you are "
            "quoting from an official financial document or annual report.\n\n"
            f"Question: {query}"
        )

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300,
                timeout=15,
            )
            hyde_doc = response.choices[0].message.content.strip()
            logger.info("HyDE document generated (%d chars).", len(hyde_doc))
            return hyde_doc
        except Exception as e:
            logger.error("HyDE generation failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    #  Cross-Encoder Re-ranking
    # ------------------------------------------------------------------ #
    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Re-rank candidates using the cross-encoder for higher precision."""
        if not self.reranker or not candidates:
            return candidates

        try:
            # Build (query, child_text) pairs for the cross-encoder
            pairs = [(query, c["child_text"]) for c in candidates]
            scores = self.reranker.predict(pairs)

            # Attach cross-encoder scores
            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = float(scores[i])

            # Filter by minimum re-rank score and sort descending
            filtered = [c for c in candidates if c["rerank_score"] > MIN_RERANK_SCORE]

            # If filtering removed everything, keep at least top 3
            if not filtered and candidates:
                filtered = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:3]

            filtered.sort(key=lambda x: x["rerank_score"], reverse=True)

            logger.info(
                "Re-ranking: %d → %d candidates (kept those above %.2f score).",
                len(candidates), len(filtered), MIN_RERANK_SCORE,
            )
            return filtered

        except Exception as e:
            logger.error("Cross-encoder re-ranking failed: %s — returning original order.", e)
            return candidates

    # ------------------------------------------------------------------ #
    #  Core Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline.

        Returns:
            List of result dicts, each containing:
              - parent_text: the parent chunk text
              - child_text: the matching child chunk text
              - source: source filename
              - page: page number
              - image_path: path to extracted image (if any)
              - score: similarity score
        """
        k = top_k or self.top_k
        all_child_ids: Dict[str, float] = {}  # child_id -> best score

        # Step 1: Build query set (original + expanded + HyDE)
        queries = self._expand_queries(query)

        hyde_doc = self._generate_hyde_document(query)
        embeddings_to_search: List[List[float]] = []

        # Embed all query variants
        for q in queries:
            emb = self.embedder.encode(q, normalize_embeddings=True).tolist()
            embeddings_to_search.append(emb)

        # Embed the HyDE document
        if hyde_doc:
            emb = self.embedder.encode(hyde_doc, normalize_embeddings=True).tolist()
            embeddings_to_search.append(emb)

        # Step 2: Query ChromaDB with each embedding
        for emb in embeddings_to_search:
            try:
                results = self.chunking_engine.query_children(
                    query_embedding=emb,
                    n_results=k,
                )
                if results and results.get("ids") and results["ids"][0]:
                    for idx, cid in enumerate(results["ids"][0]):
                        dist = results["distances"][0][idx] if results.get("distances") else 1.0
                        score = 1.0 - dist  # cosine distance to similarity
                        if cid not in all_child_ids or score > all_child_ids[cid]:
                            all_child_ids[cid] = score
            except Exception as e:
                logger.error("ChromaDB query failed: %s", e)

        if not all_child_ids:
            logger.warning("No results found for query: %s", query[:100])
            return []

        # Step 3: Apply similarity threshold to filter noise
        before_count = len(all_child_ids)
        all_child_ids = {
            cid: score for cid, score in all_child_ids.items()
            if score >= MIN_SIMILARITY_THRESHOLD
        }
        if len(all_child_ids) < before_count:
            logger.info(
                "Score threshold (%.2f): filtered %d → %d candidates.",
                MIN_SIMILARITY_THRESHOLD, before_count, len(all_child_ids),
            )

        if not all_child_ids:
            logger.warning("All candidates below threshold for query: %s", query[:100])
            return []

        # Step 4: Fetch child metadata + parent chunks, deduplicate by parent_id
        seen_parents: set = set()
        candidates: List[Dict[str, Any]] = []

        # Sort by score descending
        sorted_children = sorted(all_child_ids.items(), key=lambda x: x[1], reverse=True)

        for child_id, score in sorted_children[:k * 3]:  # fetch more for re-ranking
            try:
                child_data = self.chunking_engine.collection.get(
                    ids=[child_id],
                    include=["documents", "metadatas"],
                )
                if not child_data or not child_data["ids"]:
                    continue

                meta = child_data["metadatas"][0] if child_data["metadatas"] else {}
                child_text = child_data["documents"][0] if child_data["documents"] else ""
                parent_id = meta.get("parent_id", "")

                # Deduplicate by parent_id (keep best-scoring child per parent)
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                parent_data = self.chunking_engine.get_parent_by_id(parent_id)
                parent_text = parent_data.get("text", "") if parent_data else ""

                candidates.append({
                    "parent_text": parent_text,
                    "child_text": child_text,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "image_path": meta.get("image_path", ""),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "score": round(score, 4),
                    "parent_id": parent_id,
                })

            except Exception as e:
                logger.error("Error fetching child %s: %s", child_id, e)

        # Step 5: Cross-encoder re-ranking for precision
        results = self._rerank(query, candidates)

        # Trim to top_k
        results = results[:k]
        logger.info("Retrieved %d unique parent contexts for query.", len(results))
        return results
