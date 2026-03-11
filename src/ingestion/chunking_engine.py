"""
Parent-Child Chunking Engine
==============================
Implements the parent-child chunking strategy with ChromaDB persistence.
Parent chunks (~2000 chars) provide broad narrative context.
Child chunks (~500 chars) carry specific details + vision summaries.
Child chunks are embedded and stored in ChromaDB; parent chunks are
stored in a local JSON index for fast lookup by parent_id.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import tiktoken

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Dataclasses
# ------------------------------------------------------------------ #
@dataclass
class DocumentMetadata:
    """Comprehensive metadata for document tracking and processing"""
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    total_pages: int = 0
    total_characters: int = 0
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    quality_score: float = 0.0

@dataclass
class ChunkMetadata:
    """Metadata for individual chunks"""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_document_id: str = ""
    chunk_index: int = 0
    token_count: int = 0
    page_start: int = 0
    chunk_type: str = "text"
    semantic_score: float = 0.0

# ------------------------------------------------------------------ #
#  Defaults
# ------------------------------------------------------------------ #
PARENT_CHUNK_TOKENS = 500   # tokens (approx 2000 chars)
CHILD_CHUNK_TOKENS = 125    # tokens (approx 500 chars)
CHILD_OVERLAP_TOKENS = 20
COLLECTION_NAME = "finance_child_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ChunkingEngine:
    """
    1. Splits parsed document elements into parent and child chunks.
    2. Embeds child chunks with sentence-transformers.
    3. Stores child chunks + embeddings in a persistent ChromaDB collection.
    4. Stores parent chunks in a JSON index keyed by parent_id.
    """

    def __init__(
        self,
        chromadb_path: str = "data/chromadb_store",
        parent_index_path: str = "data/chromadb_store/parent_index.json",
        embedding_model_name: str = EMBEDDING_MODEL,
    ):
        self.chromadb_path = chromadb_path
        self.parent_index_path = parent_index_path

        # Ensure directories exist
        os.makedirs(chromadb_path, exist_ok=True)
        os.makedirs(os.path.dirname(parent_index_path), exist_ok=True)

        # Initialize sentence-transformers embedding model
        logger.info("Loading embedding model: %s", embedding_model_name)
        self.embedder = SentenceTransformer(embedding_model_name)

        # Initialize persistent ChromaDB client
        logger.info("Connecting to ChromaDB at: %s", chromadb_path)
        self.chroma_client = chromadb.PersistentClient(
            path=chromadb_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready — %d existing documents.",
            COLLECTION_NAME,
            self.collection.count(),
        )

        # Initialize tiktoken for precise LLM window chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info("Tiktoken initialized for token-aware chunking.")

        # Load existing parent index
        self.parent_index: Dict[str, Dict[str, Any]] = self._load_parent_index()

    # ------------------------------------------------------------------ #
    #  Parent Index Persistence
    # ------------------------------------------------------------------ #
    def _load_parent_index(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.parent_index_path):
            try:
                with open(self.parent_index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Loaded parent index with %d entries.", len(data))
                return data
            except Exception as e:
                logger.error("Failed to load parent index: %s", e)
        return {}

    def _save_parent_index(self) -> None:
        try:
            with open(self.parent_index_path, "w", encoding="utf-8") as f:
                json.dump(self.parent_index, f, ensure_ascii=False, indent=2)
            logger.info("Parent index saved with %d entries.", len(self.parent_index))
        except Exception as e:
            logger.error("Failed to save parent index: %s", e)

    # ------------------------------------------------------------------ #
    #  Token-Aware Hybrid Text Chunking
    # ------------------------------------------------------------------ #
    def _calculate_chunk_quality(self, text: str, token_count: int, target_tokens: int) -> float:
        """Assess quality of extracted text based on density and OCR garbage (0.0 to 1.0)"""
        if not text or token_count == 0:
            return 0.0

        length = len(text)
        # Length quality score (reward blocks that actually fill the target context window)
        length_score = 1.0 - abs(token_count - target_tokens) / target_tokens
        length_score = max(0.0, min(1.0, length_score))

        # Check for excessive special characters (indicates OCR issues)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?;:-\"\n') / max(1, length)
        special_char_score = 1.0 - min(special_char_ratio * 2, 0.5)

        # Completeness score (does it end cleanly?)
        completeness_score = 1.0 if text.rstrip().endswith(('.', '!', '?')) else 0.7

        return (length_score + special_char_score + completeness_score) / 3

    def _hybrid_chunking(
        self,
        text: str,
        target_tokens: int,
        overlap_tokens: int = 0,
    ) -> List[Tuple[str, int]]:
        """
        Hybrid chunking: Uses semantic boundaries (\n\n) where possible, 
        and falls back to character splitting mapped via tiktoken.
        Returns List of (chunk_text, token_count).
        """
        if not text or not text.strip():
            return []

        chunks: List[Tuple[str, int]] = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            p_tokens = len(self.tokenizer.encode(paragraph))
            
            # If paragraph itself is larger than target, flush current & hard split paragraph
            if p_tokens > target_tokens:
                if current_chunk:
                    chunks.append((current_chunk.strip(), current_tokens))
                    current_chunk = ""
                    current_tokens = 0
                
                # Hard split the massive paragraph
                start_char = 0
                while start_char < len(paragraph):
                    # Approx chars per token = 4. Take slightly more to be safe.
                    end_char = min(start_char + (target_tokens * 4), len(paragraph))
                    
                    # Try to break at sentence boundary
                    if end_char < len(paragraph):
                        for sep in [". ", "? ", "! "]:
                            last_sep = paragraph.rfind(sep, start_char + (target_tokens * 2), end_char)
                            if last_sep != -1:
                                end_char = last_sep + len(sep)
                                break
                                
                    sub_text = paragraph[start_char:end_char].strip()
                    sub_tokens = len(self.tokenizer.encode(sub_text))
                    if sub_text:
                        chunks.append((sub_text, sub_tokens))
                        
                    start_char = end_char - (overlap_tokens * 4) if overlap_tokens else end_char
                    if start_char <= end_char - len(sub_text):
                        start_char = end_char
            
            # Else try adding paragraph to current chunk
            elif current_tokens + p_tokens <= target_tokens + overlap_tokens:
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                current_tokens += p_tokens
            else:
                chunks.append((current_chunk.strip(), current_tokens))
                current_chunk = paragraph
                current_tokens = p_tokens

        if current_chunk:
            chunks.append((current_chunk.strip(), current_tokens))

        return chunks

    # ------------------------------------------------------------------ #
    #  Core: Process Parsed Elements into Parent-Child Chunks
    # ------------------------------------------------------------------ #
    def process_elements(
        self,
        elements: List[Dict[str, Any]],
        source_file: str,
    ) -> Dict[str, Any]:
        """
        Process parsed document elements into parent-child chunks.

        Args:
            elements: list of dicts from multi_modal_parser (each has 'type',
                      'text', 'page', and optionally 'image_path', 'vision_summary').
            source_file: original PDF filename.

        Returns:
            dict with 'parents_added', 'children_added' counts.
        """
        logger.info("Processing %d elements from '%s'", len(elements), source_file)

        # --- 1. Aggregate all text for parent chunking ---
        full_text_parts: List[str] = []
        vision_summaries: List[Dict[str, Any]] = []
        element_page_map: List[int] = []

        for elem in elements:
            elem_type = elem.get("type", "text")
            text = elem.get("text", "").strip()
            page = elem.get("page", 0)

            if elem_type in ("image", "table") and elem.get("vision_summary"):
                vision_summaries.append({
                    "summary": elem["vision_summary"],
                    "image_path": elem.get("image_path", ""),
                    "page": page,
                    "type": elem_type,
                })
            if text:
                full_text_parts.append(text)
                element_page_map.append(page)

        full_text = "\n\n".join(full_text_parts)

        # --- 2. Create parent chunks (~500 tokens) ---
        parent_texts_with_tokens = self._hybrid_chunking(full_text, PARENT_CHUNK_TOKENS, overlap_tokens=0)
        parents_added = 0
        children_added = 0

        for p_idx, (parent_text, p_tokens) in enumerate(parent_texts_with_tokens):
            parent_id = str(uuid.uuid4())

            # Store parent in index
            self.parent_index[parent_id] = {
                "text": parent_text,
                "source": source_file,
                "parent_index": p_idx,
                "page_start": element_page_map[0] if element_page_map else 0,
                "token_count": p_tokens,
            }
            parents_added += 1

            # --- 3. Create child chunks (~125 tokens) from parent text ---
            child_texts_with_tokens = self._hybrid_chunking(parent_text, CHILD_CHUNK_TOKENS, CHILD_OVERLAP_TOKENS)

            for c_idx, (child_text, c_tokens) in enumerate(child_texts_with_tokens):
                child_id = str(uuid.uuid4())
                quality_score = self._calculate_chunk_quality(child_text, c_tokens, CHILD_CHUNK_TOKENS)
                
                # Drop completely garbage OCR chunks
                if quality_score < 0.2:
                    logger.debug("Skipping child chunk %s due to low quality score %.2f", child_id, quality_score)
                    continue

                self._upsert_child(
                    child_id=child_id,
                    parent_id=parent_id,
                    text=child_text,
                    source=source_file,
                    child_index=c_idx,
                    chunk_type="text",
                    page=element_page_map[0] if element_page_map else 0,
                    token_count=c_tokens,
                    semantic_score=quality_score
                )
                children_added += 1

        # --- 4. Create child chunks for vision summaries ---
        for vs in vision_summaries:
            parent_id = self._find_nearest_parent(vs["page"])
            child_id = str(uuid.uuid4())
            vs_tokens = len(self.tokenizer.encode(vs["summary"]))
            
            self._upsert_child(
                child_id=child_id,
                parent_id=parent_id,
                text=vs["summary"],
                source=source_file,
                child_index=-1,
                chunk_type=vs["type"],
                page=vs["page"],
                image_path=vs.get("image_path", ""),
                token_count=vs_tokens,
                semantic_score=0.9  # Vision summaries are generated, so high quality
            )
            children_added += 1

        # --- 5. Persist ---
        self._save_parent_index()

        result = {"parents_added": parents_added, "children_added": children_added}
        logger.info(
            "Chunking complete for '%s': %d parents, %d children.",
            source_file, parents_added, children_added,
        )
        return result

    # ------------------------------------------------------------------ #
    #  ChromaDB Upsert
    # ------------------------------------------------------------------ #
    def _upsert_child(
        self,
        child_id: str,
        parent_id: str,
        text: str,
        source: str,
        child_index: int,
        chunk_type: str,
        page: int,
        token_count: int = 0,
        semantic_score: float = 0.0,
        image_path: str = "",
    ) -> None:
        """Embed a child chunk and upsert into ChromaDB."""
        try:
            embedding = self.embedder.encode(text, normalize_embeddings=True).tolist()
            metadata = {
                "parent_id": parent_id,
                "source": source,
                "child_index": child_index,
                "chunk_type": chunk_type,
                "page": page,
                "token_count": token_count,
                "semantic_score": semantic_score,
                "image_path": image_path,
            }
            self.collection.upsert(
                ids=[child_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
            )
        except Exception as e:
            logger.error("Failed to upsert child chunk %s: %s", child_id, e, exc_info=True)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _find_nearest_parent(self, page: int) -> str:
        """Find the parent_id closest to the given page number."""
        if not self.parent_index:
            fallback = str(uuid.uuid4())
            self.parent_index[fallback] = {"text": "", "source": "unknown", "page_start": page}
            return fallback

        best_id = list(self.parent_index.keys())[0]
        best_dist = abs(self.parent_index[best_id].get("page_start", 0) - page)
        for pid, pdata in self.parent_index.items():
            dist = abs(pdata.get("page_start", 0) - page)
            if dist < best_dist:
                best_dist = dist
                best_id = pid
        return best_id

    def get_parent_by_id(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a parent chunk by its ID."""
        return self.parent_index.get(parent_id)

    def get_collection_count(self) -> int:
        """Return the number of child chunks in ChromaDB."""
        return self.collection.count()

    def get_all_children(self) -> Dict[str, Any]:
        """Fetch all child chunks from ChromaDB for BM25 initialization."""
        count = self.get_collection_count()
        if count == 0:
            return {"ids": [], "documents": [], "metadatas": []}
            
        return self.collection.get(
            include=["documents", "metadatas"],
            limit=count
        )

    def query_children(
        self,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> Dict[str, Any]:
        """Query ChromaDB for the nearest child chunks."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def reset_collection(self) -> None:
        """Delete and recreate the collection (use with caution)."""
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.parent_index = {}
        self._save_parent_index()
        logger.warning("ChromaDB collection and parent index have been reset.")
