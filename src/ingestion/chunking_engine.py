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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Defaults
# ------------------------------------------------------------------ #
PARENT_CHUNK_SIZE = 2000   # characters
CHILD_CHUNK_SIZE = 500
CHILD_OVERLAP = 50
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
    #  Text Splitting Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_text(
        text: str,
        chunk_size: int,
        overlap: int = 0,
    ) -> List[str]:
        """Split text into chunks respecting sentence boundaries where possible."""
        if not text or not text.strip():
            return []

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # Try to break at sentence boundary
            if end < text_len:
                # Look back from `end` for a period, newline, or semicolon
                for sep in [".\n", ".\r", ". ", ";\n", "; ", "\n\n", "\n"]:
                    last_sep = text.rfind(sep, start + chunk_size // 2, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Ensure start always advances to prevent infinite loops
            new_start = end - overlap if overlap else end
            if new_start <= start:
                new_start = start + max(chunk_size // 2, 1)
            start = new_start

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

        # --- 2. Create parent chunks (~2000 chars) ---
        parent_texts = self._split_text(full_text, PARENT_CHUNK_SIZE, overlap=0)
        parents_added = 0
        children_added = 0

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = str(uuid.uuid4())

            # Store parent in index
            self.parent_index[parent_id] = {
                "text": parent_text,
                "source": source_file,
                "parent_index": p_idx,
                "page_start": element_page_map[0] if element_page_map else 0,
            }
            parents_added += 1

            # --- 3. Create child chunks (~500 chars) from parent text ---
            child_texts = self._split_text(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)

            for c_idx, child_text in enumerate(child_texts):
                child_id = str(uuid.uuid4())
                self._upsert_child(
                    child_id=child_id,
                    parent_id=parent_id,
                    text=child_text,
                    source=source_file,
                    child_index=c_idx,
                    chunk_type="text",
                    page=element_page_map[0] if element_page_map else 0,
                )
                children_added += 1

        # --- 4. Create child chunks for vision summaries ---
        for vs in vision_summaries:
            parent_id = self._find_nearest_parent(vs["page"])
            child_id = str(uuid.uuid4())
            self._upsert_child(
                child_id=child_id,
                parent_id=parent_id,
                text=vs["summary"],
                source=source_file,
                child_index=-1,
                chunk_type=vs["type"],
                page=vs["page"],
                image_path=vs.get("image_path", ""),
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
