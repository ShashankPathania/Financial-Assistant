import logging
import time
from src.ingestion.chunking_engine import ChunkingEngine
from src.rag.advanced_retriever import AdvancedRetriever

logging.basicConfig(level=logging.INFO)

print("Loading chunking engine...")
ce = ChunkingEngine(chromadb_path="data/chromadb_store", parent_index_path="data/chromadb_store/parent_index.json")

print("Initializing Advanced Retriever (Should load BM25)...")
t0 = time.time()
retriever = AdvancedRetriever(chunking_engine=ce, top_k=3)
t1 = time.time()
print(f"Retriever Init took {t1-t0:.2f}s")

print("Running test query via Hybrid Search...")
test_query = "What are the risk factors and revenue performance for the latest quarter?"
t2 = time.time()
results = retriever.retrieve(test_query)
t3 = time.time()

print(f"Retrieval took {t3-t2:.2f}s")
print(f"Found {len(results)} results.")

for i, res in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Score (RRF cross-encoded): {res.get('score', 0)}")
    print(f"Source: {res.get('source', '')} - Page {res.get('page', 0)}")
    print(f"Preview: {res.get('child_text', '')[:200]}...")
