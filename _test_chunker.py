import json
from src.ingestion.chunking_engine import ChunkingEngine

def test_chunking():
    engine = ChunkingEngine(
        chromadb_path="data/test_chroma", 
        parent_index_path="data/test_chroma/index.json"
    )
    
    dummy_elements = [
        {"type": "text", "text": "This is a very long paragraph that will act as a parent chunk. " * 50, "page": 1},
        {"type": "table", "text": "Header 1\tHeader 2\nData 1\tData 2", "page": 1, "vision_summary": "A table showing data elements."},
        {"type": "text", "text": "Here is another paragraph isolated by a double newline.\n\nAnd here is the second stanza.", "page": 2}
    ]
    
    result = engine.process_elements(dummy_elements, "dummy.pdf")
    print(f"Chunks generated: {result}")
    
    # Verify chroma has the right metadata
    res = engine.collection.get(limit=5)
    print("Metadata Check:")
    for m in res['metadatas']:
        print(m)

if __name__ == "__main__":
    test_chunking()
