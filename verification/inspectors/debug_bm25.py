import sys
import os
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

# Add project root to path
sys.path.insert(0, os.getcwd())

def tokenize(text: str):
    clean_text = "".join([c if c.isalnum() else " " for c in text.lower()])
    return [t for t in clean_text.split() if t]

def debug_bm25():
    db_path = "data/qdrant/qdrant_data_test_repo_0_0_1"
    collection_name = "test_repo_0_0_1"
    
    print(f"Connecting to {db_path}...")
    client = QdrantClient(path=db_path)
    
    print(f"Fetching documents from {collection_name}...")
    chunks = []
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    
    for point in points:
        chunk = point.payload
        chunks.append(chunk)
        
    print(f"Found {len(chunks)} chunks.")
    
    if not chunks:
        print("No chunks found!")
        return

    print("Sample chunk text:")
    print(chunks[0].get('text', 'N/A'))
    
    print("Tokenizing...")
    tokenized_corpus = [tokenize(chunk.get('text', '')) for chunk in chunks]
    print(f"Tokenized corpus: {tokenized_corpus}")
    
    print("Initializing BM25Okapi...")
    try:
        bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 initialized successfully.")
    except Exception as e:
        print(f"BM25 initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_bm25()
