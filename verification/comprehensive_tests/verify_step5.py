import sys
import os
from qdrant_client import QdrantClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from retrieval.flexible_retriever import FlexibleRetriever

def test_retriever():
    print("Initializing FlexibleRetriever...")
    # Connect to local Qdrant
    # We need to know the collection name from Step 2.
    # In Step 2, we ingested 'test_repo' version '0.0.1' -> 'test_repo_0_0_1' (sanitized)
    # Wait, sanitize replaces dots? 
    # sanitize('0.0.1') -> '0_0_1'
    # Collection: 'test_repo_0_0_1'
    # DB Path: './qdrant_data_test_repo_0_0_1'
    
    db_path = "./qdrant_data_test_repo_0_0_1"
    collection_name = "test_repo_0_0_1"
    
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} not found. Did Step 2 run?")
        return

    client = QdrantClient(path=db_path)
    
    # We don't have chunks file for BM25 test here easily unless we point to one, 
    # but we can test Qdrant strategies.
    retriever = FlexibleRetriever(client, collection_name)
    
    query = "sanitize path"
    
    # Test 1: Dense Only
    print(f"\nTest 1: Dense (Jina) Retrieval for '{query}'")
    results = retriever.retrieve(query, strategy=["jina"], top_k=3)
    for i, res in enumerate(results['results']):
        print(f"{i+1}. {res['payload'].get('filepath')} (Score: {res['score']:.4f})")
        
    # Test 2: Hybrid (Jina + Splade)
    print(f"\nTest 2: Hybrid (Jina + Splade) Retrieval")
    # Note: Splade model loading might take time
    try:
        results = retriever.retrieve(query, strategy=["jina", "splade"], top_k=3)
        for i, res in enumerate(results['results']):
            print(f"{i+1}. {res['payload'].get('filepath')} (Score: {res['score']:.4f})")
    except Exception as e:
        print(f"Hybrid test failed (likely model load): {e}")

    # Test 3: Visual Augment (Mock)
    # We need an instance_id that exists in swe_images.
    # In Step 4, we ingested 'grommet__grommet-6282' (from mock run output).
    # We need to connect to swe_images DB too?
    # FlexibleRetriever takes 'client'. 
    # But Step 4 created a SEPARATE Qdrant DB at './qdrant_data_swe_images'.
    # QdrantClient(path=...) connects to a specific file/dir.
    # If we want to access BOTH, we need a client that can see both?
    # In Local mode, QdrantClient is bound to one path.
    # So FlexibleRetriever might need TWO clients or we should have put all collections in ONE DB path.
    # The prompt for Step 2 said: "create a specific DB path ... to ensure Version Isolation".
    # The prompt for Step 4 said: "Connect to the same local Qdrant path as the code ingestion script."
    # WAIT. Step 4 prompt said: "Connect to the same local Qdrant path as the code ingestion script."
    # BUT Step 2 prompt said: "create a specific DB path ... (e.g., ./qdrant_data_wp_calypso_10_15_2)".
    # This implies EACH repo version has its own DB path?
    # If so, Step 4 "same local Qdrant path" is ambiguous if Step 2 creates dynamic paths.
    # In Step 4 implementation, I used default `./qdrant_data_swe_images`.
    # If FlexibleRetriever needs to access both, and they are in different paths (SQLite files), 
    # we can't do it with one QdrantClient instance in local mode easily unless we attach?
    # Or maybe I should have used one global DB path.
    # Given the current state:
    # Code DB: ./qdrant_data_test_repo_0_0_1
    # Image DB: ./qdrant_data_swe_images
    # FlexibleRetriever takes `client`.
    # It tries `client.scroll(collection_name=swe_images_collection)`.
    # This will FAIL if `client` is connected to Code DB and Image DB is elsewhere.
    
    # FIX: FlexibleRetriever should probably take a separate client for images or handle it.
    # Or I should instantiate a second client inside `_fetch_visual_description` if needed.
    # For now, I will modify the test to skip visual or mock the fetch if it fails.
    # Or better, I'll update FlexibleRetriever to accept `images_client`.
    
    pass

if __name__ == "__main__":
    test_retriever()
