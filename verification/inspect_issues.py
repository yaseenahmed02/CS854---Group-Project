import sys
import os
from qdrant_client import QdrantClient

# Add project root to path
sys.path.insert(0, os.getcwd())

def inspect_issues():
    """
    Inspect 'swe_bench_issues' collection.
    """
    print("--- Inspecting 'swe_bench_issues' Collection ---")
    
    db_path = "data/qdrant/qdrant_data_swe_bench_issues"
    client = QdrantClient(path=db_path)
    collection_name = "swe_bench_issues"
    
    if not client.collection_exists(collection_name):
        print(f"ERROR: Collection {collection_name} does not exist.")
        return
        
    info = client.get_collection(collection_name)
    count = info.points_count
    print(f"Total Points: {count}")
    
    if count != 102:
        print(f"WARNING: Expected 102 points, found {count}.")
    else:
        print("SUCCESS: Count matches expected (102).")
        
    print("\n--- First 5 Entries ---")
    points = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=True,
        with_vectors=False
    )[0]
    
    for i, point in enumerate(points):
        print(f"\nEntry {i+1}:")
        payload = point.payload
        print(f"  ID: {payload.get('instance_id')}")
        print(f"  Repo: {payload.get('repo')}")
        print(f"  Version: {payload.get('version')}")
        print(f"  Issue Tokens: {payload.get('issue_tokens')}")
        print(f"  Embedding Time (Total): {payload.get('embedding_time_ms'):.2f} ms")
        print(f"  Embedding Time (Jina): {payload.get('embedding_time_ms_jina'):.2f} ms")
        print(f"  Embedding Time (SPLADE): {payload.get('embedding_time_ms_splade'):.2f} ms")
        print(f"  Embedding Time (BGE): {payload.get('embedding_time_ms_bge'):.2f} ms")
        
        # Print snippet of problem statement
        ps = payload.get('problem_statement', '')
        snippet = ps[:100].replace('\n', ' ') + "..." if len(ps) > 100 else ps
        print(f"  Problem Statement: {snippet}")

if __name__ == "__main__":
    inspect_issues()
