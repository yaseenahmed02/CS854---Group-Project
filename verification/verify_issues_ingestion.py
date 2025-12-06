import sys
import os
import time
from qdrant_client import QdrantClient

# Add project root to path
sys.path.insert(0, os.getcwd())

def verify_issues_ingestion():
    """
    Verify that issues were correctly ingested into 'swe_bench_issues' collection.
    """
    print("Verifying Issue Ingestion...")
    
    db_path = "data/qdrant/qdrant_data_swe_bench_issues"
    client = QdrantClient(path=db_path)
    collection_name = "swe_bench_issues"
    
    if not client.collection_exists(collection_name):
        print(f"ERROR: Collection {collection_name} does not exist at {db_path}")
        return False
        
    info = client.get_collection(collection_name)
    print(f"Collection {collection_name} has {info.points_count} points.")
    
    if info.points_count == 0:
        print("ERROR: Collection is empty.")
        return False
        
    # Fetch one point to check payload and vectors
    points = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=True
    )[0]
    
    if not points:
        print("ERROR: Could not retrieve any points.")
        return False
        
    point = points[0]
    payload = point.payload
    vectors = point.vector
    
    print("\nSample Point Payload:")
    for k, v in payload.items():
        val_str = str(v)
        if len(val_str) > 100:
            val_str = val_str[:100] + "..."
        print(f"  {k}: {val_str}")
        
    # Checks
    required_fields = ['instance_id', 'repo', 'version', 'problem_statement', 'issue_tokens', 'embedding_time_ms', 
                       'embedding_time_ms_jina', 'embedding_time_ms_splade', 'embedding_time_ms_bge']
    missing_fields = [f for f in required_fields if f not in payload]
    
    if missing_fields:
        print(f"ERROR: Missing payload fields: {missing_fields}")
        return False
        
    if 'issue_tokens' in payload:
        print(f"SUCCESS: 'issue_tokens' found: {payload['issue_tokens']}")
    else:
        print("ERROR: 'issue_tokens' NOT found in payload.")
        return False

    print("\nVector Checks:")
    if isinstance(vectors, dict):
        print(f"  Vector Keys: {list(vectors.keys())}")
        if 'dense_jina' in vectors and 'splade' in vectors and 'bge' in vectors:
            print("SUCCESS: All expected vector types (dense_jina, splade, bge) are present.")
        else:
            print("ERROR: Missing some vector types.")
            return False
    else:
        print("ERROR: Vectors are not a dictionary (named vectors expected).")
        return False
        
    print("\nVerification PASSED!")
    return True

if __name__ == "__main__":
    verify_issues_ingestion()
