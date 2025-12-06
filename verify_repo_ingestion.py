
import os
from datasets import load_dataset
from qdrant_client import QdrantClient
from utils.ingestion_utils import sanitize_path_component

def verify_repo_ingestion():
    print("Loading dataset (dev split)...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="dev")
    
    # Get unique repo/version pairs
    repos = set()
    for instance in dataset:
        repos.add((instance['repo'], instance['version']))
        
    print(f"Found {len(repos)} unique repositories in dev split.")
    
    missing_repos = []
    valid_repos = []
    
    for repo, version in repos:
        collection_name = f"{sanitize_path_component(repo)}_{sanitize_path_component(version)}"
        db_path = f"data/qdrant/qdrant_data_{collection_name}"
        
        if not os.path.exists(db_path):
            missing_repos.append(collection_name)
            print(f"[MISSING] {collection_name}")
            continue
            
        try:
            client = QdrantClient(path=db_path)
            # Check count
            count = client.count(collection_name=collection_name).count
            
            # Check sample point
            points, _ = client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                print(f"[EMPTY] {collection_name} exists but has 0 points.")
                missing_repos.append(f"{collection_name} (EMPTY)")
                continue
                
            point = points[0]
            has_vector = "dense_jina" in point.vector if isinstance(point.vector, dict) else False
            # Note: If point.vector is a list, it's likely a single vector (unnamed). 
            # But our ingestion uses named vectors.
            
            # If vector is not a dict, it might be because we didn't use named vectors?
            # Let's check ingestion_engine.py logic later if this fails.
            # Assuming named vectors for now based on previous context.
            
            payload_keys = point.payload.keys() if point.payload else []
            
            print(f"[OK] {collection_name}: {count} points. Vector: {has_vector}. Payload: {list(payload_keys)}")
            valid_repos.append(collection_name)
            
        except Exception as e:
            print(f"[ERROR] {collection_name}: {e}")
            missing_repos.append(f"{collection_name} (ERROR)")

    print("\n" + "="*30)
    print(f"Summary: {len(valid_repos)}/{len(repos)} repos valid.")
    if missing_repos:
        print(f"Missing/Invalid Repos ({len(missing_repos)}):")
        for r in missing_repos:
            print(f" - {r}")
    else:
        print("All repos appear to be successfully ingested.")

if __name__ == "__main__":
    verify_repo_ingestion()
