import sys
import os
from qdrant_client import QdrantClient

# Add project root to path
sys.path.insert(0, os.getcwd())

def verify_qdrant(repo_name, version):
    safe_repo = repo_name.replace("/", "_")
    safe_version = version.replace(".", "_")
    db_path = f"data/qdrant/qdrant_data_{safe_repo}_{safe_version}"
    collection_name = f"{safe_repo}_{safe_version}"
    
    print(f"Checking Qdrant at: {db_path}")
    print(f"Collection: {collection_name}")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database path {db_path} does not exist.")
        return
        
    client = QdrantClient(path=db_path)
    
    try:
        info = client.get_collection(collection_name)
        print(f"Collection found. Points count: {info.points_count}")
        
        if info.points_count == 0:
            print("WARNING: Collection is empty.")
            return

        # Fetch a sample point
        points = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_vectors=True,
            with_payload=True
        )[0]
        
        if not points:
            print("No points returned from scroll.")
            return
            
        point = points[0]
        print(f"\nSample Point ID: {point.id}")
        
        # Check Vectors
        print("\nVectors:")
        if point.vector:
            if isinstance(point.vector, dict):
                for name, vec in point.vector.items():
                    if hasattr(vec, 'indices'): # Sparse
                        print(f"  - {name}: Sparse Vector (indices={len(vec.indices)}, values={len(vec.values)})")
                    else: # Dense
                        print(f"  - {name}: Dense Vector (length={len(vec)})")
            else:
                print(f"  - Default: {type(point.vector)}")
        else:
            print("  - NO VECTORS FOUND!")
            
        # Check Payload
        print("\nPayload Keys:")
        if point.payload:
            for key, val in point.payload.items():
                val_preview = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                print(f"  - {key}: {val_preview}")
        else:
            print("  - NO PAYLOAD FOUND!")

    except Exception as e:
        print(f"ERROR querying collection: {e}")

if __name__ == "__main__":
    verify_qdrant("markedjs/marked", "0.6")
