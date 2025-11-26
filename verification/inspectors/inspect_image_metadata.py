import sys
import os
from qdrant_client import QdrantClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def inspect_image_metadata():
    db_path = "data/qdrant/qdrant_data_swe_images"
    collection_name = "swe_images"
    
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} not found.")
        return

    print(f"Connecting to Qdrant at {db_path}...")
    client = QdrantClient(path=db_path)
    
    # Check if collection exists
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        print(f"Collection {collection_name} not found.")
        return

    print(f"Fetching points from {collection_name}...")
    
    # Scroll to get some points
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=True,
        with_vectors=True
    )
    
    if not points:
        print("No points found in collection.")
        return

    print(f"\nFound {len(points)} points. Showing metadata for ALL points:\n")
    
    for i, point in enumerate(points):
        print(f"\n--- Point {i+1} (ID: {point.id}) ---")
        
        # Check Vector
        if point.vector:
            # Qdrant python client might return list or numpy array depending on version/config
            # Usually list of floats
            vec_len = len(point.vector)
            print(f"Vector: Present (Length: {vec_len}, First 3: {point.vector[:3]}...)")
        else:
            print("Vector: NOT FOUND")

        payload = point.payload
        for key, value in payload.items():
            if key == "image_base64" and value:
                print(f"{key}: <Base64 Data, length={len(value)}>")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    inspect_image_metadata()
