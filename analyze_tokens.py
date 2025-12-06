
from qdrant_client import QdrantClient
import numpy as np

def analyze_tokens():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    print("Fetching points...")
    points = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False
        )
        points.extend(batch)
        if next_offset is None:
            break
            
    token_counts = []
    for point in points:
        tokens = point.payload.get('vlm_tokens', 0)
        token_counts.append(tokens)
        
    if not token_counts:
        print("No tokens found.")
        return

    print(f"Total images: {len(token_counts)}")
    print(f"Min tokens: {min(token_counts)}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"Average tokens: {sum(token_counts) / len(token_counts):.2f}")
    print(f"Median tokens: {np.median(token_counts):.2f}")

if __name__ == "__main__":
    analyze_tokens()
