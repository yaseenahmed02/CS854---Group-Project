
from qdrant_client import QdrantClient

def find_candidates():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    print("Scanning for candidates (High Tokens + Low Time)...")
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
            
    candidates = []
    for point in points:
        tokens = point.payload.get('vlm_tokens', 0)
        time_ms = point.payload.get('vlm_generation_time_ms', 0)
        
        # Heuristic: Valid descriptions usually > 50 tokens. 
        # Refusals were fast (< 2000ms?). 
        # Valid GPT-4o VLM usually takes > 2-3s.
        if tokens > 30 and time_ms < 2500:
            candidates.append(point)
            
    print(f"Found {len(candidates)} candidates.")
    for p in candidates:
        print(f"ID: {p.id} | Tokens: {p.payload.get('vlm_tokens')} | Time: {p.payload.get('vlm_generation_time_ms')}ms")
        print(f"URL: {p.payload.get('image_url')}")
        print("-" * 20)

if __name__ == "__main__":
    find_candidates()
