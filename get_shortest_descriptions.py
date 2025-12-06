
from qdrant_client import QdrantClient

def get_shortest_descriptions():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    print("Fetching all points...")
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
            
    # Sort by token count
    sorted_points = sorted(points, key=lambda p: p.payload.get('vlm_tokens', 0))
    
    print("\n--- Top 5 Shortest Descriptions ---\n")
    for i, point in enumerate(sorted_points[:5]):
        print(f"#{i+1} | Tokens: {point.payload.get('vlm_tokens')}")
        print(f"Instance ID: {point.payload.get('instance_id')}")
        print(f"Image URL: {point.payload.get('image_url')}")
        print(f"Description:\n{point.payload.get('vlm_description')}\n")
        print("-" * 40)

if __name__ == "__main__":
    get_shortest_descriptions()
