
from qdrant_client import QdrantClient

def find_min_token_image():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    print("Searching for image with 9 tokens...")
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
            
    for point in points:
        if point.payload.get('vlm_tokens') == 9:
            print(f"\n--- Found Image ---")
            print(f"Instance ID: {point.payload.get('instance_id')}")
            print(f"Image URL: {point.payload.get('image_url')}")
            print(f"Description:\n{point.payload.get('vlm_description')}")
            return

    print("Image with 9 tokens not found (maybe count changed?).")

if __name__ == "__main__":
    find_min_token_image()
