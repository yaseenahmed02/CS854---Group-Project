
from qdrant_client import QdrantClient

def check_all_descriptions():
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
            
    print(f"Total points found: {len(points)}")
    
    missing_desc = 0
    for point in points:
        desc = point.payload.get('vlm_description')
        if not desc:
            missing_desc += 1
            print(f"Missing description for {point.payload.get('instance_id')}")
            
    if missing_desc == 0:
        print("SUCCESS: All images have descriptions.")
    else:
        print(f"FAILURE: {missing_desc} images are missing descriptions.")

if __name__ == "__main__":
    check_all_descriptions()
