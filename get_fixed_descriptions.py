
from qdrant_client import QdrantClient

def get_fixed_descriptions():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    target_suffixes = [
        "test/fixtures/core.scale/border-behind-elements.png",
        "test/fixtures/core.scale/grid-lines-scriptable.png",
        "test/fixtures/core.scale/grid-lines-index-axis-y.png",
        "test/fixtures/core.scale/grid-lines-index-axis-x.png",
        "test/fixtures/controller.bar/horizontal-borders.png",
        "test/fixtures/controller.line/point-style.png",
        "test/fixtures/controller.bar/borderRadius/border-radius.png",
        "test/fixtures/controller.line/clip/default-y-max.png",
        "test/fixtures/controller.line/pointBorderWidth/value.png",
        "test/fixtures/controller.bar/borderSkipped/vertical.png",
        "test/fixtures/controller.bar/borderSkipped/horizontal.png",
        "test/fixtures/controller.line/clip/default.png"
    ]
    
    print("Fetching all points to match suffixes...")
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
            
    print(f"Total points scanned: {len(points)}\n")
    
    found_count = 0
    for suffix in target_suffixes:
        matched = False
        for point in points:
            url = point.payload.get('image_url', '')
            if url.endswith(suffix):
                print(f"--- Image: .../{suffix} ---")
                print(f"Instance ID: {point.payload.get('instance_id')}")
                print(f"Description:\n{point.payload.get('vlm_description')}\n")
                matched = True
                found_count += 1
                break # Move to next suffix
        
        if not matched:
            print(f"WARNING: Could not find image ending with {suffix}\n")
            
    print(f"Found {found_count}/{len(target_suffixes)} images.")

if __name__ == "__main__":
    get_fixed_descriptions()
