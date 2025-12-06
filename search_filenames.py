
from qdrant_client import QdrantClient

def search_filenames():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    targets = [
        "vertical.png",
        "horizontal.png",
        "default.png",
        "grid-lines-scriptable.png",
        "grid-lines-index-axis-y.png",
        "grid-lines-index-axis-x.png"
    ]
    
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
            
    print(f"Scanning {len(points)} points for targets...")
    
    for p in points:
        url = p.payload.get("image_url", "")
        for t in targets:
            if url.endswith(t):
                print(f"Found {t}:")
                print(f"  URL: {url}")
                print(f"  Instance: {p.payload.get('instance_id')}")
                print("-" * 20)

if __name__ == "__main__":
    search_filenames()
