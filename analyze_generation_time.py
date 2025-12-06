
from qdrant_client import QdrantClient
import numpy as np

def analyze_time():
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
            
    times = []
    for point in points:
        t = point.payload.get('vlm_generation_time_ms', 0)
        if t > 0:
            times.append(t)
        
    if not times:
        print("No generation times found.")
        return

    print(f"Total images with time data: {len(times)}")
    print(f"Min time: {min(times):.2f} ms")
    print(f"Max time: {max(times):.2f} ms")
    print(f"Average time: {sum(times) / len(times):.2f} ms")
    print(f"Median time: {np.median(times):.2f} ms")

if __name__ == "__main__":
    analyze_time()
