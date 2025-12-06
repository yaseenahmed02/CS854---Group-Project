
import sys
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

def verify_image_data(instance_id: str):
    db_path = "data/qdrant/qdrant_data_swe_images"
    if not os.path.exists(db_path):
        print(f"Error: Database path {db_path} does not exist.")
        return

    client = QdrantClient(path=db_path)
    
    print("Listing collections...")
    collections = client.get_collections().collections
    for c in collections:
        print(f" - {c.name}")

    collection_name = "swe_images"

    print(f"Searching for instance_id: {instance_id} in {collection_name}...")

    try:
        res = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="instance_id",
                        match=models.MatchValue(value=instance_id)
                    )
                ]
            ),
            limit=10,
            with_payload=True
        )
        points, _ = res
        
        if not points:
            print(f"No points found for instance_id: {instance_id}")
        else:
            print(f"Found {len(points)} points.")
            for point in points:
                payload = point.payload
                print(f"--- Point ID: {point.id} ---")
                print(f"Instance ID: {payload.get('instance_id')}")
                print(f"Image URL: {payload.get('image_url')}")
                print(f"VLM Generation Time: {payload.get('vlm_generation_time_ms')}")
                print(f"VLM Tokens: {payload.get('vlm_tokens')}")
                print(f"Has Description: {bool(payload.get('vlm_description'))}")
                print(f"Description: {payload.get('vlm_description')}")
                
    except Exception as e:
        print(f"Error querying Qdrant: {e}")

if __name__ == "__main__":
    verify_image_data("markedjs__marked-1889")
