
from qdrant_client import QdrantClient
from qdrant_client.http import models

def verify_manual_ingest():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    target_instance_id = "markedjs__marked-1683"
    
    print(f"Fetching data for {target_instance_id}...")
    
    res = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="instance_id",
                    match=models.MatchValue(value=target_instance_id)
                )
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    points, _ = res
    
    if not points:
        print("No points found for this instance.")
        return
        
    for point in points:
        # We are looking for the one with the demo URL (or the one we updated)
        # Since we updated the point in place, it should be the one.
        url = point.payload.get('image_url', '')
        print(f"\n--- Point ID: {point.id} ---")
        print(f"URL: {url}")
        print(f"VLM Tokens: {point.payload.get('vlm_tokens')}")
        
        base64_data = point.payload.get('image_base64')
        has_base64 = bool(base64_data) and len(base64_data) > 100
        print(f"Has Base64 Image: {has_base64}")
        if has_base64:
            print(f"Base64 Length: {len(base64_data)}")
            
        print(f"Description:\n{point.payload.get('vlm_description')}\n")

if __name__ == "__main__":
    verify_manual_ingest()
