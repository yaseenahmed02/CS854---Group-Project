import sys
import os
import base64
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qdrant_client import QdrantClient
from ingest_images_to_qdrant import ingest_images

def verify_image_storage():
    print("1. Clearing existing collection to ensure clean state...")
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    client.delete_collection("swe_images")
    client.close()
    
    print("2. Running ingestion for 1 instance (limit=1)...")
    # We use a known repo/version or just limit=1 to get *something*
    # We'll use mock=True to avoid OpenAI costs, but we need real image download.
    # ingest_images uses real download even if mock_vlm is True.
    
    # Ensure we use a split/repo that has images. markedjs/marked usually has them.
    ingest_images(limit=1, mock_vlm=True, split="dev", repo_filter="markedjs/marked", version_filter="1.2")
    
    print("\n2. Connecting to Qdrant to verify storage...")
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    
    # Scroll to get points
    points, _ = client.scroll(
        collection_name="swe_images",
        limit=1,
        with_payload=True,
        with_vectors=False
    )
    
    if not points:
        print("FAIL: No points found in 'swe_images' collection.")
        sys.exit(1)
        
    point = points[0]
    payload = point.payload
    
    print(f"Retrieved point ID: {point.id}")
    print(f"Instance ID: {payload.get('instance_id')}")
    print(f"Image URL: {payload.get('image_url')}")
    
    image_base64 = payload.get('image_base64')
    
    if not image_base64:
        print("FAIL: 'image_base64' field is missing or empty in payload.")
        sys.exit(1)
        
    print(f"SUCCESS: 'image_base64' found (Length: {len(image_base64)} chars).")
    
    # Decode and save
    try:
        image_data = base64.b64decode(image_base64)
        output_path = Path("verification/retrieved_image.png")
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"SUCCESS: Image decoded and saved to {output_path.absolute()}")
    except Exception as e:
        print(f"FAIL: Could not decode/save image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_image_storage()
