import sys
import os
import base64
from pathlib import Path
from datasets import load_dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qdrant_client import QdrantClient
from ingestion_engine import IngestionEngine

def verify_image_storage():
    print("1. Clearing existing collection to ensure clean state...")
    # New path for global images collection
    db_path = "data/qdrant/qdrant_data_swe_bench_images"
    if os.path.exists(db_path):
        client = QdrantClient(path=db_path)
        if client.collection_exists("swe_bench_images"):
            client.delete_collection("swe_bench_images")
        client.close()
    
    print("2. Running ingestion for 1 instance (limit=1)...")
    # Load a sample instance with images
    print("   Loading dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="dev")
    
    target_instance = None
    for inst in dataset:
        # markedjs/marked usually has images and is small
        if inst['repo'] == "markedjs/marked" and inst.get('image_assets'):
            target_instance = inst
            break
            
    if not target_instance:
        print("   Warning: Could not find markedjs/marked instance with images. Using first available.")
        for inst in dataset:
            if inst.get('image_assets'):
                target_instance = inst
                break
    
    if not target_instance:
        print("FAIL: No instances with images found in dataset.")
        sys.exit(1)

    print(f"   Ingesting instance: {target_instance['instance_id']}")
    
    engine = IngestionEngine(mock_vlm=True)
    engine.ingest_visuals([target_instance])
    
    print("\n3. Connecting to Qdrant to verify storage...")
    client = QdrantClient(path=db_path)
    
    # Scroll to get points
    points, _ = client.scroll(
        collection_name="swe_bench_images",
        limit=1,
        with_payload=True,
        with_vectors=False
    )
    
    if not points:
        print("FAIL: No points found in 'swe_bench_images' collection.")
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
