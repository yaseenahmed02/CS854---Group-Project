from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_image_payload():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    try:
        # Get one point
        points = client.scroll(
            collection_name="swe_images",
            limit=10,
            with_payload=True,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="version",
                        match=models.MatchValue(value="1.2")
                    )
                ]
            )
        )[0]
        
        if points:
            for p in points:
                print(json.dumps(p.payload, indent=2))
        else:
            print("No points found in swe_images.")
            
    except Exception as e:
        print(f"Error inspecting payload: {e}")

if __name__ == "__main__":
    inspect_image_payload()
