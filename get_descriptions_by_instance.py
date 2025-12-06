
from qdrant_client import QdrantClient
from qdrant_client.http import models

def get_descriptions_by_instance():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    target_instances = [
        "chartjs__Chart.js-9764",
        "chartjs__Chart.js-9399",
        "chartjs__Chart.js-8650"
    ]
    
    print(f"Fetching images for instances: {target_instances}\n")
    
    for instance_id in target_instances:
        print(f"=== Instance: {instance_id} ===")
        
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
            limit=100, # Should cover all images for an instance
            with_payload=True,
            with_vectors=False
        )
        points, _ = res
        
        if not points:
            print("No images found for this instance.")
            continue
            
        for point in points:
            url = point.payload.get('image_url')
            desc = point.payload.get('vlm_description')
            print(f"Image: {url}")
            print(f"Description:\n{desc}\n")
            print("-" * 40)

if __name__ == "__main__":
    get_descriptions_by_instance()
