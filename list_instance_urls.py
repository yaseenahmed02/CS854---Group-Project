
from qdrant_client import QdrantClient
from qdrant_client.http import models

def list_instance_urls():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    target_instances = ["chartjs__Chart.js-9399", "chartjs__Chart.js-8650", "chartjs__Chart.js-9764"]
    
    for instance_id in target_instances:
        print(f"=== {instance_id} ===")
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
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        points, _ = res
        for p in points:
            print(p.payload.get("image_url"))
        print("-" * 20)

if __name__ == "__main__":
    list_instance_urls()
