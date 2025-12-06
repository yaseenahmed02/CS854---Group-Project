
from qdrant_client import QdrantClient
from qdrant_client.http import models

def get_specific_descriptions():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    target_instances = [
        "markedjs__marked-1683",
        "processing__p5.js-5917"
    ]
    
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
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        points, _ = res
        for point in points:
            print(f"Image: {point.payload.get('image_url')}")
            print(f"Description:\n{point.payload.get('vlm_description')}\n")
            print("-" * 40)

if __name__ == "__main__":
    get_specific_descriptions()
