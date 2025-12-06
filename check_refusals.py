
from qdrant_client import QdrantClient

def check_refusals():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    
    print("Scanning for refusals...")
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
            
    refusal_phrases = [
        "I'm sorry", 
        "I cannot", 
        "I can't help", 
        "I am sorry",
        "unable to analyze",
        "cannot analyze"
    ]
    
    found_refusals = 0
    for point in points:
        desc = point.payload.get('vlm_description', '').lower()
        if any(phrase.lower() in desc for phrase in refusal_phrases):
            found_refusals += 1
            print(f"{point.payload.get('instance_id')} | {point.payload.get('image_url')}")
            
    if found_refusals == 0:
        print("\nNo obvious refusals found.")
    else:
        print(f"\nFound {found_refusals} potential refusals.")

if __name__ == "__main__":
    check_refusals()
