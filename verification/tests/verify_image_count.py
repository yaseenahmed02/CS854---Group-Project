from qdrant_client import QdrantClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_images():
    db_path = "data/qdrant/qdrant_data_swe_images"
    collection_name = "swe_images"
    
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} does not exist.")
        return

    client = QdrantClient(path=db_path)
    try:
        if client.collection_exists(collection_name):
            count = client.count(collection_name).count
            print(f"Image Collection '{collection_name}' Count: {count}")
        else:
            print(f"Collection '{collection_name}' does not exist.")
    except Exception as e:
        print(f"Error checking image collection: {e}")

if __name__ == "__main__":
    verify_images()
