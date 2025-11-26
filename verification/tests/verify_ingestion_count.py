
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from pathlib import Path
from qdrant_client import QdrantClient

# Add project root to path
sys.path.insert(0, os.getcwd())

from utils.file_loader import FileLoader
from utils.ingestion_utils import SemanticChunker

def verify_counts():
    repo_path = "data/repos/markedjs__marked__v1_2_0"
    db_path = "data/qdrant/qdrant_data_markedjs_marked_1_2"
    collection_name = "markedjs_marked_1_2"
    
    print(f"Verifying ingestion for {repo_path}...")
    
    # 1. Get Qdrant Count
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} does not exist.")
        return
        
    client = QdrantClient(path=db_path)
    try:
        qdrant_count = client.count(collection_name).count
        print(f"Qdrant Vector Count: {qdrant_count}")
    except Exception as e:
        print(f"Error getting Qdrant count: {e}")
        return

    # 2. Calculate Expected Count
    print("Calculating expected chunks...")
    loader = FileLoader(repo_path)
    documents = loader.load_repo(repo_path)
    
    chunker = SemanticChunker()
    expected_count = 0
    
    for doc in documents:
        chunks = chunker.chunk_file(doc['text'], doc['path'])
        expected_count += len(chunks)
        
    print(f"Expected Chunk Count: {expected_count}")
    
    # 3. Compare
    if qdrant_count == expected_count:
        print("SUCCESS: Counts match.")
    else:
        print(f"FAILURE: Counts do not match. Diff: {qdrant_count - expected_count}")

if __name__ == "__main__":
    verify_counts()
