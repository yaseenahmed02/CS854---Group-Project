import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from qdrant_client import QdrantClient
import tiktoken
import numpy as np

def inspect_sizes(repo="markedjs/marked", version="1.2"):
    repo_san = repo.replace("/", "_")
    version_san = version.replace(".", "_")
    collection_name = f"{repo_san}_{version_san}"
    db_path = f"data/qdrant/qdrant_data_{collection_name}"
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    print(f"Connecting to {db_path}...")
    client = QdrantClient(path=db_path)
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    token_counts = []
    next_offset = None
    
    print("Scanning documents...")
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False
        )
        
        for point in points:
            payload = point.payload
            text = payload.get('text', '')
            if not text:
                # Try nested payload if structure is different
                # Based on previous debugging, sometimes payload is nested or keys differ
                # But usually it's at top level for code ingestion
                pass
            
            tokens = len(tokenizer.encode(text))
            token_counts.append(tokens)
        
        if next_offset is None:
            break
            
    if not token_counts:
        print("No documents found.")
        return

    avg_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    max_tokens = np.max(token_counts)
    min_tokens = np.min(token_counts)
    total_docs = len(token_counts)
    
    print(f"\nStats for {collection_name}:")
    print(f"Total Documents: {total_docs}")
    print(f"Average Tokens: {avg_tokens:.2f}")
    print(f"Median Tokens: {median_tokens:.2f}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Min Tokens: {min_tokens}")

if __name__ == "__main__":
    inspect_sizes()
