import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.ingestion_utils import SemanticChunker
from embeddings.embed import EmbeddingGenerator

def verify_chunking_and_embedding():
    print("=== 1. Verifying Semantic Chunking ===")
    
    # Sample code with class and methods
    sample_code = """
import os

class ResourceManager:
    def __init__(self, path):
        self.path = path
        self.resources = []

    def load_resources(self):
        # Load resources from path
        if os.path.exists(self.path):
            self.resources = os.listdir(self.path)
            print(f"Loaded {len(self.resources)} resources")
        else:
            print("Path not found")

    def process_resource(self, resource_name):
        # Process a single resource
        print(f"Processing {resource_name}")
        return True

def standalone_function():
    print("I am a standalone function")
"""
    
    chunker = SemanticChunker()
    chunks = chunker.chunk_file(sample_code, "sample.py")
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.strip())
        print("----------------")

    if not chunks:
        print("Error: No chunks generated.")
        return

    print("\n=== 2. Verifying Embeddings ===")
    test_chunk = chunks[0]
    print(f"Testing embeddings on Chunk 1 (Length: {len(test_chunk)} chars)")

    # 2.1 Dense Embedding
    print("\n[Dense] Initializing Jina model...")
    try:
        dense_gen = EmbeddingGenerator(model_type='dense', model_name='jinaai/jina-embeddings-v2-base-code')
        dense_emb = dense_gen.embed_query(test_chunk)
        print(f"Dense Embedding Shape: {dense_emb.shape}")
        print(f"First 5 values: {dense_emb[:5]}")
    except Exception as e:
        print(f"Dense Embedding Failed: {e}")

    # 2.2 Sparse SPLADE
    print("\n[Sparse] Initializing SPLADE model...")
    try:
        splade_gen = EmbeddingGenerator(model_type='sparse_splade', model_name='prithivida/Splade_PP_en_v1')
        splade_emb = splade_gen.embed_query(test_chunk)
        print(f"SPLADE Embedding Type: {type(splade_emb)}")
        print(f"Number of non-zero tokens: {len(splade_emb)}")
        print(f"Sample tokens: {list(splade_emb.items())[:5]}")
    except Exception as e:
        print(f"SPLADE Embedding Failed: {e}")

    # 2.3 Sparse BGE-M3
    print("\n[Sparse] Initializing BGE-M3 model...")
    try:
        bge_gen = EmbeddingGenerator(model_type='sparse_bgem3', model_name='BAAI/bge-m3')
        bge_emb = bge_gen.embed_query(test_chunk)
        print(f"BGE-M3 Embedding Type: {type(bge_emb)}")
        if isinstance(bge_emb, dict):
            print(f"Number of non-zero tokens: {len(bge_emb)}")
            print(f"Sample tokens: {list(bge_emb.items())[:5]}")
        else:
            print(f"BGE Output: {bge_emb}")
    except Exception as e:
        print(f"BGE-M3 Embedding Failed: {e}")

if __name__ == "__main__":
    verify_chunking_and_embedding()
