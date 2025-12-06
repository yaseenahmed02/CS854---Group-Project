import sys
import os
import time
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.getcwd())

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

from utils.file_loader import FileLoader
from utils.ingestion_utils import sanitize_path_component, SemanticChunker
from embeddings.embed import EmbeddingGenerator

def setup_qdrant(repo_name: str, version: str, mode: str = "create") -> tuple[QdrantClient, str, bool]:
    """
    Setup Qdrant client and collection.
    
    Args:
        repo_name: Name of the repository
        version: Version string
        mode: Ingestion mode ('create', 'append', 'skip', 'overwrite')
            - 'create': Create new collection
            - 'append': Append to existing collection
            - 'skip': Skip if collection exists
            - 'overwrite': Overwrite existing collection
        
    Returns:
        Tuple of (client, collection_name, created/ready)
    """
    # Create version-isolated path
    safe_repo = sanitize_path_component(repo_name)
    safe_version = sanitize_path_component(version)
    db_path = f"data/qdrant/qdrant_data_{safe_repo}_{safe_version}"
    
    print(f"Initializing Qdrant at {db_path}...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Absolute DB path: {os.path.abspath(db_path)}")
    client = QdrantClient(path=db_path)
    
    collection_name = f"{safe_repo}_{safe_version}"
    
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if exists:
        if mode == "skip":
            print(f"Collection {collection_name} exists. Mode is 'skip'. Skipping.")
            return client, collection_name, False
        elif mode == "overwrite":
            print(f"Collection {collection_name} exists. Mode is 'overwrite'. Deleting...")
            client.delete_collection(collection_name)
            exists = False # Proceed to create
        elif mode == "append":
            print(f"Collection {collection_name} exists. Mode is 'append'. Appending data.")
            return client, collection_name, True
        else: # default create/unknown
            print(f"Collection {collection_name} exists. Mode is '{mode}'. Skipping to prevent accidental duplicate.")
            return client, collection_name, False

    if not exists:
        print(f"Creating collection {collection_name}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense_jina": VectorParams(size=768, distance=Distance.COSINE) # Jina-v2-base-code is 768
            },
            sparse_vectors_config={
                "splade": SparseVectorParams(),
                "bge": SparseVectorParams()
            }
        )
        
    return client, collection_name, True

def get_dir_size(path: str) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except Exception as e:
        print(f"Error calculating size for {path}: {e}")
    return total

def ingest_repo(repo_path: str, repo_name: str, version: str, mode: str = "create") -> Dict[str, Any]:
    """
    Ingest a repository into Qdrant.
    Returns metrics dictionary.
    """
    start_time = time.time()
    repo_size = get_dir_size(repo_path)
    
    # 1. Setup Qdrant
    client, collection_name, ready = setup_qdrant(repo_name, version, mode)
    
    metrics = {
        "repo": repo_name,
        "version": version,
        "repo_size_bytes": repo_size,
        "embedding_time_ms": 0,
        "chunking_time_ms": 0,
        "dense_jina_time_ms": 0,
        "splade_time_ms": 0,
        "bge_time_ms": 0,
        "vector_db_size_points": 0,
        "num_files": 0,
        "total_chunks": 0
    }

    if not ready:
        # Fetch existing stats if possible
        try:
            info = client.get_collection(collection_name)
            metrics["vector_db_size_points"] = info.points_count
        except:
            pass
        return metrics
    
    # 2. Load Documents
    print(f"Loading documents from {repo_path}...")
    loader = FileLoader(repo_path)
    documents = loader.load_repo(repo_path)
    
    metrics["num_files"] = len(documents)
    
    if not documents:
        print("No documents found. Exiting.")
        return metrics

    # 3. Initialize Models
    # We need 3 generators. 
    # Note: Loading 3 models might be heavy on memory.
    print("Initializing embedding models...")
    dense_gen = EmbeddingGenerator(model_type='dense', model_name='jinaai/jina-embeddings-v2-base-code')
    
    # For sparse, we use the ones defined in embed.py
    # Note: embed.py implementation for BGE might be a placeholder if libraries are missing, but we proceed.
    splade_gen = EmbeddingGenerator(model_type='sparse_splade', model_name='prithivida/Splade_PP_en_v1')
    bge_gen = EmbeddingGenerator(model_type='sparse_bgem3', model_name='BAAI/bge-m3')
    
    chunker = SemanticChunker()
    
    points = []
    total_chunks = 0
    
    # Timers
    total_chunking_time = 0
    total_dense_time = 0
    total_splade_time = 0
    total_bge_time = 0
    
    print("Processing documents...")
    for doc in documents:
        file_content = doc['text']
        file_path = doc['path']
        
        # Chunking
        t0 = time.time()
        chunks = chunker.chunk_file(file_content, file_path)
        total_chunking_time += (time.time() - t0)
        
        for i, chunk_text in enumerate(chunks):
            total_chunks += 1
            
            # Generate Embeddings
            # Dense
            t0 = time.time()
            dense_emb = dense_gen.embed_query(chunk_text)
            total_dense_time += (time.time() - t0)
            
            # Sparse
            t0 = time.time()
            splade_emb = splade_gen.embed_query(chunk_text)
            total_splade_time += (time.time() - t0)
            
            t0 = time.time()
            bge_emb = bge_gen.embed_query(chunk_text)
            total_bge_time += (time.time() - t0)
            
            # Prepare Point
            point_id = str(uuid.uuid4())
            
            # Convert sparse dicts to Qdrant format (indices, values)
            def to_sparse_vector(sparse_dict):
                # Qdrant expects integer indices. 
                # BUT our embed.py returns token strings -> weights.
                # Qdrant sparse vectors require integer indices (usually hash of token or vocabulary index).
                # Wait, Qdrant sparse vectors usually work with integer indices.
                # If our models return token strings, we need to map them to integers or use a hashing trick.
                # However, standard SPLADE/BGE usually work with vocabulary indices.
                # Let's check embed.py again.
                # embed.py _generate_splade returns {token: weight}.
                # We need the tokenizer's vocab ID for Qdrant if we want to use standard sparse vectors.
                pass

            # Correction: I need to get indices.
            # Since `embed.py` returns strings, I will re-encode them to get IDs? No that's wasteful.
            # Let's rely on the fact that I can access `splade_gen.tokenizer` if I need to.
            # But `embed.py` returns a dict `token_str -> weight`.
            # I will use `splade_gen.tokenizer.convert_tokens_to_ids(token_str)` to get the ID.
            
            # Helper to deduplicate and keep max weight
            def deduplicate_sparse(indices, values):
                if not indices:
                    return [], []
                
                # Use dict to keep max value for each index
                idx_to_val = {}
                for idx, val in zip(indices, values):
                    if idx in idx_to_val:
                        idx_to_val[idx] = max(idx_to_val[idx], val)
                    else:
                        idx_to_val[idx] = val
                
                return list(idx_to_val.keys()), list(idx_to_val.values())

            # SPLADE
            splade_indices = []
            splade_values = []
            for token, weight in splade_emb.items():
                try:
                    idx = splade_gen.tokenizer.convert_tokens_to_ids(token)
                    splade_indices.append(idx)
                    splade_values.append(weight)
                except:
                    pass
            splade_indices, splade_values = deduplicate_sparse(splade_indices, splade_values)
            
            # BGE
            bge_indices = []
            bge_values = []
            if isinstance(bge_emb, dict) and "error" not in bge_emb:
                for token, weight in bge_emb.items():
                    try:
                        # BGEM3FlagModel has tokenizer at self.model.tokenizer
                        # But here bge_gen.model is the BGEM3FlagModel instance
                        idx = bge_gen.model.tokenizer.convert_tokens_to_ids(token)
                        bge_indices.append(idx)
                        bge_values.append(weight)
                    except:
                        pass
            bge_indices, bge_values = deduplicate_sparse(bge_indices, bge_values)
            
            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense_jina": dense_emb.tolist(),
                    "splade": models.SparseVector(indices=splade_indices, values=splade_values),
                    "bge": models.SparseVector(indices=bge_indices, values=bge_values)
                },
                payload={
                    "text": chunk_text,
                    "filepath": str(file_path),
                    "repo": repo_name,
                    "version": version,
                    "chunk_index": i
                }
            )
            points.append(point)
            
            # Upsert in batches of 50
            if len(points) >= 50:
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"Upserted batch of {len(points)} chunks.")
                points = []

    # Upsert remaining
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Upserted final batch of {len(points)} chunks.")

    print(f"Ingestion complete. Total chunks: {total_chunks}")
    
    end_time = time.time()
    metrics["embedding_time_ms"] = round((end_time - start_time) * 1000, 2)
    metrics["chunking_time_ms"] = round(total_chunking_time * 1000, 2)
    metrics["dense_jina_time_ms"] = round(total_dense_time * 1000, 2)
    metrics["splade_time_ms"] = round(total_splade_time * 1000, 2)
    metrics["bge_time_ms"] = round(total_bge_time * 1000, 2)
    metrics["total_chunks"] = total_chunks
    
    try:
        info = client.get_collection(collection_name)
        metrics["vector_db_size_points"] = info.points_count
    except Exception as e:
        print(f"Error fetching collection info: {e}")
        
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument("repo_name", help="Name of the repository (e.g., owner/repo)")
    parser.add_argument("version", nargs="?", default="1.0.0", help="Version string")
    parser.add_argument("--mode", choices=['create', 'append', 'skip', 'overwrite'], default="create", help="Ingestion mode")
    
    args = parser.parse_args()
    
    metrics = ingest_repo(args.repo_path, args.repo_name, args.version, args.mode)
    
    # Save metrics
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    metrics_file = results_dir / "ingestion_metrics.json"
    
    existing_metrics = []
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                existing_metrics = json.load(f)
        except:
            pass
            
    # Add timestamp
    metrics['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
    existing_metrics.append(metrics)
    
    with open(metrics_file, 'w') as f:
        json.dump(existing_metrics, f, indent=2)
    print(f"Saved ingestion metrics to {metrics_file}")
