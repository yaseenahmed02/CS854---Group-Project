import sys
import os
import time
import uuid
import json
import base64
import requests
import tiktoken
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.getcwd())

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from openai import OpenAI

from utils.file_loader import FileLoader
from utils.ingestion_utils import sanitize_path_component, SemanticChunker
from utils.clone_repo import clone_repo
from embeddings.embed import EmbeddingGenerator

load_dotenv()

class IngestionEngine:
    def __init__(self, mock_vlm: bool = False, vlm_model: str = "gpt-4o-2024-08-06"):
        self.mock_vlm = mock_vlm
        self.vlm_model = vlm_model
        
        # Initialize OpenAI client if not mocking
        self.openai_client = None
        if not self.mock_vlm:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                print("Warning: OPENAI_API_KEY not found. Switching to mock VLM.")
                self.mock_vlm = True

        # Initialize Embedding Generators (Lazy load or load once?)
        # Loading them once here to avoid reloading for every repo
        print("Initializing Embedding Models (Dense, SPLADE, BGE)...")
        self.dense_gen = EmbeddingGenerator(model_type='dense', model_name='jinaai/jina-embeddings-v2-base-code')
        self.splade_gen = EmbeddingGenerator(model_type='sparse_splade', model_name='prithivida/Splade_PP_en_v1')
        self.bge_gen = EmbeddingGenerator(model_type='sparse_bgem3', model_name='BAAI/bge-m3')
        self.bge_gen = EmbeddingGenerator(model_type='sparse_bgem3', model_name='BAAI/bge-m3')
        self.chunker = SemanticChunker()
        
        # Initialize Tokenizer for metrics
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.vlm_model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _get_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _setup_repo_collection(self, repo: str, version: str) -> tuple[QdrantClient, str]:
        safe_repo = sanitize_path_component(repo)
        safe_version = sanitize_path_component(version)
        db_path = f"data/qdrant/qdrant_data_{safe_repo}_{safe_version}"
        collection_name = f"{safe_repo}_{safe_version}"
        
        client = QdrantClient(path=db_path)
        
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense_jina": VectorParams(size=768, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "splade": SparseVectorParams(),
                    "bge": SparseVectorParams()
                }
            )
        return client, collection_name, db_path

    def _setup_global_collection(self, name: str) -> tuple[QdrantClient, str]:
        db_path = f"data/qdrant/qdrant_data_{name}"
        client = QdrantClient(path=db_path)
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config={"dense_jina": VectorParams(size=768, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "splade": SparseVectorParams(),
                    "bge": SparseVectorParams()
                }
            )
        return client, name

    def _generate_all_embeddings(self, text: str) -> tuple[dict, dict]:
        """Generate Dense, SPLADE, and BGE embeddings for a text."""
        start = time.time()
        dense = self.dense_gen.embed_query(text)
        dense_time = (time.time() - start) * 1000

        start = time.time()
        splade = self.splade_gen.embed_query(text)
        splade_time = (time.time() - start) * 1000

        start = time.time()
        bge = self.bge_gen.embed_query(text)
        bge_time = (time.time() - start) * 1000
        
        # Helper to process sparse vectors
        def process_sparse(emb, tokenizer):
            indices = []
            values = []
            if isinstance(emb, dict):
                for token, weight in emb.items():
                    try:
                        # Use tokenizer to get ID if possible, or hash?
                        # Re-using logic from ingest_code_to_qdrant:
                        # We need integer indices. 
                        # Assuming we can access tokenizer.
                        if hasattr(tokenizer, 'convert_tokens_to_ids'):
                            idx = tokenizer.convert_tokens_to_ids(token)
                            indices.append(idx)
                            values.append(weight)
                    except:
                        pass
            
            # Deduplicate (keep max)
            idx_to_val = {}
            for i, v in zip(indices, values):
                idx_to_val[i] = max(idx_to_val.get(i, -float('inf')), v)
            return list(idx_to_val.keys()), list(idx_to_val.values())

        splade_indices, splade_values = process_sparse(splade, self.splade_gen.tokenizer)
        
        # BGE might be tricky if bge_gen.model is the wrapper. 
        # Accessing underlying tokenizer:
        bge_tokenizer = getattr(self.bge_gen.model, 'tokenizer', None)
        bge_indices, bge_values = process_sparse(bge, bge_tokenizer)

        vectors = {
            "dense_jina": dense.tolist(),
            "splade": models.SparseVector(indices=splade_indices, values=splade_values),
            "bge": models.SparseVector(indices=bge_indices, values=bge_values)
        }
        
        timings = {
            "embedding_time_ms_jina": dense_time,
            "embedding_time_ms_splade": splade_time,
            "embedding_time_ms_bge": bge_time
        }
        
        return vectors, timings

    def ingest_codebase(self, repo: str, version: str):
        print(f"--- Ingesting Codebase: {repo} v{version} ---")
        
        # 1. Clone
        try:
            repo_dir = clone_repo(repo, version)
        except Exception as e:
            print(f"Failed to clone {repo} v{version}: {e}")
            return

        # 2. Setup Qdrant
        client, collection_name, db_path = self._setup_repo_collection(repo, version)
        
        # 3. Load & Chunk
        loader = FileLoader(repo_dir)
        documents = loader.load_repo(repo_dir)
        
        points = []
        bm25_chunks = []
        
        print(f"Processing {len(documents)} files...")
        for doc in documents:
            file_path = doc['path']
            content = doc['text']
            
            rel_path = doc.get('metadata', {}).get('rel_path', Path(file_path).name)
            
            # Augment content with RELATIVE path
            augmented_content = f"File Path: {rel_path}\n\n{content}"
            
            # Chunking (using augmented content)
            chunks = self.chunker.chunk_file(augmented_content, file_path)
            
            for i, chunk_text in enumerate(chunks):
                # BM25 Data
                bm25_chunks.append({
                    "text": chunk_text,
                    "filepath": str(file_path),
                    "rel_path": str(rel_path),
                    "chunk_index": i
                })
                
                # Embeddings
                vectors, _ = self._generate_all_embeddings(chunk_text)
                
                # Point
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors,
                    payload={
                        "text": chunk_text,
                        "filepath": str(file_path),
                        "rel_path": str(rel_path),
                        "repo": repo,
                        "version": version,
                        "chunk_index": i
                    }
                )
                points.append(point)
                
                if len(points) >= 50:
                    client.upsert(collection_name=collection_name, points=points)
                    points = []
        
        if points:
            client.upsert(collection_name=collection_name, points=points)
            
        # Save BM25 chunks
        chunks_dir = Path(db_path)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        with open(chunks_dir / "chunks.json", "w") as f:
            json.dump(bm25_chunks, f)
        print(f"Saved {len(bm25_chunks)} chunks to {chunks_dir / 'chunks.json'}")

    def ingest_issues(self, instances: List[Dict]):
        print(f"--- Ingesting {len(instances)} Issues ---")
        client, collection_name = self._setup_global_collection("swe_bench_issues")
        
        points = []
        for instance in instances:
            instance_id = instance['instance_id']
            problem_statement = instance['problem_statement']
            
            start_time = time.time()
            vectors, timings = self._generate_all_embeddings(problem_statement)
            ingestion_time_ms = (time.time() - start_time) * 1000
            
            issue_tokens = self._get_token_count(problem_statement)
            
            payload = {
                "instance_id": instance_id,
                "repo": instance['repo'],
                "version": instance['version'],
                "problem_statement": problem_statement,
                "embedding_time_ms": ingestion_time_ms,
                "issue_tokens": issue_tokens
            }
            payload.update(timings)
            
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload=payload
            )
            points.append(point)
            
        if points:
            client.upsert(collection_name=collection_name, points=points)
            print(f"Upserted {len(points)} issues to {collection_name}")

    def ingest_visuals(self, instances: List[Dict], max_images: Optional[int] = None):
        print(f"--- Ingesting Visual Assets for {len(instances)} Instances ---")
        client, collection_name = self._setup_global_collection("swe_bench_images")
        
        points = []
        for instance in instances:
            instance_id = instance['instance_id']
            images = instance.get('image_assets', [])
            
            # Normalize images list
            image_urls = []
            # print(f"DEBUG: Processing instance {instance_id}, image_assets type: {type(images)}")
            if isinstance(images, list):
                image_urls = [img if isinstance(img, str) else img.get('url') for img in images]
            elif isinstance(images, dict):
                 for key, urls in images.items():
                    if isinstance(urls, list):
                        image_urls.extend(urls)
            
            # print(f"DEBUG: Found {len(image_urls)} images: {image_urls}")
            
            if max_images is not None:
                image_urls = image_urls[:max_images]
            
            for url in image_urls:
                if not url: continue
                
                print(f"Processing image for {instance_id}: {url}")
                
                start_time = time.time()
                
                # Download & Base64
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code != 200:
                        print(f"Failed to download {url}")
                        continue
                    image_base64 = base64.b64encode(resp.content).decode('utf-8')
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                    continue
                
                # VLM Description
                if self.mock_vlm:
                    vlm_desc = f"Mock description for {url}"
                else:
                    # Reuse existing VLM logic or implement here
                    # For brevity, implementing inline or calling helper
                    # I'll implement a simple inline call using self.openai_client
                     vlm_desc = self._generate_vlm_desc(url, instance['problem_statement'])
                
                vlm_time_ms = (time.time() - start_time) * 1000
                vlm_tokens = self._get_token_count(vlm_desc)

                # Embeddings Description (All 3 types)
                vectors, _ = self._generate_all_embeddings(vlm_desc)
                
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors,
                    payload={
                        "instance_id": instance_id,
                        "image_url": url,
                        "vlm_description": vlm_desc,
                        "image_base64": image_base64,
                        "repo": instance['repo'],
                        "version": instance['version'],
                        "vlm_generation_time_ms": vlm_time_ms,
                        "vlm_tokens": vlm_tokens
                    }
                )
                points.append(point)
                
        if points:
            client.upsert(collection_name=collection_name, points=points)
            print(f"Upserted {len(points)} visual assets to {collection_name}")

    def _generate_vlm_desc(self, image_url: str, issue_text: str) -> str:
        system_prompt = "You are a Senior Front-End Engineer... technically reverse-engineer it into a search query."
        user_prompt = f"Issue Context: {issue_text[:500]}...\n\nAnalyze the screenshot."
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                    ]}
                ],
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"VLM Error: {e}")
            return "Error generating description."
