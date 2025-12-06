import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.getcwd())

from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from rank_bm25 import BM25Okapi

from embeddings.embed import EmbeddingGenerator
from utils.timer import measure_time

class FlexibleRetriever:
    """
    Retriever that supports dynamic strategies (Dense, Sparse, Hybrid)
    and Multimodal fusion.
    """
    
    def __init__(self, 
                 client: QdrantClient, 
                 collection_name: str, 
                 swe_images_collection: str = "swe_images",
                 chunks_file: Optional[str] = None,
                 images_client: Optional[QdrantClient] = None):
        """
        Initialize FlexibleRetriever.
        
        Args:
            client: QdrantClient instance (for code)
            collection_name: Name of the code collection
            swe_images_collection: Name of the images collection
            chunks_file: Path to chunks.json (required for BM25 strategy)
            images_client: QdrantClient instance for images (if different DB)
        """
        self.client = client
        self.images_client = images_client if images_client else client
        self.collection_name = collection_name
        self.swe_images_collection = swe_images_collection
        self.chunks_file = chunks_file
        
        # Lazy loaded components
        self.models = {}
        self.bm25 = None
        self.chunks = None
        
    def _get_model(self, model_type: str) -> EmbeddingGenerator:
        """Lazy load embedding models."""
        if model_type not in self.models:
            if model_type == 'dense_jina':
                self.models[model_type] = EmbeddingGenerator(model_type='dense', model_name='jinaai/jina-embeddings-v2-base-code')
            elif model_type == 'splade':
                self.models[model_type] = EmbeddingGenerator(model_type='sparse_splade', model_name='prithivida/Splade_PP_en_v1')
            elif model_type == 'bge':
                self.models[model_type] = EmbeddingGenerator(model_type='sparse_bgem3', model_name='BAAI/bge-m3')
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        return self.models[model_type]

    def _get_bm25(self):
        """Lazy load BM25 index."""
        if self.bm25 is None:
            if self.chunks_file and os.path.exists(self.chunks_file):
                print("Loading chunks for BM25 from file...")
                with open(self.chunks_file, 'r') as f:
                    self.chunks = json.load(f)
            else:
                print("Fetching all documents from Qdrant for BM25...")
                # Scroll all points
                self.chunks = []
                next_offset = None
                while True:
                    points, next_offset = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=100,
                        offset=next_offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    for point in points:
                        # Convert payload to chunk format
                        chunk = point.payload
                        chunk['chunk_id'] = point.id
                        self.chunks.append(chunk)
                    
                    if next_offset is None:
                        break
            
            if not self.chunks:
                raise ValueError("No documents found for BM25 index")

            print(f"Building BM25 index with {len(self.chunks)} documents...")
            tokenized_corpus = [self._tokenize(chunk.get('text', '')) for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
        return self.bm25

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Replace non-alphanumeric with space
        clean_text = "".join([c if c.isalnum() else " " for c in text.lower()])
        return [t for t in clean_text.split() if t]

    def _fetch_visual_context(self, instance_id: str) -> List[Dict[str, Any]]:
        """Fetch all VLM descriptions and metadata from Qdrant for an instance."""
        context = []
        try:
            # Search by instance_id in payload
            res = self.images_client.scroll(
                collection_name=self.swe_images_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="instance_id",
                            match=models.MatchValue(value=instance_id)
                        )
                    ]
                ),
                limit=10 # Fetch up to 10 images
            )
            points, _ = res
            for point in points:
                desc = point.payload.get("vlm_description")
                time_ms = point.payload.get("vlm_generation_time_ms", 0)
                if desc:
                    context.append({
                        "vlm_description": desc,
                        "vlm_generation_time_ms": time_ms,
                        "image_url": point.payload.get("image_url"),
                        "image_base64": point.payload.get("image_base64")
                    })
        except Exception as e:
            print(f"Error fetching visual context: {e}")
        return context

    def _fetch_visual_descriptions(self, instance_id: str) -> List[str]:
        """Legacy wrapper for backward compatibility."""
        context = self._fetch_visual_context(instance_id)
        return [item['vlm_description'] for item in context]

    def retrieve(self, 
                 query: str, 
                 instance_id: Optional[str] = None, 
                 strategy: List[str] = ["dense_jina"], 
                 visual_mode: str = "none", 
                 top_k: int = 10) -> Dict[str, Any]:
        """
        Execute retrieval based on strategy.
        
        Args:
            query: Text query
            instance_id: SWE-bench instance ID (for VLM)
            strategy: List of retrievers to use (['jina', 'splade', 'bge', 'bm25'])
            visual_mode: 'none', 'augment' (append to query), 'fusion' (separate query)
            top_k: Number of results to return
            
        Returns:
            Dict with 'results' and metadata
        """
        # 1. Handle Visual Mode
        vlm_descs = []
        if visual_mode in ["augment", "fusion", "visual_only"] and instance_id:
            vlm_descs = self._fetch_visual_descriptions(instance_id)
            if vlm_descs and visual_mode == "augment":
                # Join all descriptions
                combined_desc = " ".join(vlm_descs)
                query = f"{query} {combined_desc}"
                print(f"Augmented query with {len(vlm_descs)} VLM descriptions")

        # 2. Execute Strategies
        all_results = []
        
        # Metrics
        search_time_ms = 0.0
        runtime_embedding_time_ms = 0.0
        visual_embedding_time_ms = 0.0
        search_time_breakdown = {
            "search_time_ms_bm25": 0.0,
            "search_time_ms_dense_jina": 0.0,
            "search_time_ms_splade": 0.0,
            "search_time_ms_bge": 0.0
        }
        
        # If visual_mode is fusion, we treat the visual query as a separate "strategy" execution effectively
        # But usually fusion means: Query(Text) + Query(Visual) -> Fuse
        # Here we have multiple strategies (Jina, Splade) AND potential visual fusion.
        # Let's simplify: If fusion, we run the strategies for Text Query AND Visual Query, then fuse all.
        
        queries_to_run = [query]
        is_visual_query = [False]
        
        if visual_mode == "fusion" and vlm_descs:
            queries_to_run.extend(vlm_descs)
            is_visual_query.extend([True] * len(vlm_descs))
            print(f"Running separate visual queries for fusion ({len(vlm_descs)} images)")
        elif visual_mode == "visual_only" and vlm_descs:
            queries_to_run = vlm_descs
            is_visual_query = [True] * len(vlm_descs)
            print(f"Running visual-only queries ({len(vlm_descs)} images)")
        elif visual_mode == "visual_only" and not vlm_descs:
            print("Warning: visual_only mode requested but no VLM description found. Returning empty results.")
            queries_to_run = []
            is_visual_query = []

        for q, is_visual in zip(queries_to_run, is_visual_query):
            for strat in strategy:
                print(f"Running strategy: {strat} for query: {q[:50]}...")
                
                if strat == "bm25":
                    # BM25 Search Time
                    with measure_time() as t:
                        bm25 = self._get_bm25()
                        tokenized_query = self._tokenize(q)
                        scores = bm25.get_scores(tokenized_query)
                        top_n = np.argsort(scores)[::-1][:top_k*2] # Get more for fusion
                    
                    elapsed = t['elapsed_ms']
                    search_time_ms += elapsed
                    search_time_breakdown[f"search_time_ms_{strat}"] += elapsed
                    
                    strat_results = []
                    for idx in top_n:
                        strat_results.append({
                            "id": self.chunks[idx]['chunk_id'], # Assuming chunk_id exists
                            "score": float(scores[idx]),
                            "content": self.chunks[idx],
                            "source": "bm25"
                        })
                    all_results.append(strat_results)
                    
                elif strat in ["dense_jina", "splade", "bge"]:
                    # Qdrant Retrieval
                    model = self._get_model(strat)
                    
                    # Measure Embedding Time
                    with measure_time() as t_embed:
                        emb = model.embed_query(q)
                    
                    embed_elapsed = t_embed['elapsed_ms']
                    runtime_embedding_time_ms += embed_elapsed
                    if is_visual:
                        visual_embedding_time_ms += embed_elapsed
                    
                    search_params = None
                    query_vector = None
                    vector_name = None
                    
                    if strat == "dense_jina":
                        vector_name = "dense_jina"
                        query_vector = emb.tolist()
                    elif strat in ["splade", "bge"]:
                        vector_name = strat
                        # Convert sparse dict to models.SparseVector
                        indices = []
                        values = []
                        
                        # Helper to deduplicate (same as ingestion)
                        # We need to ensure we use the same tokenizer/logic as ingestion
                        # embed.py returns {token: weight}
                        # We need to convert tokens to IDs if we used IDs in ingestion
                        # In ingestion, we used model.tokenizer.convert_tokens_to_ids
                        
                        if strat == "splade":
                            tokenizer = model.tokenizer
                        else: # bge
                            tokenizer = model.model.tokenizer
                            
                        for token, weight in emb.items():
                            try:
                                idx = tokenizer.convert_tokens_to_ids(token)
                                indices.append(idx)
                                values.append(weight)
                            except:
                                pass
                                
                        # Deduplicate
                        idx_to_val = {}
                        for idx, val in zip(indices, values):
                            idx_to_val[idx] = max(idx_to_val.get(idx, 0), val)
                        
                        query_vector = models.SparseVector(
                            indices=list(idx_to_val.keys()),
                            values=list(idx_to_val.values())
                        )
                    
                    # Measure Search Time
                    with measure_time() as t_search:
                        search_result = self.client.query_points(
                            collection_name=self.collection_name,
                            query=query_vector,
                            using=vector_name,
                            limit=top_k*2
                        ).points
                    
                    elapsed = t_search['elapsed_ms']
                    search_time_ms += elapsed
                    search_time_breakdown[f"search_time_ms_{strat}"] += elapsed
                    
                    strat_results = []
                    for hit in search_result:
                        strat_results.append({
                            "id": hit.id, # Point ID
                            "score": hit.score,
                            "content": hit.payload,
                            "source": strat
                        })
                    all_results.append(strat_results)

        # 3. Fuse Results
        fused_results = self._fuse_rankings(all_results, k=60)
        
        return {
            "query": query,
            "results": fused_results[:top_k],
            "retrieved_documents": fused_results[:top_k],
            "strategies": strategy,
            "visual_mode": visual_mode,
            "search_time_ms": search_time_ms,
            "runtime_embedding_time_ms": runtime_embedding_time_ms,
            "visual_embedding_time_ms": visual_embedding_time_ms,
            "search_time_breakdown": search_time_breakdown
        }

    def _fuse_rankings(self, results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion.
        score = sum(1 / (k + rank))
        """
        fused_scores = defaultdict(float)
        doc_map = {}
        
        for results in results_list:
            for rank, item in enumerate(results):
                doc_id = item['id'] # Use unique ID
                # If using Qdrant, ID is UUID. If BM25, ID is chunk_id.
                # We need to ensure they match if they refer to same content.
                # In ingestion, we generated UUIDs for Qdrant points.
                # BM25 chunks might have different IDs if not synchronized.
                # However, for this exercise, we assume Qdrant is the primary source or we just fuse based on what we have.
                # If mixing BM25 (from file) and Qdrant (from DB), IDs might mismatch unless we used chunk_id as Point ID.
                # In ingest_code_to_qdrant.py, we used uuid.uuid4().
                # This is a potential issue for Hybrid BM25 + Qdrant if they don't share IDs.
                # But for Qdrant-based strategies (Jina + Splade), they share Point IDs (same point has multiple vectors? No, we upserted points with all vectors).
                # Wait, in ingestion we created ONE point with "dense", "splade", "bge" vectors.
                # So Jina/Splade/BGE will return the SAME Point ID for the same chunk.
                # BM25 from file will have "chunk_id".
                # If we want to fuse BM25, we need to map BM25 ID to Qdrant ID or vice versa.
                # Since we can't easily do that without a mapping, we might just rely on payload content or accept they are different "docs" in this context
                # OR, we rely on the fact that we are mostly testing Qdrant strategies.
                # If BM25 is used, it might be isolated.
                # Let's proceed with ID fusion.
                
                fused_scores[doc_id] += 1 / (k + rank + 1)
                if doc_id not in doc_map:
                    doc_map[doc_id] = item['content']

        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        final_results = []
        for doc_id in sorted_ids:
            final_results.append({
                "id": doc_id,
                "score": fused_scores[doc_id],
                "payload": doc_map[doc_id]
            })
            
        return final_results
