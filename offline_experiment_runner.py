import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi

# Add project root to path
sys.path.insert(0, os.getcwd())

from qdrant_client import QdrantClient
from qdrant_client.http import models
from rag.pipeline import RAGPipeline
from utils.ingestion_utils import sanitize_path_component

# Define Experiments (Copied from run_experiments.py or imported if possible)
# For now, redefining to ensure self-containment or I can import if I refactor run_experiments
EXPERIMENTS = {
    "text_bm25":   {"strategies": ["bm25"], "visual_mode": "none"},
    "text_bge":    {"strategies": ["bge"], "visual_mode": "none"},
    "text_dense_jina":   {"strategies": ["dense_jina"], "visual_mode": "none"},
    "text_hybrid_jina_bge":    {"strategies": ["dense_jina", "bge"], "visual_mode": "none"},
    "multimodal_fusion_jina":   {"strategies": ["dense_jina"], "visual_mode": "fusion"},
    "multimodal_fusion_hybrid_jina_best_sparse": {"strategies": ["dense_jina", "bge"], "visual_mode": "fusion"},
}

class OfflineRetriever:
    def __init__(self, client: QdrantClient, collection_name: str, issues_client: QdrantClient, images_client: QdrantClient):
        self.client = client
        self.collection_name = collection_name
        self.issues_client = issues_client
        self.images_client = images_client
        self.bm25 = None
        self.chunks = None

    def _get_bm25(self, chunks_file: str):
        if self.bm25 is None:
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r') as f:
                    self.chunks = json.load(f)
                tokenized_corpus = [self._tokenize(chunk.get('text', '')) for chunk in self.chunks]
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                print(f"Warning: chunks.json not found at {chunks_file}")
                return None
        return self.bm25

    def _tokenize(self, text: str) -> List[str]:
        clean_text = "".join([c if c.isalnum() else " " for c in text.lower()])
        return [t for t in clean_text.split() if t]

    def _fetch_issue_vectors(self, instance_id: str) -> Dict[str, Any]:
        """Fetch pre-computed vectors for the issue."""
        res = self.issues_client.scroll(
            collection_name="swe_bench_issues",
            scroll_filter=models.Filter(must=[models.FieldCondition(key="instance_id", match=models.MatchValue(value=instance_id))]),
            limit=1,
            with_vectors=True
        )
        points, _ = res
        if points:
            return points[0].vector
        return {}

    def _fetch_visual_vectors(self, instance_id: str) -> List[Dict[str, Any]]:
        """Fetch pre-computed vectors for all images of the instance."""
        res = self.images_client.scroll(
            collection_name="swe_bench_images",
            scroll_filter=models.Filter(must=[models.FieldCondition(key="instance_id", match=models.MatchValue(value=instance_id))]),
            limit=10,
            with_vectors=True,
            with_payload=True
        )
        points, _ = res
        vectors = []
        for p in points:
            v = p.vector
            v['payload'] = p.payload # Keep payload for context
            vectors.append(v)
        return vectors

    def retrieve(self, query: str, instance_id: str, strategy: List[str], visual_mode: str, top_k: int = 10, repo: str = None, version: str = None) -> Dict[str, Any]:
        all_results = []
        
        # 1. Fetch Vectors
        issue_vectors = self._fetch_issue_vectors(instance_id)
        visual_vectors_list = self._fetch_visual_vectors(instance_id) if visual_mode != "none" else []
        
        # 2. Prepare Queries
        # List of (source_name, vectors_dict)
        queries_to_run = [("text", issue_vectors)]
        
        if visual_mode == "fusion":
            for i, v in enumerate(visual_vectors_list):
                queries_to_run.append((f"visual_{i}", v))
        elif visual_mode == "augment":
            # For offline augment, we can't easily re-embed. 
            # We will skip augmentation for Neural strategies here or treat as fusion.
            # But for BM25 we can concatenate text.
            pass

        # 3. Execute Strategies
        for source, vectors in queries_to_run:
            if not vectors and "bm25" not in strategy: continue
            
            for strat in strategy:
                if strat == "bm25":
                    # BM25 requires text
                    # If source is text, use query (problem_statement)
                    # If source is visual, use vlm_description from payload
                    text_query = query
                    if source.startswith("visual"):
                        text_query = vectors['payload'].get('vlm_description', '')
                    
                    # If augment BM25, we concatenate
                    if visual_mode == "augment" and source == "text":
                        # Append all VLM descs
                        descs = [v['payload'].get('vlm_description', '') for v in visual_vectors_list]
                        text_query += " " + " ".join(descs)
                    
                    # Load BM25
                    if repo and version:
                        safe_repo = sanitize_path_component(repo)
                        safe_version = sanitize_path_component(version)
                        chunks_file = f"data/qdrant/qdrant_data_{safe_repo}_{safe_version}/chunks.json"
                        
                        bm25 = self._get_bm25(chunks_file)
                        if bm25:
                            tokenized_query = self._tokenize(text_query)
                            scores = bm25.get_scores(tokenized_query)
                            top_n = np.argsort(scores)[::-1][:top_k*2]
                            
                            strat_results = []
                            for idx in top_n:
                                strat_results.append({
                                    "id": self.chunks[idx].get('chunk_id', str(idx)), # Use index if no ID
                                    "score": float(scores[idx]),
                                    "content": self.chunks[idx],
                                    "source": f"{source}_bm25"
                                })
                            all_results.append(strat_results)
                    else:
                        print("Warning: Repo/Version not provided for BM25 retrieval.")
                    
                elif strat in ["dense_jina", "splade", "bge"]:
                    if strat not in vectors:
                        print(f"Warning: Vector for {strat} not found in {source}")
                        continue
                        
                    query_vector = vectors[strat]
                    vector_name = strat # Now they match!
                    
                    # Qdrant Search
                    search_result = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        using=vector_name,
                        limit=top_k*2
                    ).points
                    
                    strat_results = []
                    for hit in search_result:
                        strat_results.append({
                            "id": hit.id,
                            "score": hit.score,
                            "content": hit.payload,
                            "source": f"{source}_{strat}"
                        })
                    all_results.append(strat_results)

        # 4. Fuse
        fused_results = self._fuse_rankings(all_results)
        
        return {
            "retrieved_documents": fused_results[:top_k],
            "visual_context": [v['payload'] for v in visual_vectors_list]
        }

    def _fuse_rankings(self, results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
        fused_scores = defaultdict(float)
        doc_map = {}
        for results in results_list:
            for rank, item in enumerate(results):
                doc_id = item['id']
                fused_scores[doc_id] += 1 / (k + rank + 1)
                doc_map[doc_id] = item['content']
        
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [{"id": doc_id, "score": fused_scores[doc_id], "payload": doc_map[doc_id]} for doc_id in sorted_ids]

class OfflineExperimentRunner:
    def __init__(self, output_dir: str = "results/swebench_predictions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Global clients
        self.issues_client = QdrantClient(path="data/qdrant/qdrant_data_swe_bench_issues")
        self.images_client = QdrantClient(path="data/qdrant/qdrant_data_swe_bench_images")

    def run(self, instances: List[Dict], mock_llm: bool = False):
        print(f"Starting Offline Experiments for {len(instances)} instances...")
        
        for exp_id, config in EXPERIMENTS.items():
            print(f"\n=== Experiment: {exp_id} ===")
            predictions = []
            
            for instance in instances:
                instance_id = instance['instance_id']
                repo = instance['repo']
                version = instance['version']
                problem_statement = instance['problem_statement']
                
                # Setup Repo Client
                safe_repo = sanitize_path_component(repo)
                safe_version = sanitize_path_component(version)
                db_path = f"data/qdrant/qdrant_data_{safe_repo}_{safe_version}"
                collection_name = f"{safe_repo}_{safe_version}"
                
                if not os.path.exists(db_path):
                    print(f"DB not found for {repo} v{version}. Skipping.")
                    continue
                    
                client = QdrantClient(path=db_path)
                
                # Initialize Retriever
                retriever = OfflineRetriever(client, collection_name, self.issues_client, self.images_client)
                
                # Monkey patch retrieve for RAGPipeline
                original_retrieve = retriever.retrieve
                retriever.retrieve = lambda q, top_k: original_retrieve(
                    q, instance_id, config['strategies'], config['visual_mode'], top_k, repo=repo, version=version
                )
                
                # Initialize Pipeline
                pipeline = RAGPipeline(
                    retriever=retriever,
                    llm_provider="mock" if mock_llm else "openai", # or vllm
                    llm_model="gpt-4o-2024-08-06"
                )
                
                # Run Query
                try:
                    # Fetch visual context for prompt
                    visual_context = retriever._fetch_visual_vectors(instance_id)
                    vlm_context = [v['payload'] for v in visual_context]
                    
                    result = pipeline.query(
                        problem_statement, 
                        mode="code_gen",
                        vlm_context=vlm_context,
                        visual_input_mode="vlm_desc_url_image_file" # Pass everything to prompt
                    )
                    
                    predictions.append({
                        "instance_id": instance_id,
                        "model_patch": result['answer'],
                        "model_name_or_path": exp_id,
                        "metrics": result['metrics']
                    })
                except Exception as e:
                    print(f"Error processing {instance_id}: {e}")
                finally:
                    client.close()

            # Save Predictions
            output_file = self.output_dir / f"{exp_id}_predictions.json"
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved predictions to {output_file}")

            # Save Instance Metrics
            metrics_file = self.output_dir.parent / "instance_metrics.json"
            existing_metrics = []
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        existing_metrics = json.load(f)
                except:
                    pass
            
            # Extract metrics from predictions and append
            for p in predictions:
                m = p.get('metrics', {})
                if m:
                    m['instance_id'] = p['instance_id']
                    m['experiment_id'] = exp_id
                    m['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    existing_metrics.append(m)
                
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
            print(f"Saved instance metrics to {metrics_file}")

if __name__ == "__main__":
    # Test run
    pass
