import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.getcwd())

from retrieval.flexible_retriever import FlexibleRetriever
from rag.pipeline import RAGPipeline
from utils.ingestion_utils import sanitize_path_component

# Load environment variables
load_dotenv()

# Configuration
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Define Experiments
# Format: id -> {strategies: [], visual_mode: str}
EXPERIMENTS = {
    # --- Text-Only Baselines ---
    "text_bm25":   {"strategies": ["bm25"], "visual_mode": "none"},
    "text_splade": {"strategies": ["splade"], "visual_mode": "none"},
    "text_bge":    {"strategies": ["bge"], "visual_mode": "none"},
    "text_jina":   {"strategies": ["jina"], "visual_mode": "none"},

    # --- Text-Only Hybrid (RRF Fusion) ---
    "text_hybrid_jina_bm25":   {"strategies": ["jina", "bm25"], "visual_mode": "none"},
    "text_hybrid_jina_splade": {"strategies": ["jina", "splade"], "visual_mode": "none"},
    "text_hybrid_jina_bge":    {"strategies": ["jina", "bge"], "visual_mode": "none"},

    # --- Visual-Only ---
    "visual_only_jina": {"strategies": ["jina"], "visual_mode": "visual_only"},

    # --- Multimodal Fusion (Text + Visual) ---
    "multimodal_fusion_bm25":   {"strategies": ["bm25"], "visual_mode": "fusion"},
    "multimodal_fusion_splade": {"strategies": ["splade"], "visual_mode": "fusion"},
    "multimodal_fusion_bge":    {"strategies": ["bge"], "visual_mode": "fusion"},
    "multimodal_fusion_jina":   {"strategies": ["jina"], "visual_mode": "fusion"},

    # --- The "Kitchen Sink" (Hybrid Text + Visual) ---
    "multimodal_fusion_hybrid_jina_best_sparse": {"strategies": ["jina", "bge"], "visual_mode": "fusion"}, # Double-check if it will be BGE
}

def get_collection_name(repo: str, version: str) -> str:
    """Get sanitized collection name."""
    repo_san = sanitize_path_component(repo)
    ver_san = sanitize_path_component(version)
    return f"{repo_san}_{ver_san}"

def run_experiments(limit: int = None, mock_vllm: bool = False, split: str = "test"):
    """
    Run all experiments.
    """
    print(f"Loading SWE-bench Multimodal dataset ({split} split)...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
    
    # Image Client (Step 4 used ./qdrant_data_swe_images)
    image_db_path = "./qdrant_data_swe_images"
    images_client = QdrantClient(path=image_db_path) if os.path.exists(image_db_path) else None
    
    if not images_client:
        print("Warning: Image DB not found. Multimodal experiments might fail.")

    try:
        # Results container
        # We want one file per experiment
        
        for exp_id, config in EXPERIMENTS.items():
            print(f"\n=== Running Experiment: {exp_id} ===")
            print(f"Config: {config}")
            
            predictions = []
            count = 0
            
            for instance in dataset:
                if limit and count >= limit:
                    break
                    
                instance_id = instance['instance_id']
                repo = instance['repo']
                version = instance['version']
                problem_statement = instance['problem_statement']
                
                # Check if we have this repo ingested
                collection_name = get_collection_name(repo, version)
                db_path = f"./qdrant_data_{collection_name}"
                
                if not os.path.exists(db_path):
                    if count == 0 and os.path.exists("./qdrant_data_test_repo_0_0_1"):
                        print(f"Using test_repo DB for instance {instance_id} (Testing mode)")
                        db_path = "./qdrant_data_test_repo_0_0_1"
                        collection_name = "test_repo_0_0_1"
                    else:
                        # print(f"Warning: DB for {repo} v{version} not found at {db_path}. Skipping instance {instance_id}.")
                        continue
                
                print(f"Processing {instance_id} ({repo} v{version})...")
                
                try:
                    # Init Client for this repo
                    client = QdrantClient(path=db_path)
                    
                    # Init Retriever
                    retriever = FlexibleRetriever(
                        client=client,
                        collection_name=collection_name,
                        swe_images_collection="swe_images",
                        images_client=images_client
                    )
                    
                    # Init Pipeline
                    # We need to bypass RAGPipeline's default retriever init which loads files
                    from unittest.mock import patch, MagicMock
                    
                    with patch('rag.pipeline.HybridRetriever') as MockRetriever:
                        # Configure mock to not crash
                        MockRetriever.return_value = MagicMock()
                        
                        pipeline = RAGPipeline(retriever_type='hybrid', vllm_url='http://localhost:8000')
                    
                    # Inject our actual retriever
                    pipeline.retriever = retriever 
                    
                    # Mock vLLM if requested
                    if mock_vllm:
                        pipeline._call_vllm = lambda prompt, **kwargs: {
                            'text': f"Fixed code for {instance_id}\n```python\n# Fixed\n```",
                            'tokens_generated': 10
                        }
                    
                    # Run Query
                    start_time = time.time()
                    
                    # Bind args to retriever.retrieve
                    original_retrieve = retriever.retrieve
                    retriever.retrieve = lambda q, top_k: original_retrieve(
                        q, 
                        instance_id=instance_id,
                        strategy=config['strategies'],
                        visual_mode=config['visual_mode'],
                        top_k=top_k
                    )
                    
                    # NOW run query
                    result = pipeline.query(problem_statement, mode="code_gen")
                    
                    end_time = time.time()
                    
                    prediction = {
                        "instance_id": instance_id,
                        "model_patch": result['answer'],
                        "model_name_or_path": exp_id,
                        "metrics": result['metrics'] # Store full metrics including time and token counts
                    }
                    predictions.append(prediction)
                    count += 1
                    
                except Exception as e:
                    print(f"Error processing {instance_id}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Close client to release lock
                    if 'client' in locals() and client:
                        client.close()
            
            # Save Results
            output_file = RESULTS_DIR / f"{exp_id}_predictions.json"
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved {len(predictions)} predictions to {output_file}")

    finally:
        if images_client:
            print("Closing images_client...")
            images_client.close()

    print("\nAll experiments completed.")
    print("To run evaluation:")
    print("python -m swebench.harness.run_evaluation --predictions_path results/ --dataset_name princeton-nlp/SWE-bench_Multimodal")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit instances per experiment")
    parser.add_argument("--mock", action="store_true", help="Mock vLLM")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (dev/test)")
    args = parser.parse_args()
    
    run_experiments(limit=args.limit, mock_vllm=args.mock, split=args.split)
