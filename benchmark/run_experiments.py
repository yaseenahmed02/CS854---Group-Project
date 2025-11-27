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
PREDICTIONS_DIR = RESULTS_DIR / "swebench_predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

# Define Experiments
# Format: id -> {strategies: [], visual_mode: str}
EXPERIMENTS = {                                       ## Running only 6 experiments for now.
    # --- Text-Only Baselines ---
    "text_bm25":   {"strategies": ["bm25"], "visual_mode": "none"},                                         ## The "Control Group". 
    # "text_splade": {"strategies": ["splade"], "visual_mode": "none"},
    "text_bge":    {"strategies": ["bge"], "visual_mode": "none"},                                          ## The "Neural Sparse" Challenger.
    "text_dense_jina":   {"strategies": ["dense_jina"], "visual_mode": "none"},                                         ## The "Dense" Challenger.      

    # --- Text-Only Hybrid (RRF Fusion) ---
    # "text_hybrid_jina_bm25":   {"strategies": ["dense_jina", "bm25"], "visual_mode": "none"},
    # "text_hybrid_jina_splade": {"strategies": ["dense_jina", "splade"], "visual_mode": "none"},
    "text_hybrid_jina_bge":    {"strategies": ["dense_jina", "bge"], "visual_mode": "none"},                      ## The "Hybrid" Challenger, possibly the"Text-Only" Champion. 

    # --- Visual-Only ---
    # "visual_only_jina": {"strategies": ["dense_jina"], "visual_mode": "visual_only"},

    # --- Multimodal Fusion (Text + Visual) ---
    # "multimodal_fusion_bm25":   {"strategies": ["bm25"], "visual_mode": "fusion"},
    # "multimodal_fusion_splade": {"strategies": ["splade"], "visual_mode": "fusion"},
    # "multimodal_fusion_bge":    {"strategies": ["bge"], "visual_mode": "fusion"},
    "multimodal_fusion_jina":   {"strategies": ["dense_jina"], "visual_mode": "fusion"},                          ## The "Multimodal Fusion" Challenger, showing the "Visual Delta" test against `text_jina`.

    # --- The "Kitchen Sink" (Hybrid Text + Visual) ---
    "multimodal_fusion_hybrid_jina_best_sparse": {"strategies": ["dense_jina", "bge"], "visual_mode": "fusion"},  ## The "State-of-the-Art" Attempt, fusing Dense + BGE + Visual.

    # --- New Multimodal Input Modes ---
    # "multimodal_url_jina": {"strategies": ["dense_jina"], "visual_mode": "fusion", "visual_input_mode": "vlm_desc_url"},
    # "multimodal_file_jina": {"strategies": ["dense_jina"], "visual_mode": "fusion", "visual_input_mode": "image_file"},
    # "multimodal_all_jina": {"strategies": ["dense_jina"], "visual_mode": "fusion", "visual_input_mode": "vlm_desc_url_image_file"},
}

def get_collection_name(repo: str, version: str) -> str:
    """Get sanitized collection name."""
    repo_san = sanitize_path_component(repo)
    ver_san = sanitize_path_component(version)
    return f"{repo_san}_{ver_san}"

def run_experiments(limit: int = None, 
                    mock_vllm: bool = False, 
                    split: str = "test", 
                    repo_filter: str = None, 
                    version_filter: str = None,
                    total_token_limit: int = None,
                    retrieval_limit: int = None,
                    llm_model: str = "gpt-4o-2024-08-06",
                    llm_provider: str = "openai",
                    max_output_tokens: int = 16384):
    """
    Run experiments on SWE-bench Multimodal.
    """
    print(f"Running experiments (Limit: {limit}, Provider: {llm_provider}, Model: {llm_model})...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
    
    # Image Client (Step 4 used ./qdrant_data_swe_images)
    image_db_path = "data/qdrant/qdrant_data_swe_images"
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
                
                # Apply filters
                if repo_filter and repo != repo_filter:
                    continue
                if version_filter and version != version_filter:
                    continue
                
                problem_statement = instance['problem_statement']
                
                # Check if we have this repo ingested
                collection_name = get_collection_name(repo, version)
                db_path = f"data/qdrant/qdrant_data_{collection_name}"
                
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
                    
                    # Initialize Pipeline
                    # We initialize it here to allow for strategy-specific config if needed, 
                    # but for now we reuse the same retriever.
                    # Note: RAGPipeline now takes llm_model and llm_provider
                    pipeline = RAGPipeline(
                        retriever=retriever,
                        llm_model=llm_model,
                        llm_provider=llm_provider
                    )
                    
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
                    
                    # Fetch VLM descriptions if needed for token counting/context
                    vlm_context = []
                    if config['visual_mode'] != 'none':
                        vlm_context = retriever._fetch_visual_context(instance_id)

                    # NOW run query
                    # Pass retrieval_limit if set
                    query_kwargs = {"mode": "code_gen"}
                    if retrieval_limit:
                        query_kwargs["retrieval_token_limit"] = retrieval_limit
                    
                    if total_token_limit:
                        query_kwargs["total_token_limit"] = total_token_limit
                        query_kwargs["vlm_context"] = vlm_context
                    
                    if max_output_tokens:
                        query_kwargs["max_tokens"] = max_output_tokens
                    
                    query_kwargs["visual_input_mode"] = config.get("visual_input_mode", "vlm_desc_url_image_file")
                        
                    result = pipeline.query(problem_statement, **query_kwargs)
                    
                    end_time = time.time()
                    
                    prediction = {
                        "instance_id": instance_id,
                        "model_patch": result['answer'],
                        "model_name_or_path": exp_id,
                        "metrics": result['metrics'], # Store full metrics including time and token counts
                        "retrieved_documents": result['retrieved_documents'] # Persist retrieved docs for recall analysis
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
            output_file = PREDICTIONS_DIR / f"{exp_id}_predictions.json"
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved {len(predictions)} predictions to {output_file}")
            
            # Save Instance Metrics
            metrics_file = RESULTS_DIR / "instance_metrics.json"
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
                m['instance_id'] = p['instance_id']
                m['experiment_id'] = exp_id
                m['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
                existing_metrics.append(m)
                
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
            print(f"Saved instance metrics to {metrics_file}")

    finally:
        if images_client:
            print("Closing images_client...")
            images_client.close()

    print("\nAll experiments completed.")
    print("To run evaluation:")
    print("  ./evaluate_all.sh")
    print("Or manually:")
    print("  python -m swebench.harness.run_evaluation --predictions_path results/swebench_predictions/<file> --dataset_name princeton-nlp/SWE-bench_Multimodal --report_dir results/swebench_evaluation --run_id <run_id>")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit instances per experiment")
    parser.add_argument("--mock", action="store_true", help="Mock vLLM")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (dev/test)")
    parser.add_argument("--retrieval_limit", type=int, default=None, help="Token limit for retrieval context (e.g. 13000)")
    parser.add_argument("--repo", type=str, default=None, help="Filter by repository name")
    parser.add_argument("--version", type=str, default=None, help="Filter by version")
    args = parser.parse_args()
    
    run_experiments(limit=args.limit, mock_vllm=args.mock, split=args.split, retrieval_limit=args.retrieval_limit, repo_filter=args.repo, version_filter=args.version)
