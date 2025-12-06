import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.getcwd())

from retrieval.flexible_retriever import FlexibleRetriever
from rag.pipeline import RAGPipeline
from utils.ingestion_utils import sanitize_path_component

# Load environment variables
load_dotenv()

# Configuration
# Create timestamped run directory
timestamp = time.strftime("%Y-%m-%d_%H-%M")
RUN_DIR = Path("results") / timestamp
RUN_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = RUN_DIR / "swebench_predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

print(f"Results will be saved to: {RUN_DIR}")

# Define Experiments
# Format: id -> {strategies: [], visual_mode: str}
EXPERIMENTS = {                                       ## Running only 6 experiments for now.
    # The Text-Only experiments challenge the SWE-Bench article affirmation regarding dense vector not being suitable for this task.
    # --- Text-Only Baselines ---
    "text_bm25":   {"strategies": ["bm25"], "visual_mode": "none"},                                         ## The "Control Group". 
    "text_splade": {"strategies": ["splade"], "visual_mode": "none"},
    "text_bge":    {"strategies": ["bge"], "visual_mode": "none"},                                          ## The "Neural Sparse" Challenger.
    "text_dense_jina":   {"strategies": ["dense_jina"], "visual_mode": "none"},                                         ## The "Dense" Challenger.      

    # --- Text-Only Hybrid (RRF Fusion) ---
    "text_hybrid_jina_bm25":   {"strategies": ["dense_jina", "bm25"], "visual_mode": "none"},
    "text_hybrid_jina_splade": {"strategies": ["dense_jina", "splade"], "visual_mode": "none"},
    "text_hybrid_jina_bge":    {"strategies": ["dense_jina", "bge"], "visual_mode": "none"},                      ## The "Hybrid" Challenger, possibly the"Text-Only" Champion. 

    # --- Visual-Only ---
    "visual_only_jina": {"strategies": ["dense_jina"], "visual_mode": "visual_only"},

    # --- Multimodal Fusion (Text + Visual) ---
    "multimodal_fusion_bm25":   {"strategies": ["bm25"], "visual_mode": "fusion"},
    "multimodal_fusion_splade": {"strategies": ["splade"], "visual_mode": "fusion"},
    "multimodal_fusion_bge":    {"strategies": ["bge"], "visual_mode": "fusion"},
    "multimodal_fusion_jina":   {"strategies": ["dense_jina"], "visual_mode": "fusion"},                          ## The "Multimodal Fusion" Challenger, showing the "Visual Delta" test against `text_jina`.

    # --- The "Kitchen Sink" (Hybrid Text + Visual) ---
    "multimodal_fusion_hybrid_jina_best_sparse": {"strategies": ["dense_jina", "bge"], "visual_mode": "fusion"},  ## The "State-of-the-Art" Attempt, fusing Dense + BGE + Visual.
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
                    max_output_tokens: int = 16384,
                    experiments_filter: str = None,
                    run_dir: Path = None):
    """
    Run experiments on SWE-bench Multimodal.
    """
    if run_dir:
        # Use passed run_dir
        current_run_dir = run_dir
    else:
        # Fallback to global or create new
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        current_run_dir = Path("results") / timestamp
        current_run_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"Running experiments (Limit: {limit}, Provider: {llm_provider}, Model: {llm_model})...")
    print(f"Saving results to: {current_run_dir}")
    
    PREDICTIONS_DIR = current_run_dir / "swebench_predictions"
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
    
    # Image Client (Step 4 used ./qdrant_data_swe_images)
    image_db_path = "data/qdrant/qdrant_data_swe_images"
    images_client = QdrantClient(path=image_db_path) if os.path.exists(image_db_path) else None
    
    if not images_client:
        print("Warning: Image DB not found. Multimodal experiments might fail.")

    # Issues Client (for embedding times)
    issues_db_path = "data/qdrant/qdrant_data_swe_bench_issues"
    issues_client = QdrantClient(path=issues_db_path) if os.path.exists(issues_db_path) else None
    
    if not issues_client:
        print("Warning: Issues DB not found. Embedding time metrics will be missing.")

    try:
        # Results container
        # We want one file per experiment
        
        # Filter experiments if specified
        active_experiments = EXPERIMENTS
        if experiments_filter:
            target_experiments = [e.strip() for e in experiments_filter.split(',')]
            active_experiments = {k: v for k, v in EXPERIMENTS.items() if k in target_experiments}
            print(f"Running specific experiments: {list(active_experiments.keys())}")
    
        for exp_id, config in active_experiments.items():
            print(f"\n=== Running Experiment: {exp_id} ===")
            print(f"Config: {config}")
            
            predictions = []
            count = 0
            
            for instance in dataset:
                # if limit and count >= limit:
                #    break
                    
                instance_id = instance['instance_id']
                repo = instance['repo']
                version = instance['version']
                
                # Apply filters
                if repo_filter and repo != repo_filter:
                    # print(f"Skipping {repo} != {repo_filter}")
                    continue
                if version_filter and version != version_filter:
                    # print(f"Skipping {version} != {version_filter}")
                    continue
                
                print(f"Found match: {instance_id} ({repo} v{version})")
                
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
                    
                    # Add Issue Embedding Time to Metrics
                    if issues_client:
                        try:
                            # Fetch issue payload
                            points = issues_client.scroll(
                                collection_name="swe_bench_issues",
                                scroll_filter=models.Filter(
                                    must=[
                                        models.FieldCondition(
                                            key="instance_id",
                                            match=models.MatchValue(value=instance_id)
                                        )
                                    ]
                                ),
                                limit=1,
                                with_payload=True,
                                with_vectors=False
                            )[0]
                            
                            if points:
                                payload = points[0].payload
                                strategies = config['strategies']
                                total_embed_time = 0.0
                                
                                # Granular times
                                jina_time = payload.get('embedding_time_ms_jina', 0)
                                splade_time = payload.get('embedding_time_ms_splade', 0)
                                bge_time = payload.get('embedding_time_ms_bge', 0)
                                
                                # Reconstruct metrics dictionary in desired order
                                ordered_metrics = {}
                                
                                # 1. VLM Metrics
                                if config['visual_mode'] != 'none':
                                    ordered_metrics['num_images'] = result.get('num_images', 0)
                                    ordered_metrics['vlm_generation_time_ms'] = result.get('vlm_generation_time_ms', 0)
                                    ordered_metrics['visual_embedding_time_ms'] = result.get('visual_embedding_time_ms', 0)
                                
                                # 2. Issue Embedding Metrics (Granular + Total)
                                if "dense_jina" in strategies:
                                    ordered_metrics['issue_embedding_time_ms_jina'] = jina_time
                                    total_embed_time += jina_time
                                if "splade" in strategies:
                                    ordered_metrics['issue_embedding_time_ms_splade'] = splade_time
                                    total_embed_time += splade_time
                                if "bge" in strategies:
                                    ordered_metrics['issue_embedding_time_ms_bge'] = bge_time
                                    total_embed_time += bge_time
                                    
                                ordered_metrics['issue_embedding_time_ms'] = total_embed_time

                                # 3. Retrieval Metrics
                                # Note: RAGPipeline now sets 'retrieval_time_ms' to pure search time if available
                                breakdown = result.get('search_time_breakdown', {})
                                if "bm25" in strategies:
                                    ordered_metrics['retrieval_time_ms_bm25'] = breakdown.get('search_time_ms_bm25', 0)
                                if "splade" in strategies:
                                    ordered_metrics['retrieval_time_ms_splade'] = breakdown.get('search_time_ms_splade', 0)
                                if "bge" in strategies:
                                    ordered_metrics['retrieval_time_ms_bge'] = breakdown.get('search_time_ms_bge', 0)
                                if "dense_jina" in strategies:
                                    ordered_metrics['retrieval_time_ms_jina'] = breakdown.get('search_time_ms_dense_jina', 0)
                                
                                ordered_metrics['retrieval_time_ms'] = result['metrics'].get('retrieval_time_ms', 0)
                                
                                # 4. Total Retrieval Time
                                # Total = Issue Embed + VLM Gen + Visual Embed + Search
                                total_retrieval_time = (
                                    total_embed_time + 
                                    ordered_metrics.get('vlm_generation_time_ms', 0) + 
                                    ordered_metrics.get('visual_embedding_time_ms', 0) +
                                    ordered_metrics['retrieval_time_ms']
                                )
                                ordered_metrics['total_retrieval_time_ms'] = total_retrieval_time
                                
                                # 4. Generation & IO Metrics
                                ordered_metrics['generation_time_ms'] = result['metrics'].get('generation_time_ms', 0)
                                ordered_metrics['total_io_time_ms'] = total_retrieval_time + ordered_metrics['generation_time_ms']
                                
                                # 5. Token Metrics (Copy remaining)
                                for k, v in result['metrics'].items():
                                    if k not in ordered_metrics:
                                        ordered_metrics[k] = v
                                        
                                # Update result metrics with ordered version
                                # Round time metrics to 2 decimal places
                                for k, v in ordered_metrics.items():
                                    if 'time_ms' in k and isinstance(v, (int, float)):
                                        ordered_metrics[k] = round(v, 2)
                                        
                                result['metrics'] = ordered_metrics
                                
                        except Exception as e:
                            print(f"Error fetching issue metrics: {e}")
                    
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
            metrics_file = RUN_DIR / "instance_metrics.json"
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
    parser.add_argument("--total_token_limit", type=int, default=None, help="Total token limit for the LLM context (e.g. 16000)")
    parser.add_argument("--repo", type=str, default=None, help="Filter by repository name")
    parser.add_argument("--version", type=str, default=None, help="Filter by version")
    parser.add_argument("--experiments", type=str, default=None, help="Comma-separated list of experiment IDs to run")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="LLM model to use (e.g., 'gpt-4o', 'claude-3-opus-20240229')")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider (e.g., 'openai', 'anthropic')")
    parser.add_argument("--max_output_tokens", type=int, default=None, help="Maximum number of tokens to generate in the LLM response")
    parser.add_argument("--output_dir", type=str, help="Custom output directory for results")
    
    args = parser.parse_args()

    # Determine RUN_DIR
    if args.output_dir:
        RUN_DIR = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        RUN_DIR = Path("results") / timestamp
    
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR = RUN_DIR / "swebench_predictions"
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    print(f"Results will be saved to: {RUN_DIR}")

    run_experiments(
        limit=args.limit, 
        mock_vllm=args.mock, 
        split=args.split,
        repo_filter=args.repo,
        version_filter=args.version,
        retrieval_limit=args.retrieval_limit,
        total_token_limit=args.total_token_limit,
        llm_model=args.model,
        llm_provider=args.provider,
        max_output_tokens=args.max_output_tokens,
        experiments_filter=args.experiments,
        run_dir=RUN_DIR # Pass RUN_DIR to function
    )

