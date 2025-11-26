import argparse
import sys
import os
import json
import datetime
from datasets import load_dataset
from typing import List, Set

# Add project root to path
sys.path.insert(0, os.getcwd())

from utils.clone_repo import clone_repo
from ingest_code_to_qdrant import ingest_repo
from ingest_images_to_qdrant import ingest_images
from benchmark.run_experiments import run_experiments

def get_versions_from_dataset(repo: str, split: str) -> List[str]:
    """
    Identify all unique versions for a given repository in the dataset split.
    """
    print(f"Scanning dataset ({split}) for versions of {repo}...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
        versions = set()
        for instance in dataset:
            if instance['repo'] == repo:
                versions.add(instance['version'])
        
        sorted_versions = sorted(list(versions))
        print(f"Found versions: {sorted_versions}")
        return sorted_versions
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def run_pipeline(repo: str, split: str, version: str = None, limit: int = None, mock_vlm: bool = False, mock_llm: bool = False, vlm_model: str = "gpt-4o-2024-08-06", total_token_limit: int = None, llm_model: str = "gpt-4o-2024-08-06", llm_provider: str = "openai", max_output_tokens: int = 16384):
    """
    Orchestrate the end-to-end pipeline.
    """
    
    # 1. Identify Versions
    if version:
        versions_to_process = [version]
    else:
        versions_to_process = get_versions_from_dataset(repo, split)
        
    if not versions_to_process:
        print(f"No versions found for {repo} in {split} split.")
        return

    print(f"Starting pipeline for {repo} on {split} split.")
    print(f"Versions to process: {versions_to_process}")
    
    for ver in versions_to_process:
        print(f"\n\n{'='*50}")
        print(f"Processing Version: {ver}")
        print(f"{'='*50}\n")
        
        # 2. Clone Repository
        # clone_repo returns the path to the cloned directory
        print(f"--- Step 1: Cloning Repository ({ver}) ---")
        try:
            repo_dir = clone_repo(repo, ver)
        except Exception as e:
            print(f"Failed to clone {repo} v{ver}: {e}")
            continue
            
        # 3. Ingest Code
        print(f"\n--- Step 2: Ingesting Code ({ver}) ---")
        try:
            # We use 'create' mode which skips if collection exists
            metrics = ingest_repo(repo_dir, repo, ver, mode="create")
            
            # Save Metrics
            if metrics:
                metrics["timestamp"] = datetime.datetime.now().isoformat()
                metrics_file = os.path.join("results", "repo_metrics.json")
                os.makedirs("results", exist_ok=True)
                
                existing_metrics = []
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            existing_metrics = json.load(f)
                    except:
                        pass
                
                existing_metrics.append(metrics)
                
                with open(metrics_file, 'w') as f:
                    json.dump(existing_metrics, f, indent=2)
                print(f"Saved metrics to {metrics_file}")
                
        except Exception as e:
            print(f"Failed to ingest code for {repo} v{ver}: {e}")
            continue

        # 4. Ingest Images
        print(f"\n--- Step 3: Ingesting Images ({ver}) ---")
        try:
            # ingest_images filters by repo and version
            ingest_images(
                limit=limit, 
                mock_vlm=mock_vlm, 
                split=split, 
                repo_filter=repo, 
                version_filter=ver,
                vlm_model=vlm_model
            )
        except Exception as e:
            print(f"Failed to ingest images for {repo} v{ver}: {e}")
            continue

        # 5. Run Experiments
        print(f"\n--- Step 4: Running Experiments ({ver}) ---")
        try:
            run_experiments(
                limit=limit, 
                mock_vllm=mock_llm, 
                split=split, 
                repo_filter=repo, 
                version_filter=ver,
                total_token_limit=total_token_limit,
                llm_model=llm_model,
                llm_provider=llm_provider,
                max_output_tokens=max_output_tokens
            )
        except Exception as e:
            print(f"Failed to run experiments for {repo} v{ver}: {e}")
            continue
            
    print(f"\nPipeline completed for {repo}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End-to-End Multimodal RAG Pipeline")
    parser.add_argument("--repo", type=str, required=True, help="Repository name (e.g., markedjs/marked)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (dev/test)")
    parser.add_argument("--version", type=str, default=None, help="Specific version to process (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Limit items per step (for testing)")
    parser.add_argument("--mock", action="store_true", help="Use mocks for both VLM and LLM (deprecated, use specific flags)")
    parser.add_argument("--mock-vlm", action="store_true", help="Use mock for VLM")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock for LLM")
    parser.add_argument("--vlm_model", type=str, default="gpt-4o-2024-08-06", help="VLM model to use (default: gpt-4o-2024-08-06)")
    parser.add_argument("--total_token_limit", type=int, default=None, help="Total token limit for LLM input (e.g., 13000)")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-2024-08-06", help="LLM model to use (default: gpt-4o-2024-08-06)")
    parser.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "vllm", "mock"], help="LLM provider (default: openai)")
    parser.add_argument("--max_output_tokens", type=int, default=16384, help="Max tokens for LLM generation")
    
    args = parser.parse_args()
    
    # Handle legacy --mock
    mock_vlm = args.mock or args.mock_vlm
    mock_llm = args.mock or args.mock_llm
    
    # Override provider if mock_llm is set
    llm_provider = "mock" if mock_llm else args.llm_provider
    
    run_pipeline(args.repo, args.split, args.version, args.limit, mock_vlm, mock_llm, args.vlm_model, args.total_token_limit, args.llm_model, llm_provider, args.max_output_tokens)
