
import os
import shutil
from datasets import load_dataset
from utils.clone_repo import clone_repo
from ingest_code_to_qdrant import ingest_repo
from utils.ingestion_utils import sanitize_path_component

def ingest_missing_repos(limit=None, target_repo=None, target_version=None, mode="create"):
    print("Loading dataset (dev split)...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="dev")
    
    # Get unique repo/version pairs
    repos = set()
    for instance in dataset:
        repos.add((instance['repo'], instance['version']))
        
    print(f"Found {len(repos)} unique repositories.")
    
    count = 0
    for repo, version in repos:
        # Apply filters
        if target_repo and repo != target_repo:
            continue
        if target_version and version != target_version:
            continue
            
        if limit and count >= limit:
            print(f"Limit of {limit} reached. Stopping.")
            break
            
        collection_name = f"{sanitize_path_component(repo)}_{sanitize_path_component(version)}"
        db_path = f"data/qdrant/qdrant_data_{collection_name}"
        
        # If mode is overwrite, we don't skip if exists
        if mode != "overwrite" and os.path.exists(db_path):
            print(f"[SKIP] {collection_name} already exists.")
            continue
            
        print(f"\n=== Processing {repo} v{version} ({count+1}/{len(repos)}) ===")
        
        try:
            # 1. Clone
            repo_dir = clone_repo(repo, version, target_dir="data/repos")
            
            # 2. Ingest
            print(f"Ingesting {repo_dir}...")
            metrics = ingest_repo(repo_dir, repo, version, mode=mode)
            print(f"Ingestion metrics: {metrics}")
            
            # 3. Cleanup (Optional - to save space)
            # shutil.rmtree(repo_dir)
            
            count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process {repo} v{version}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of repos to process")
    parser.add_argument("--repo", type=str, default=None, help="Filter by repository name")
    parser.add_argument("--version", type=str, default=None, help="Filter by version")
    parser.add_argument("--mode", choices=['create', 'append', 'skip', 'overwrite'], default="create", help="Ingestion mode")
    args = parser.parse_args()
    
    ingest_missing_repos(limit=args.limit, target_repo=args.repo, target_version=args.version, mode=args.mode)
