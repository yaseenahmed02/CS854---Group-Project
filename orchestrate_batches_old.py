import os
import sys
import json
import time
import subprocess
from typing import List, Dict, Tuple
from datasets import load_dataset

# Configuration
TARGET_PROJECTS = [
    "diegomura/react-pdf",
    "markedjs/marked",
    "processing/p5.js", 
    # "chartjs/Chart.js",
    # "Automattic/wp-calypso", 
]

BATCHES = {
    "Batch 1": [
        "text_bm25",
        "text_bge",
        "text_dense_jina",
        "text_hybrid_jina_bm25",
        "text_hybrid_jina_bge"
    ],
    "Batch 2": [
        "multimodal_fusion_bm25",
        "multimodal_fusion_bge",
        "multimodal_fusion_jina"
    ],
    "Batch 3": [
        "multimodal_fusion_hybrid_jina_best_sparse"
    ],
    "Batch 4": [
        "text_splade",
        "text_hybrid_jina_splade",
        "multimodal_fusion_splade"
    ]
}

def get_repo_versions(projects: List[str], split: str = "dev") -> Dict[str, List[str]]:
    """
    Get all versions for the specified projects from the dataset.
    Returns a dict: {repo_name: [version1, version2, ...]}
    """
    print(f"Loading dataset split '{split}' to find versions...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
    
    repo_versions = {p: set() for p in projects}
    
    for instance in dataset:
        repo = instance['repo']
        if repo in projects:
            repo_versions[repo].add(instance['version'])
            
    # Sort versions for consistent execution order
    return {k: sorted(list(v)) for k, v in repo_versions.items()}

def run_command(command: str):
    """Run a shell command and check for errors."""
    print(f"Running: {command}")
    try:
        # Capture output to print on error
        subprocess.run(command, shell=True, check=True, capture_output=False) 
        # Note: capture_output=False lets it stream to stdout/stderr which is better for monitoring.
        # If it fails, the error is already in the logs.
        # But if we want to catch it and print it specifically:
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        raise e

def ingest_repo(repo: str, version: str):
    """Ingest a specific repo version from a dedicated directory."""
    print(f"\n--- Ingesting {repo} v{version} ---")
    
    # 1. Clone/Checkout into version-specific folder
    # Path: data/repos/{owner}/{name}-{version}
    owner, name = repo.split('/')
    version_suffix = version.replace('.', '_') # Avoid dot issues if any
    local_path = f"data/repos/{owner}/{name}-{version}"
    
    # Check if already cloned/exists
    if not os.path.exists(local_path):
        print(f"Cloning {repo} to {local_path}...")
        os.makedirs(f"data/repos/{owner}", exist_ok=True)
        run_command(f"git clone https://github.com/{repo}.git {local_path}")
    else:
        print(f"Directory {local_path} already exists. Using existing clone.")
        
    # Always enforce checkout to ensure correct version
    print(f"Checking out version {version} in {local_path}...")
    cwd = os.getcwd()
    try:
        os.chdir(local_path)
        # Clean state
        run_command("git reset --hard")
        run_command("git clean -fd")
        run_command("git fetch --all --tags")
        
        # Try multiple tag formats
        # Dataset version might be '0.3', tag might be 'v0.3.0', '0.3.0', 'v0.3', '0.3'
        candidates = [
            f"v{version}",
            f"{version}",
            f"v{version}.0",
            f"{version}.0",
            f"v{version}.0.0",
            f"{version}.0.0"
        ]
        
        success = False
        for tag in candidates:
            try:
                print(f"Trying checkout {tag}...")
                run_command(f"git checkout {tag}")
                success = True
                break
            except:
                continue
                
        if not success:
            print(f"CRITICAL: Could not checkout version {version}. Tried: {candidates}")
            # Fallback: List tags and try to find partial match?
            # For now, just raise to stop ingestion of this version
            raise Exception(f"Failed to checkout version {version}")
            
        # Verify checkout
        run_command("git log -1 --format='%H %d'")
    finally:
        os.chdir(cwd)
        
    # 2. Run Ingestion Script
    # We use --mode overwrite to ensure we get fresh metrics
    cmd = f"./venv/bin/python ingest_code_to_qdrant.py {local_path} {repo} {version} --mode overwrite"
    run_command(cmd)

def run_experiment_batch(batch_name: str, experiment_ids: List[str], repo: str, version: str):
    """Run a batch of experiments for a specific repo version."""
    print(f"\n--- Running {batch_name} for {repo} v{version} ---")
    
    exp_list = ",".join(experiment_ids)
    # We filter by repo and version to run only relevant instances
    cmd = (f"./venv/bin/python benchmark/run_experiments.py "
           f"--repo {repo} --version {version} --split dev "
           f"--experiments {exp_list}")
    
    run_command(cmd)

def main():
    # 1. Get Versions
    repo_versions = get_repo_versions(TARGET_PROJECTS)
    
    print("\nFound the following versions to process:")
    for repo, versions in repo_versions.items():
        print(f"{repo}: {versions}")
        
    # 2. Phase 1: Ingestion & Batch 1 (Text Baselines)
    print("\n=== Phase 1: Ingestion & Batch 1 (Text Baselines) ===")
    for repo in TARGET_PROJECTS:
        versions = repo_versions.get(repo, [])
        print(f"\n>> Processing Project: {repo} ({len(versions)} versions)")
        
        # Step A: Ingest ALL versions
        print(f"\n[Ingestion Phase] Ingesting all versions for {repo}...")
        successful_ingestions = []
        for version in versions:
            try:
                print(f"*** STARTING INGESTION FOR {repo} v{version} ***")
                ingest_repo(repo, version)
                print(f"*** INGESTION SUCCESSFUL FOR {repo} v{version} ***")
                successful_ingestions.append(version)
            except Exception as e:
                print(f"CRITICAL ERROR ingesting {repo} v{version}: {e}")
                print("Skipping to next version...")
        
        # Step B: Run Batch 1 for SUCCESSFULLY INGESTED versions
        print(f"\n[Experiment Phase] Running Batch 1 for {len(successful_ingestions)} versions of {repo}...")
        if not successful_ingestions:
            print(f"WARNING: No versions were successfully ingested for {repo}. Skipping experiments.")
            continue
            
        for version in successful_ingestions:
            try:
                # Optional: Check if experiments already ran? 
                # For now, we re-run experiments. It's safer and relatively fast.
                run_experiment_batch("Batch 1", BATCHES["Batch 1"], repo, version)
            except Exception as e:
                print(f"Error running Batch 1 for {repo} v{version}: {e}")

    # 3. Phase 2: Subsequent Batches (2, 3, 4)
    # Only runs after ALL projects have completed Phase 1
    
    for batch_name in ["Batch 2", "Batch 3", "Batch 4"]:
        print(f"\n=== Phase 2+: {batch_name} ===")
        for repo in TARGET_PROJECTS:
            versions = repo_versions.get(repo, [])
            # We should ideally only run for successfully ingested versions, 
            # but we lost that state (local var). 
            # Ideally we should check if Qdrant collection exists.
            # For now, we'll try all, and if Qdrant is missing, run_experiments.py might fail or skip.
            
            for version in versions:
                try:
                    run_experiment_batch(batch_name, BATCHES[batch_name], repo, version)
                except Exception as e:
                     print(f"Error processing {repo} v{version} in {batch_name}: {e}")

    print("\nAll phases completed.")

if __name__ == "__main__":
    main()
