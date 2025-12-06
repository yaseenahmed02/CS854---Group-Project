import sys
import os
import argparse
from datasets import load_dataset
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.getcwd())

from ingestion_engine import IngestionEngine

# Load environment variables
load_dotenv()

def ingest_issues(limit: int = None, split: str = "test", repo_filter: str = None, version_filter: str = None):
    """
    Ingest issue descriptions from SWE-bench Multimodal.
    """
    
    # 1. Load Dataset
    print(f"Loading SWE-bench Multimodal dataset ({split} split)...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Filter Instances
    instances = []
    count = 0
    
    print("Filtering instances...")
    for instance in dataset:
        if limit and count >= limit:
            break
            
        repo = instance.get('repo')
        version = instance.get('version')
        
        if repo_filter and repo != repo_filter:
            continue
        if version_filter and version != version_filter:
            continue
            
        instances.append(instance)
        count += 1
        
    print(f"Found {len(instances)} instances to ingest.")
    
    if not instances:
        return

    # 3. Initialize Engine & Ingest
    # We don't need VLM for text ingestion, so we can mock it to avoid OpenAI init if not needed, 
    # but IngestionEngine init might check for key. 
    # The user didn't specify VLM usage for text, but IngestionEngine init does.
    # We'll pass mock_vlm=True since we are only doing text ingestion here and don't need VLM.
    engine = IngestionEngine(mock_vlm=True)
    
    engine.ingest_issues(instances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of issues to process")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (dev/test)")
    parser.add_argument("--repo", type=str, default=None, help="Filter by repository name")
    parser.add_argument("--version", type=str, default=None, help="Filter by version")
    
    args = parser.parse_args()
    
    ingest_issues(limit=args.limit, split=args.split, repo_filter=args.repo, version_filter=args.version)
