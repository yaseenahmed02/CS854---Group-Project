import json
import glob
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from typing import List, Set, Dict, Any

def get_gold_files(patch_text: str) -> Set[str]:
    """
    Extract modified files from a unified diff patch.
    
    Args:
        patch_text: The unified diff string.
        
    Returns:
        Set of modified file paths (relative to repo root).
    """
    files = set()
    if not patch_text:
        return files
        
    # Regex to find modified files in diff
    # Matches: --- a/path/to/file or +++ b/path/to/file
    # We want to capture the path after a/ or b/
    pattern = re.compile(r'^(?:--- a/|\+\+\+ b/)(.*)$', re.MULTILINE)
    
    for match in pattern.finditer(patch_text):
        path = match.group(1).strip()
        # Sometimes paths might be /dev/null for new/deleted files, ignore those if needed
        # But usually we want to know if we retrieved the file that was deleted or created (context)
        if path == '/dev/null':
            continue
        files.add(path)
        
    return files

def get_retrieved_files(retrieved_docs: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract and normalize file paths from retrieved documents.
    
    Args:
        retrieved_docs: List of retrieved document dictionaries.
        
    Returns:
        Set of normalized file paths.
    """
    files = set()
    for doc in retrieved_docs:
        # Path can be in 'path' or 'metadata' -> 'path' or 'payload' -> 'filepath'
        # Based on ingest_code_to_qdrant.py: payload['filepath']
        # Based on pipeline.py: doc.get('path') or doc.get('metadata', {}).get('path')
        # Let's try all locations
        path = doc.get('path')
        if not path:
            path = doc.get('payload', {}).get('filepath')
        if not path:
            path = doc.get('metadata', {}).get('path')
            
        if not path:
            continue
            
        # Normalize Path
        # The path might be absolute: /Users/.../repo_name/src/file.py
        # We want: src/file.py
        # Strategy: Find the repo name in the path and take everything after it.
        # But we don't strictly know the repo name here unless we pass it.
        # However, the payload has 'repo' field!
        repo_name = doc.get('payload', {}).get('repo')
        
        normalized_path = normalize_path(path, repo_name)
        if normalized_path:
            files.add(normalized_path)
            
    return files

def normalize_path(path_str: str, repo_name: str = None) -> str:
    """
    Normalize absolute path to relative path based on repo name.
    """
    if not path_str:
        return None
        
    # If it's already relative (doesn't start with /), assume it's good
    if not path_str.startswith('/'):
        return path_str
        
    # Try to split by repo name if available
    if repo_name:
        # repo_name is like "owner/repo" or just "repo"
        # We care about the "repo" part usually
        repo_part = repo_name.split('/')[-1]
        
        if f"/{repo_part}/" in path_str:
            parts = path_str.split(f"/{repo_part}/")
            if len(parts) > 1:
                return parts[-1]
    
    # Fallback: try to find standard repo indicators if repo_name fails or is missing
    # This is heuristic and might be flaky
    parts = path_str.split('/')
    if 'test_repo' in parts: # For testing
        idx = parts.index('test_repo')
        return "/".join(parts[idx+1:])
        
    # If we can't normalize, return the basename as a last resort? 
    # Or return the full path and hope for the best?
    # Let's return the full path but warn? 
    # Actually, for the purpose of matching, if we can't normalize, we probably won't match.
    # Let's try to return the relative path from the current working directory if possible
    try:
        return str(Path(path_str).relative_to(os.getcwd()))
    except ValueError:
        pass
        
    return path_str

def calculate_recall(retrieved: Set[str], gold: Set[str]) -> float:
    """
    Calculate recall: |Intersection| / |Gold|
    """
    if not gold:
        return 1.0 # If no gold files (e.g. only message change?), recall is trivially 1? Or 0?
                   # Usually patches modify files. If gold is empty, it's an edge case.
                   # Let's assume 0 if we expected something, but 1 if there was nothing to find.
                   # In SWE-bench, there's always a patch.
        return 0.0
        
    intersection = retrieved.intersection(gold)
    return len(intersection) / len(gold)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="Dataset split (dev/test)")
    args = parser.parse_args()

    results_dir = Path("results")
    prediction_files = glob.glob(str(results_dir / "*_predictions.json"))
    
    if not prediction_files:
        print("No prediction files found in results/")
        return

    print(f"Loading SWE-bench Multimodal dataset ({args.split} split)...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=args.split)
    
    # Create lookup
    ground_truth = {item['instance_id']: item for item in dataset}
    
    all_stats = []
    summary_stats = []
    
    for pred_file in prediction_files:
        print(f"Processing {pred_file}...")
        try:
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")
            continue
            
        experiment_id = Path(pred_file).stem.replace("_predictions", "")
        recalls = []
        
        for pred in predictions:
            instance_id = pred['instance_id']
            
            if instance_id not in ground_truth:
                print(f"Warning: Instance {instance_id} not found in ground truth.")
                continue
                
            gold_item = ground_truth[instance_id]
            gold_patch = gold_item['patch']
            gold_files = get_gold_files(gold_patch)
            
            # Retrieve docs might be missing if we run on old results
            retrieved_docs = pred.get('retrieved_documents', [])
            retrieved_files = get_retrieved_files(retrieved_docs)
            
            recall = calculate_recall(retrieved_files, gold_files)
            recalls.append(recall)
            
            # Intersection for stats
            matches = retrieved_files.intersection(gold_files)
            
            all_stats.append({
                "experiment_id": experiment_id,
                "instance_id": instance_id,
                "recall": recall,
                "num_gold_files": len(gold_files),
                "num_retrieved_files": len(retrieved_files),
                "num_matches": len(matches),
                "gold_files": list(gold_files),
                "retrieved_files": list(retrieved_files)
            })
            
        if recalls:
            avg_recall = np.mean(recalls)
            summary_stats.append({
                "Experiment ID": experiment_id,
                "Average Recall": avg_recall,
                "Count": len(recalls)
            })
            
    # Output Table
    if summary_stats:
        df_summary = pd.DataFrame(summary_stats)
        print("\n=== Recall Analysis Summary ===")
        print(df_summary.to_string(index=False))
        
    # Save Detailed Stats
    if all_stats:
        df_detailed = pd.DataFrame(all_stats)
        output_csv = results_dir / "recall_analysis.csv"
        df_detailed.to_csv(output_csv, index=False)
        print(f"\nDetailed analysis saved to {output_csv}")

if __name__ == "__main__":
    main()
