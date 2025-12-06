import json
import glob
import os
from collections import defaultdict

CONSOLIDATED_DIR = "results/consolidated"
METRICS_FILE = os.path.join(CONSOLIDATED_DIR, "consolidated_metrics.json")
PRED_FILES = glob.glob(os.path.join(CONSOLIDATED_DIR, "*_predictions.json"))

def main():
    # 1. Load Metrics to map instance_id -> (repo, version)
    id_map = {}
    ids_per_repo_version = defaultdict(int) 
    
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                metrics = json.load(f)
                for m in metrics:
                    iid = m.get("instance_id")
                    repo = m.get("repo")
                    version = m.get("version")
                    if iid and repo and version:
                        id_map[iid] = (repo, version)
                        ids_per_repo_version[(repo, version)] += 1
                        
            # print("Known instances from metrics:")
            # for (r, v), count in ids_per_repo_version.items():
            #     print(f"  {r} v{version}: {count} instances") # Bug in version var here in loop
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return
    else:
        print("Metrics file not found. Cannot map versions.")
        return

    # 2. Analyze Prediction Files
    print(f"\nAnalyzing {len(PRED_FILES)} prediction files...\n")
    
    for pred_file in sorted(PRED_FILES):
        basename = os.path.basename(pred_file)
        try:
            with open(pred_file, 'r') as f:
                preds = json.load(f)
                
            counts = defaultdict(int)
            unknowns = 0
            
            for p in preds:
                iid = p.get("instance_id")
                if iid in id_map:
                    repo, version = id_map[iid]
                    counts[(repo, version)] += 1
                else:
                    unknowns += 1
            
            print(f"=== {basename} ({len(preds)} total) ===")
            for (repo, version), count in sorted(counts.items()):
                print(f"  {repo} v{version}: {count}")
            if unknowns > 0:
                print(f"  Unknown (not in metrics): {unknowns}")
            print()
            
        except Exception as e:
            print(f"Error reading {basename}: {e}")

if __name__ == "__main__":
    main()
