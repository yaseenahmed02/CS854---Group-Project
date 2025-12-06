
import os
import json
import subprocess
import glob
from pathlib import Path

# Configuration
CONSOLIDATED_DIR = "results/consolidated"
HARNESS_OUTPUT_DIR = "results/harness_results"
TEMP_DIR = "results/temp_predictions"
DATASET_NAME = "princeton-nlp/SWE-bench_Multimodal"
SPLIT = "dev"
MAX_WORKERS = 4 

TARGET_REPO = "diegomura/react-pdf"
TARGET_VERSIONS = {"1.1", "1.2", "2.0"}

# Ensure directories exist
os.makedirs(HARNESS_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_harness():
    # 1. Provide Context
    print(f"Running SWE-bench Harness for {TARGET_REPO} versions {TARGET_VERSIONS}")
    
    # 2. Get Consolidated Files
    pred_files = glob.glob(os.path.join(CONSOLIDATED_DIR, "*_predictions.json"))
    
    # 3. Load Metrics to map instance_id -> (repo, version)
    # Actually, we can just infer from instance_id prefix "diegomura__react-pdf"
    # But to be precise on version, we might need metrics or just assume all 'diegomura__react-pdf' belong to target versions
    # since we verified earlier that only 1.1, 1.2, 2.0 exist in the predictions for this repo.
    # Step 2047 confirmed: 1.1, 1.2, 2.0 are the only versions present for diegomura/react-pdf.
    # So filtering by instance_id prefix is safe.
    
    prefix = "diegomura__react-pdf"
    
    for pred_file in sorted(pred_files):
        basename = os.path.basename(pred_file)
        approach_name = basename.replace("_predictions.json", "")
        print(f"\n>> Processing Approach: {approach_name}")
        
        # Filter predictions
        all_preds = load_json(pred_file)
        filtered_preds = [
            p for p in all_preds 
            if p.get("instance_id", "").startswith(prefix)
        ]
        
        if not filtered_preds:
            print(f"No predictions found for {TARGET_REPO} in {basename}")
            continue
            
        print(f"Found {len(filtered_preds)} instances for {TARGET_REPO}")
        
        # Save filtered predictions to temp file
        temp_pred_path = os.path.join(TEMP_DIR, f"{approach_name}_react-pdf.json")
        with open(temp_pred_path, 'w') as f:
            json.dump(filtered_preds, f, indent=2)
            
        # Run Harness
        # python -m swebench.harness.run_evaluation
        # --dataset_name princeton-nlp/SWE-bench_Multimodal 
        # --split dev 
        # --predictions_path <path> 
        # --max_workers <N>
        # --run_id <run_id>
        
        run_id = f"{approach_name}_react-pdf"
        
        cmd = [
            "./venv/bin/python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", DATASET_NAME,
            "--split", SPLIT,
            "--predictions_path", temp_pred_path,
            "--max_workers", str(MAX_WORKERS),
            "--run_id", run_id
            # Output dir is controlled by harness, typically defaults to current dir or we can specify?
            # Looking at docs/code: flags like --output_dir might exist but usually writes to file based on run_id.
        ]
        
        print(f"Running harness command: {' '.join(cmd)}")
        try:
            # We explicitly don't capture output to stream it, but orchestrator captures it.
            # Using check=False to allow continuing even if some evaluation fails.
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"Error running harness for {approach_name}: {e}")

if __name__ == "__main__":
    run_harness()
