import os
import json
import glob
from pathlib import Path

RESULTS_DIR = "results"
CONSOLIDATED_DIR = os.path.join(RESULTS_DIR, "consolidated")
os.makedirs(CONSOLIDATED_DIR, exist_ok=True)

PREDICTION_FILES = [
    "text_bge_predictions.json",
    "text_bm25_predictions.json",
    "text_dense_jina_predictions.json",
    "text_hybrid_jina_bge_predictions.json",
    "text_hybrid_jina_bm25_predictions.json",
    "multimodal_fusion_bge_predictions.json",
    "multimodal_fusion_bm25_predictions.json",
    "multimodal_fusion_jina_predictions.json",
    "multimodal_fusion_hybrid_jina_best_sparse_predictions.json",
    "multimodal_fusion_splade_predictions.json",
    "text_hybrid_jina_splade_predictions.json",
    "text_splade_predictions.json"
]

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    print(f"Consolidating results from {RESULTS_DIR} to {CONSOLIDATED_DIR}...")
    
    # Initialize containers
    consolidated_predictions = {fname: [] for fname in PREDICTION_FILES}
    consolidated_metrics = []
    
    # Find all run directories (exclude files and consolidated dir)
    run_dirs = [
        d for d in glob.glob(os.path.join(RESULTS_DIR, "*")) 
        if os.path.isdir(d) and os.path.basename(d) != "consolidated"
    ]
    
    print(f"Found {len(run_dirs)} run directories.")
    
    for run_dir in run_dirs:
        print(f"Processing {os.path.basename(run_dir)}...")
        dirname = os.path.basename(run_dir)
        
        # Try to parse repo/version from dirname
        # Format: {safe_repo}_{safe_version}_{timestamp}
        # safe_repo = repo.replace('/', '_')
        # safe_version = version.replace('.', '_')
        # This parsing is tricky because of multiple underscores.
        # Known projects: 
        # diegomura/react-pdf -> diegomura_react-pdf
        # markedjs/marked -> markedjs_marked
        # processing/p5.js -> processing_p5_js
        
        repo = None
        version = None
        
        if dirname.startswith("diegomura_react-pdf"):
            repo = "diegomura/react-pdf"
            # remaining: _1_1_TIMESTAMP or _1_2_TIMESTAMP
            remainder = dirname[len("diegomura_react-pdf_"):]
            # version is everything before the date (YYYY-MM-DD)
            # date starts with 2025
            parts = remainder.split("_2025")
            if len(parts) > 0:
                ver_str = parts[0]
                version = ver_str.replace("_", ".")
        elif dirname.startswith("markedjs_marked"):
            repo = "markedjs/marked"
            remainder = dirname[len("markedjs_marked_"):]
            parts = remainder.split("_2025")
            if len(parts) > 0:
                ver_str = parts[0]
                version = ver_str.replace("_", ".")
        elif dirname.startswith("processing_p5.js"): # p5.js -> processing/p5.js
            repo = "processing/p5.js"
            remainder = dirname[len("processing_p5.js_"):]
            parts = remainder.split("_2025")
            if len(parts) > 0:
                ver_str = parts[0]
                version = ver_str.replace("_", ".")
        
        if not repo or not version:
            print(f"  Warning: Could not parse repo/version from {dirname}")

        # 1. Process Instance Metrics
        metrics_path = os.path.join(run_dir, "instance_metrics.json")
        if os.path.exists(metrics_path):
            metrics = load_json(metrics_path)
            if metrics:
                items = []
                if isinstance(metrics, list):
                    items = metrics
                elif isinstance(metrics, dict):
                    items = [metrics]
                
                # Inject repo/version
                for item in items:
                    if repo: item["repo"] = repo
                    if version: item["version"] = version
                    consolidated_metrics.append(item)
        
        # 2. Process Predictions
        predictions_dir = os.path.join(run_dir, "swebench_predictions")
        if os.path.exists(predictions_dir):
            for fname in PREDICTION_FILES:
                pred_path = os.path.join(predictions_dir, fname)
                if os.path.exists(pred_path):
                    preds = load_json(pred_path)
                    if preds:
                        if isinstance(preds, list):
                            # Optional: inject repo/version into predictions too?
                            # Not strictly asked but helpful. The user asked "which repo/version are present".
                            # If I inject it, I can see it.
                            # But predictions usually have strict schema.
                            # I won't inject into predictions to avoid breaking SWE-bench harness.
                            # But I rely on metrics to map instance_id -> repo/version.
                            consolidated_predictions[fname].extend(preds)
                        else:
                            print(f"Warning: {fname} in {run_dir} is not a list.")

    # Write Consolidated Predictions
    print("\nWriting consolidated predictions...")
    for fname, preds in consolidated_predictions.items():
        if preds:
            out_path = os.path.join(CONSOLIDATED_DIR, fname)
            with open(out_path, "w") as f:
                json.dump(preds, f, indent=2)
            print(f"  {fname}: {len(preds)} predictions -> {out_path}")
        else:
            print(f"  {fname}: No predictions found.")
            
    # Write Consolidated Metrics
    if consolidated_metrics:
        metrics_out_path = os.path.join(CONSOLIDATED_DIR, "consolidated_metrics.json")
        with open(metrics_out_path, "w") as f:
            json.dump(consolidated_metrics, f, indent=2)
        print(f"\nConsolidated metrics: {len(consolidated_metrics)} items -> {metrics_out_path}")
    else:
        print("\nNo metrics found to consolidate.")

if __name__ == "__main__":
    main()
