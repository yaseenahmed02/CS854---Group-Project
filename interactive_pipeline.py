import sys
import os
import questionary
from datasets import load_dataset
from typing import List, Dict, Any
import json
import pathlib

# Add project root to path
sys.path.insert(0, os.getcwd())

def load_dataset_split(split: str):
    print(f"Loading dataset split: {split}...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def get_repo_version_counts(dataset):
    counts = {}
    for instance in dataset:
        key = (instance['repo'], instance['version'])
        counts[key] = counts.get(key, 0) + 1
    return counts

def main():
    print("\n==================================================")
    print("      SWE-bench Multimodal Interactive Wizard     ")
    print("==================================================\n")

    # Phase 1.1: Split Selection
    split = questionary.select(
        "Select Dataset Split:",
        choices=["dev", "test"]
    ).ask()
    
    if not split:
        sys.exit(0)

    dataset = load_dataset_split(split)

    # Phase 1.2: Target Selection
    repo_ver_counts = get_repo_version_counts(dataset)
    
    # Format choices
    choices = []
    sorted_keys = sorted(repo_ver_counts.keys())
    for repo, ver in sorted_keys:
        count = repo_ver_counts[(repo, ver)]
        choices.append(questionary.Choice(
            title=f"{repo} | {ver} | {count} instances",
            value=(repo, ver)
        ))

    selected_targets = questionary.checkbox(
        "Select Targets (Repo | Version | Count):",
        choices=choices
    ).ask()

    if not selected_targets:
        print("No targets selected. Exiting.")
        sys.exit(0)

    # Phase 1.3: Instance Preview & Filtering
    selected_instances = []
    total_images = 0
    
    print("\n--- Instance Preview ---")
    print(f"{'Repo':<30} | {'Version':<10} | {'Instance ID':<40} | {'# Images':<10}")
    print("-" * 100)

    for instance in dataset:
        if (instance['repo'], instance['version']) in selected_targets:
            # Check images
            # Check images
            # image_assets is a list of paths/urls
            images = instance.get('image_assets', [])
            
            # Normalize images to list if it's not already
            if images is None:
                images = []
            elif isinstance(images, str):
                images = [images]
            elif isinstance(images, dict):
                # Handle dict case (e.g. {'url': [...]})
                all_imgs = []
                for k, v in images.items():
                    if isinstance(v, list):
                        all_imgs.extend(v)
                    elif isinstance(v, str):
                        all_imgs.append(v)
                images = all_imgs
                
            num_images = len(images)
            
            selected_instances.append(instance)
            total_images += num_images
            
            print(f"{instance['repo']:<30} | {instance['version']:<10} | {instance['instance_id']:<40} | {num_images:<10}")

    print("-" * 100)
    print(f"Total Instances Selected: {len(selected_instances)} | Total Images: {total_images}")
    
    proceed = questionary.confirm("Proceed with ingestion?").ask()
    
    if not proceed:
        print("Aborted.")
        sys.exit(0)

    # Filter out instances with no images
    final_instances = []
    skipped_count = 0
    
    print("\nFiltering instances with valid images...")
    print(f"{'Repo':<30} | {'Version':<10} | {'Instance ID':<40} | {'# Images':<10}")
    print("-" * 100)
    
    for instance in selected_instances:
        images = instance.get('image_assets', [])
        
        # Normalize images to list
        if images is None:
            images = []
        elif isinstance(images, str):
            images = [images]
        elif isinstance(images, dict):
            all_imgs = []
            for k, v in images.items():
                if isinstance(v, list):
                    all_imgs.extend(v)
                elif isinstance(v, str):
                    all_imgs.append(v)
            images = all_imgs
            
        if images and len(images) > 0:
            final_instances.append(instance)
            print(f"{instance['repo']:<30} | {instance['version']:<10} | {instance['instance_id']:<40} | {len(images):<10}")
        else:
            skipped_count += 1
            
    print("-" * 100)
    
    if skipped_count > 0:
        print(f"Warning: Filtered out {skipped_count} instances that have no images.")
    
    print(f"Final Count: {len(final_instances)} instances ready for ingestion.")
    
    if not final_instances:
        print("No instances remaining after filtering. Exiting.")
        sys.exit(0)

    # Phase 2: Ingestion Engine
    print("\n[Phase 2] Starting Ingestion Engine...")
    
    # Configure VLM
    use_mock = questionary.confirm("Use Mock VLM? (Avoids OpenAI API costs)", default=False).ask()
    
    # Test Run Option
    is_test_run = questionary.confirm("Is this a test run? (Limits images to 3 per instance)").ask()
    max_images = 3 if is_test_run else None
    
    from ingestion_engine import IngestionEngine
    engine = IngestionEngine(mock_vlm=use_mock)
    
    # 2.1 Codebase Ingestion
    # Identify unique repo/versions
    unique_targets = set((inst['repo'], inst['version']) for inst in final_instances)
    
    print(f"\n--- Phase 2.1: Ingesting {len(unique_targets)} Codebases ---")
    for repo, version in unique_targets:
        engine.ingest_codebase(repo, version)
        
    # 2.2 Issue Ingestion
    print(f"\n--- Phase 2.2: Ingesting {len(final_instances)} Issues ---")
    engine.ingest_issues(final_instances)
    
    # 2.3 Visual Ingestion
    print(f"\n--- Phase 2.3: Ingesting Visual Assets ---")
    engine.ingest_visuals(final_instances, max_images=max_images)
    
    print("\n[Phase 2] Ingestion Complete!")
    
    # Phase 3: Experiment Runner
    print("\n[Phase 3] Starting Offline Experiments...")
    
    # Create timestamped run directory
    import time
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    run_dir = pathlib.Path("results") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_dir = run_dir / "swebench_predictions"
    
    from offline_experiment_runner import OfflineExperimentRunner
    runner = OfflineExperimentRunner(output_dir=str(predictions_dir))
    runner.run(final_instances, mock_llm=use_mock)
    
    print("\n[Phase 3] Experiments Completed!")
    
    # Phase 4: Evaluation
    run_eval = questionary.confirm("Run SWE-bench Evaluation now? (Requires Docker)").ask()
    
    if run_eval:
        print("\n[Phase 4] Starting Evaluation...")
        import subprocess
        
        results_dir = predictions_dir
        eval_dir = run_dir / "swebench_evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        for pred_file in results_dir.glob("*_predictions.json"):
            exp_name = pred_file.stem.replace("_predictions", "")
            print(f"\nEvaluating {exp_name}...")
            
            cmd = [
                "python", "-m", "swebench.harness.run_evaluation",
                "--predictions_path", str(pred_file),
                "--dataset_name", "princeton-nlp/SWE-bench_Multimodal",
                "--split", split, # Use selected split
                "--run_id", f"eval_{exp_name}",
                "--report_dir", str(eval_dir)
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Evaluation failed for {exp_name}: {e}")
                
        print("\n[Phase 4] Evaluation Complete. Check results/swebench_evaluation.")
        
        # Cleanup: Move logs and reports if they ended up in root
        import shutil
        
        # Move logs
        if os.path.exists("logs"):
            logs_dest = run_dir / "logs"
            logs_dest.parent.mkdir(parents=True, exist_ok=True)
            if logs_dest.exists():
                shutil.rmtree(logs_dest)
            shutil.move("logs", logs_dest)
            print(f"Moved logs to {logs_dest}")

        # Move any JSON reports from root to eval dir
        for json_file in pathlib.Path(".").glob("*.json"):
            if ".eval_" in json_file.name and json_file.name not in ["package.json", "tsconfig.json"]: # Avoid moving project files
                dest = eval_dir / json_file.name
                shutil.move(str(json_file), str(dest))
                print(f"Moved {json_file.name} to {eval_dir}")
    else:
        print("\nSkipping evaluation. You can run it later using ./evaluate_all.sh")

if __name__ == "__main__":
    main()
