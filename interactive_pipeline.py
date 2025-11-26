import argparse
import sys
import os
import subprocess
from datasets import load_dataset
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.getcwd())

def get_available_repos(split: str) -> Dict[str, int]:
    """
    Scan the dataset split and return a dictionary of unique repos and their instance counts.
    """
    print(f"Scanning dataset ({split}) for repositories...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
        repos = {}
        for instance in dataset:
            repo = instance['repo']
            repos[repo] = repos.get(repo, 0) + 1
        return repos
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

def get_repo_versions(repo: str, split: str) -> Dict[str, int]:
    """
    Identify all unique versions for a given repository in the dataset split and their instance counts.
    """
    print(f"Scanning dataset ({split}) for versions of {repo}...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
        versions = {}
        for instance in dataset:
            if instance['repo'] == repo:
                ver = instance['version']
                versions[ver] = versions.get(ver, 0) + 1
        return versions
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

def prompt_selection(options: List[str], prompt_text: str, allow_multiple: bool = False) -> Any:
    """
    Helper to prompt user for selection from a list.
    """
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options):
        print(f"[{i+1}] {opt}")
    
    while True:
        try:
            choice = input(f"Select option (1-{len(options)}): ").strip()
            if not choice:
                continue
            
            if allow_multiple:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                if all(0 <= idx < len(options) for idx in indices):
                    return [options[idx] for idx in indices]
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def prompt_input(prompt_text: str, default: str = None, value_type: type = str) -> Any:
    """
    Helper to prompt user for input with default value.
    """
    while True:
        prompt = f"{prompt_text}"
        if default is not None:
            prompt += f" [{default}]"
        prompt += ": "
        
        user_input = input(prompt).strip()
        
        if not user_input and default is not None:
            return default
            
        if not user_input and default is None:
            continue
            
        try:
            if value_type == bool:
                return user_input.lower() in ('y', 'yes', 'true', '1')
            return value_type(user_input)
        except ValueError:
            print(f"Invalid input. Expected {value_type.__name__}.")

def main():
    print("==================================================")
    print("      SWE-bench Multimodal Interactive Wizard     ")
    print("==================================================\n")

    # 1. Select Split
    splits = ["dev", "test"]
    split = prompt_selection(splits, "Select Dataset Split:")
    
    # 2. Select Repo
    repos_dict = get_available_repos(split)
    if not repos_dict:
        print("No repositories found. Exiting.")
        return
        
    # Sort repos by name
    sorted_repos = sorted(repos_dict.keys())
    repo_options = [f"{r} ({repos_dict[r]} instances)" for r in sorted_repos]
    
    # Map selection back to repo name
    selected_option = prompt_selection(repo_options, "Select Repository:")
    repo = sorted_repos[repo_options.index(selected_option)]
    
    # 3. Select Version
    versions_dict = get_repo_versions(repo, split)
    if not versions_dict:
        print(f"No versions found for {repo}. Exiting.")
        return
        
    sorted_versions = sorted(versions_dict.keys())
    version_options = [f"{v} ({versions_dict[v]} instances)" for v in sorted_versions]
    version_options.append("All Versions")
    
    selected_ver_option = prompt_selection(version_options, "Select Version:")
    
    version_arg = None
    if selected_ver_option != "All Versions":
        version_arg = sorted_versions[version_options.index(selected_ver_option)]

    # 4. Instance Limit
    limit = prompt_input("Limit number of instances (0 or empty for all)", default=0, value_type=int)
    if limit == 0:
        limit = None

    # 5. Retrieval Token Limit
    total_token_limit = prompt_input("Retrieval Token Limit", default=13000, value_type=int)

    # 6. LLM Configuration
    llm_providers = ["openai", "vllm", "mock"]
    llm_provider = prompt_selection(llm_providers, "Select LLM Provider:")
    
    llm_model = "gpt-4o-2024-08-06"
    if llm_provider == "openai":
        llm_model = prompt_input("LLM Model Name", default="gpt-4o-2024-08-06")
    elif llm_provider == "vllm":
        llm_model = prompt_input("LLM Model Name (vLLM)", default="meta-llama/Meta-Llama-3-8B-Instruct")
    
    max_output_tokens = prompt_input("Max Output Tokens", default=16384, value_type=int)

    # 7. VLM Configuration
    use_same_model = prompt_input("Use same model for VLM?", default="y", value_type=bool)
    
    vlm_model = llm_model
    if not use_same_model:
        vlm_model = prompt_input("VLM Model Name", default="gpt-4o-2024-08-06")

    # Construct Command
    cmd = [
        "python3", "run_pipeline.py",
        "--repo", repo,
        "--split", split,
        "--llm_provider", llm_provider,
        "--llm_model", llm_model,
        "--vlm_model", vlm_model,
        "--total_token_limit", str(total_token_limit)
    ]
    
    if version_arg:
        cmd.extend(["--version", version_arg])
        
    if limit:
        cmd.extend(["--limit", str(limit)])
        
    if llm_provider == "mock":
        cmd.append("--mock-llm") # Legacy flag support if needed, but provider arg handles it
        
    # Note: run_pipeline.py doesn't currently accept max_output_tokens as an arg, 
    # it's hardcoded/defaulted in pipeline.py. 
    # But wait, I should probably add it to run_pipeline.py to make it fully configurable via this wizard.
    # For now, I will add it to the command and update run_pipeline.py to accept it.
    cmd.extend(["--max_output_tokens", str(max_output_tokens)])

    print("\n" + "="*50)
    print("Constructed Command:")
    print(" ".join(cmd))
    print("="*50 + "\n")
    
    confirm = prompt_input("Run this command?", default="y", value_type=bool)
    if confirm:
        subprocess.run(cmd)
    else:
        print("Aborted.")

if __name__ == "__main__":
    main()
