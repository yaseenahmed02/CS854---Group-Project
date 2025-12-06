from datasets import load_dataset
from collections import defaultdict

def main():
    print("Loading SWE-bench_Multimodal (dev split)...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="dev")
    
    target_repos = {
        "diegomura/react-pdf",
        "markedjs/marked",
        "processing/p5.js"
    }
    
    counts = defaultdict(int)
    
    print("\nCounting instances per repo/version...")
    for instance in dataset:
        repo = instance["repo"]
        if repo in target_repos:
            version = instance["version"]
            counts[(repo, version)] += 1
            
    print("\n=== Dataset Counts (Dev Split) ===")
    
    # Sort for consistent output
    sorted_keys = sorted(counts.keys())
    
    current_repo = None
    repo_total = 0
    
    for repo, version in sorted_keys:
        if repo != current_repo:
            if current_repo:
                print(f"  Total for {current_repo}: {repo_total}")
                print()
            current_repo = repo
            repo_total = 0
            print(f"Repo: {repo}")
            
        count = counts[(repo, version)]
        repo_total += count
        print(f"  v{version}: {count}")
        
    if current_repo:
        print(f"  Total for {current_repo}: {repo_total}")
        
    total_target = sum(counts.values())
    print(f"\nTotal target instances found: {total_target}")

if __name__ == "__main__":
    main()
