
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inspect_dataset():
    print("Loading SWE-bench Multimodal dataset (dev split)...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="dev")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Searching for markedjs/marked instances...")
    marked_instances = []
    for instance in dataset:
        if "marked" in instance['repo']:
            marked_instances.append(instance)
            
    print(f"Found {len(marked_instances)} instances.")
    for inst in marked_instances:
        print(f"ID: {inst['instance_id']}")
        print(f"Repo: {inst['repo']}")
        print(f"Version: {inst['version']}")
        print("-" * 20)

if __name__ == "__main__":
    inspect_dataset()
