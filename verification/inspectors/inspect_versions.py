import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset

def inspect_versions():
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="dev")
    versions = set()
    for instance in dataset:
        if instance['repo'] == 'markedjs/marked':
            versions.add(instance['version'])
    print(f"Versions in dataset: {versions}")

if __name__ == "__main__":
    inspect_versions()
