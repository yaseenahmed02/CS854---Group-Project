
from datasets import load_dataset

def check_dataset():
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="dev")
    print(f"Total instances: {len(dataset)}")
    
    found = 0
    for instance in dataset:
        if instance['repo'] == 'markedjs/marked' and instance['version'] == '0.3':
            found += 1
            print(f"Found: {instance['instance_id']}")
            
    print(f"Total markedjs/marked 0.3 instances: {found}")

if __name__ == "__main__":
    check_dataset()
