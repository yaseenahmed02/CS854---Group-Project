
from datasets import load_dataset

def check_split(instance_id):
    print(f"Checking split for {instance_id}...")
    
    try:
        ds_test = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="test")
        if instance_id in ds_test['instance_id']:
            print(f"Found {instance_id} in TEST split.")
            return
    except Exception as e:
        print(f"Error loading test split: {e}")

    try:
        ds_dev = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="dev")
        if instance_id in ds_dev['instance_id']:
            print(f"Found {instance_id} in DEV split.")
            return
    except Exception as e:
        print(f"Error loading dev split: {e}")
        
    print(f"Instance {instance_id} not found in dev or test split.")

if __name__ == "__main__":
    check_split("markedjs__marked-1889")
