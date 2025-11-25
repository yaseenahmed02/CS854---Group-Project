from datasets import load_dataset

def inspect_grommet_issue():
    print("Loading SWE-bench Multimodal dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="test")
    
    # Get the first instance (which we know is grommet from previous logs)
    instance = dataset[0]
    
    print(f"\nInstance ID: {instance['instance_id']}")
    print(f"Repo: {instance['repo']}")
    print(f"Version: {instance.get('version') or '<EMPTY>'}")
    print(f"Base Commit: {instance.get('base_commit')}")
    print(f"Available Keys: {list(instance.keys())}")
    
    print("\n--- Problem Statement (First 500 chars) ---")
    print(instance['problem_statement'][:500] + "...")
    
    print("\n--- Image Assets ---")
    images_raw = instance.get('image_assets')
    
    if images_raw:
        import json
        import ast
        
        images = []
        if isinstance(images_raw, str):
            try:
                # Try JSON first
                data = json.loads(images_raw)
                # It seems the structure is {"problem_statement": ["url"]}
                if isinstance(data, dict):
                    for key, urls in data.items():
                        if isinstance(urls, list):
                            images.extend(urls)
            except:
                try:
                    # Try literal eval if JSON fails (single quotes etc)
                    data = ast.literal_eval(images_raw)
                    if isinstance(data, dict):
                        for key, urls in data.items():
                            if isinstance(urls, list):
                                images.extend(urls)
                except:
                    print(f"Could not parse image_assets string: {images_raw}")
        elif isinstance(images_raw, dict):
             for key, urls in images_raw.items():
                if isinstance(urls, list):
                    images.extend(urls)
        elif isinstance(images_raw, list):
            images = images_raw
            
        print(f"Found {len(images)} image(s):")
        for img in images:
            print(f"- {img}")
    else:
        print("No image_assets found in metadata.")

if __name__ == "__main__":
    inspect_grommet_issue()
