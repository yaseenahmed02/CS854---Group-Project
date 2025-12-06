
from datasets import load_dataset
import json
import ast

def count_images(split="dev"):
    print(f"Loading {split} split...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
    
    total_images = 0
    instances_with_images = 0
    
    for instance in dataset:
        images_raw = instance.get('image_assets')
        images = []
        
        if images_raw:
            if isinstance(images_raw, str):
                try:
                    data = json.loads(images_raw)
                    if isinstance(data, dict):
                        for key, urls in data.items():
                            if isinstance(urls, list):
                                images.extend(urls)
                except:
                    try:
                        data = ast.literal_eval(images_raw)
                        if isinstance(data, dict):
                            for key, urls in data.items():
                                if isinstance(urls, list):
                                    images.extend(urls)
                    except:
                        pass
            elif isinstance(images_raw, dict):
                 for key, urls in images_raw.items():
                    if isinstance(urls, list):
                        images.extend(urls)
            elif isinstance(images_raw, list):
                images = images_raw
        
        if images:
            total_images += len(images)
            instances_with_images += 1
            
    print(f"Total instances in {split}: {len(dataset)}")
    print(f"Instances with images: {instances_with_images}")
    print(f"Total images found: {total_images}")

if __name__ == "__main__":
    count_images("dev")
