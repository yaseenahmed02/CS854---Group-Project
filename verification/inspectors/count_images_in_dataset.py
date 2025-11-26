from datasets import load_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import ast

def has_images(instance):
    images_raw = instance.get('image_assets')
    if not images_raw:
        return False
        
    images = []
    if isinstance(images_raw, str):
        try:
            # Try JSON first
            data = json.loads(images_raw)
            if isinstance(data, dict):
                for key, urls in data.items():
                    if isinstance(urls, list):
                        images.extend(urls)
        except:
            try:
                # Try literal eval
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
        
    return len(images) > 0

def count_images(split):
    print(f"Loading {split} split...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
    except Exception as e:
        print(f"Could not load {split}: {e}")
        return 0, 0

    total = len(dataset)
    with_images = 0
    
    for instance in dataset:
        if has_images(instance):
            with_images += 1
            
    return total, with_images

def main():
    splits = ['dev', 'test']
    grand_total = 0
    grand_with_images = 0
    
    for split in splits:
        total, with_img = count_images(split)
        print(f"Split: {split}")
        print(f"  Total Instances: {total}")
        print(f"  With Images: {with_img}")
        print(f"  Percentage: {with_img/total*100:.2f}%" if total > 0 else "  Percentage: 0%")
        print("-" * 20)
        
        grand_total += total
        grand_with_images += with_img
        
    print("Overall:")
    print(f"  Total Instances: {grand_total}")
    print(f"  With Images: {grand_with_images}")
    print(f"  Percentage: {grand_with_images/grand_total*100:.2f}%" if grand_total > 0 else "  Percentage: 0%")

if __name__ == "__main__":
    main()
