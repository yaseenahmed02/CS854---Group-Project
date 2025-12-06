
import json
import os

def inspect_metrics():
    path = "results/2025-12-05_03-14/instance_metrics.json"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, 'r') as f:
        metrics = json.load(f)

    print(f"Loaded {len(metrics)} metrics entries.")
    
    for m in metrics:
        exp_id = m.get('experiment_id')
        num_imgs = m.get('num_images')
        vlm_time = m.get('vlm_generation_time_ms')
        vlm_tokens = m.get('vlm_tokens')
        
        print(f"Exp: {exp_id}")
        print(f"  Num Images: {num_imgs}")
        print(f"  VLM Time: {vlm_time}")
        print(f"  VLM Tokens: {vlm_tokens}")
        print("-" * 20)

if __name__ == "__main__":
    inspect_metrics()
