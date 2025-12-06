import json
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models

def fix_metrics(results_dir: str):
    metrics_file = Path(results_dir) / "swebench_predictions" / "instance_metrics.json"
    if not metrics_file.exists():
        metrics_file = Path(results_dir) / "instance_metrics.json"
        
    if not metrics_file.exists():
        print(f"Metrics file not found in {results_dir}")
        return

    print(f"Fixing metrics in {metrics_file}...")
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
        
    # Connect to Qdrant
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_bench_images")
    collection_name = "swe_bench_images"
    
    print("Listing collections...")
    try:
        collections = client.get_collections().collections
        for c in collections:
            print(f"Collection: {c.name}")
    except Exception as e:
        print(f"Error listing collections: {e}")
        
    updated_count = 0
    
    # Debug: Inspect collection
    print("Inspecting collection...")
    res = client.scroll(collection_name=collection_name, limit=5, with_payload=True)
    print(f"Found {len(res[0])} points in sample.")
    for p in res[0]:
        print(f"Sample Payload: {p.payload}")
    
    for m in metrics:
        # Only fix if 0 and we expect it to be non-zero (multimodal experiments)
        # But checking all is safer.
        instance_id = m.get('instance_id')
        if not instance_id:
            continue
            
        # Fetch images for this instance
        res = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="instance_id", match=models.MatchValue(value=instance_id))]),
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        points, _ = res
        print(f"Debug: instance_id={instance_id}, found {len(points)} images")
        
        total_vlm_time = 0
        for p in points:
            total_vlm_time += p.payload.get('vlm_generation_time_ms', 0)
            
        # Update metric
        # Only update if it's currently 0 or missing, to avoid overwriting if it was somehow correct?
        # Actually, the bug was it was always 0.
        # But wait, 'total_retrieval_time_ms' and 'total_io_time_ms' also depend on this.
        
        old_vlm_time = m.get('vlm_generation_time_ms', 0)
        
        if total_vlm_time > 0:
            m['vlm_generation_time_ms'] = total_vlm_time
            
            # Update derived metrics
            retrieval_time = m.get('retrieval_time_ms', 0)
            m['total_retrieval_time_ms'] = total_vlm_time + retrieval_time
            
            generation_time = m.get('generation_time_ms', 0)
            m['total_io_time_ms'] = m['total_retrieval_time_ms'] + generation_time
            
            updated_count += 1

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Updated {updated_count} records.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_metrics.py <results_dir>")
        sys.exit(1)
    
    fix_metrics(sys.argv[1])
