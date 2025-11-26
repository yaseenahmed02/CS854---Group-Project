from datasets import load_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_split():
    print("Checking splits...")
    splits = ['dev', 'test']
    target_repo = 'markedjs/marked'
    target_version = '2.0'
    target_instance = 'markedjs__marked-1928' # Picking a likely instance ID or just checking count

    for split in splits:
        print(f"Loading {split} split...")
        ds = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
        
        repo_count = 0
        version_count = 0
        instance_found = False
        image_count = 0
        
        for item in ds:
            if item['repo'] == target_repo:
                repo_count += 1
                if item['version'] == target_version:
                    version_count += 1
                    if item['instance_id'] == target_instance:
                        instance_found = True
                    
                    # Check images
                    images = item.get('image_assets')
                    if images:
                        image_count += 1
        
        print(f"Split: {split}")
        print(f"  Repo {target_repo}: {repo_count}")
        print(f"  Version {target_version}: {version_count}")
        print(f"  Instance {target_instance} found: {instance_found}")
        print(f"  Images in version: {image_count}")

if __name__ == "__main__":
    check_split()
