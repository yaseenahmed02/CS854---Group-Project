import sys
import os
import time
from qdrant_client import QdrantClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.flexible_retriever import FlexibleRetriever

def print_status(message, status):
    color = "\033[92m" if status == "PASS" else "\033[91m"
    reset = "\033[0m"
    print(f"{message:.<60} [{color}{status}{reset}]")

def verify_e2e():
    print("\n=== System Health Check (E2E) ===\n")
    all_passed = True
    
    # 1. Check Qdrant Connection
    try:
        # Assuming local Qdrant or path-based
        if os.path.exists("data/qdrant/qdrant_data_swe_images"):
            client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
            collections = client.get_collections()
            client.close()
            print_status("Qdrant Connection (Images)", "PASS")
        else:
            print_status("Qdrant Connection (Images)", "FAIL")
            print("  -> data/qdrant/qdrant_data_swe_images not found")
            all_passed = False
    except Exception as e:
        print_status("Qdrant Connection (Images)", "FAIL")
        print(f"  -> Error: {e}")
        all_passed = False

    # 2. Check swe_images collection
    try:
        if os.path.exists("data/qdrant/qdrant_data_swe_images"):
            # Reuse client if possible or ensure it's closed
            client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
            try:
                count = client.count(collection_name="swe_images").count
                if count > 0:
                    print_status(f"swe_images Collection (Count: {count})", "PASS")
                else:
                    print_status("swe_images Collection (Empty)", "FAIL")
                    all_passed = False
            finally:
                client.close()
    except Exception as e:
        print_status("swe_images Collection", "FAIL")
        print(f"  -> Error: {e}")
        all_passed = False

    # 3. Check for at least one repo collection
    repo_found = False
    try:
        # Look for any qdrant_data_* directory that isn't swe_images
        import glob
        dbs = glob.glob("data/qdrant/qdrant_data_*")
        repo_dbs = [d for d in dbs if "swe_images" not in d]
        
        if repo_dbs:
            print_status(f"Repo Collections Found ({len(repo_dbs)})", "PASS")
            repo_found = True
        else:
            print_status("Repo Collections Found", "FAIL")
            print("  -> No repository Qdrant databases found (qdrant_data_*)")
            all_passed = False
    except Exception as e:
        print_status("Repo Collections Check", "FAIL")
        print(f"  -> Error: {e}")
        all_passed = False

    # 4. Simple Retrieval Test (if repo found)
    if repo_found:
        try:
            # Pick the first one
            db_path = repo_dbs[0]
            # Extract collection name from folder name
            folder_name = os.path.basename(db_path)
            collection_name = folder_name.replace("qdrant_data_", "")
            
            client = QdrantClient(path=db_path)
            images_client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
            
            retriever = FlexibleRetriever(
                client=client,
                collection_name=collection_name,
                swe_images_collection="swe_images",
                images_client=images_client
            )
            
            # Dummy query
            results = retriever.retrieve("test query", strategy=["bm25"], top_k=1)
            if results:
                print_status("Simple Retrieval Test", "PASS")
            else:
                print_status("Simple Retrieval Test (No results)", "FAIL")
                all_passed = False
                
        except Exception as e:
            print_status("Simple Retrieval Test", "FAIL")
            print(f"  -> Error: {e}")
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("\033[92mSYSTEM HEALTH CHECK PASSED\033[0m")
    else:
        print("\033[91mSYSTEM HEALTH CHECK FAILED\033[0m")
    print("="*70 + "\n")

if __name__ == "__main__":
    verify_e2e()
