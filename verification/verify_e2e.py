import sys
import os
import shutil
import uuid
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from ingest_code_to_qdrant import ingest_repo
from ingest_images_to_qdrant import ingest_images
from rag.pipeline import RAGPipeline
from retrieval.flexible_retriever import FlexibleRetriever

# Configuration
TEST_REPO_NAME = "test_e2e_repo"
TEST_VERSION = "1.0.0"
TEST_DIR = Path("temp_e2e_test")
QDRANT_CODE_DIR = Path(f"qdrant_data_{TEST_REPO_NAME}_{TEST_VERSION.replace('.', '_')}")
QDRANT_IMAGES_DIR = Path("qdrant_data_swe_images_test")

def log(step: str, message: str):
    print(f"\n[{step.upper()}] {message}")

def setup_test_data():
    log("SETUP", f"Creating temporary directory: {TEST_DIR}")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir()

    # Create dummy code file
    code_file = TEST_DIR / "main.py"
    code_content = """
def hello_world():
    print("Hello from E2E test")

def broken_function():
    # This function has a bug
    x = 1 / 0
"""
    with open(code_file, "w") as f:
        f.write(code_content)
    log("SETUP", f"Created dummy code file: {code_file}")
    print(f"Content:\n{code_content}")

def cleanup():
    log("CLEANUP", "Removing temporary directories...")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"Removed {TEST_DIR}")
    
    if QDRANT_CODE_DIR.exists():
        shutil.rmtree(QDRANT_CODE_DIR)
        print(f"Removed {QDRANT_CODE_DIR}")
        
    if QDRANT_IMAGES_DIR.exists():
        shutil.rmtree(QDRANT_IMAGES_DIR)
        print(f"Removed {QDRANT_IMAGES_DIR}")

def run_e2e_test():
    try:
        setup_test_data()

        # 1. Ingest Code
        log("INGESTION", "Ingesting code into Qdrant...")
        # We need to ensure ingest_repo uses our custom logic or just call it
        # ingest_repo uses global QdrantClient path logic, which matches our QDRANT_CODE_DIR expectation
        ingest_repo(str(TEST_DIR), TEST_REPO_NAME, TEST_VERSION, mode="overwrite")
        
        # Verify Code Ingestion
        client_code = QdrantClient(path=str(QDRANT_CODE_DIR))
        collection_name = f"{TEST_REPO_NAME}_{TEST_VERSION.replace('.', '_')}"
        count = client_code.count(collection_name).count
        log("VERIFICATION", f"Code collection '{collection_name}' has {count} vectors.")
        assert count > 0, "Code ingestion failed: No vectors found."

        # 2. Ingest Images (Mocked)
        log("INGESTION", "Ingesting images (mocked) into Qdrant...")
        # We'll manually insert a point to simulate image ingestion or use the script with a mock
        # Let's use the script but patch setup_qdrant to use our test dir
        with patch('ingest_images_to_qdrant.setup_qdrant') as mock_setup:
            # Configure mock to return client pointing to our test dir
            client_images = QdrantClient(path=str(QDRANT_IMAGES_DIR))
            # Ensure collection exists
            if not client_images.collection_exists("swe_images"):
                client_images.create_collection(
                    collection_name="swe_images",
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
                )
            mock_setup.return_value = (client_images, "swe_images")
            
            # We also need to mock load_dataset to return a dummy instance
            with patch('ingest_images_to_qdrant.load_dataset') as mock_load:
                mock_load.return_value = [{
                    "instance_id": "test-instance-1",
                    "repo": TEST_REPO_NAME,
                    "version": TEST_VERSION,
                    "problem_statement": "Fix the broken function. See screenshot.",
                    "image_assets": [{"url": "http://example.com/screenshot.png"}]
                }]
                
                # Run ingestion
                ingest_images(limit=1, mock_vlm=True, store_image=False, split="test")
        
        # Verify Image Ingestion
        count_images = client_images.count("swe_images").count
        log("VERIFICATION", f"Image collection 'swe_images' has {count_images} vectors.")
        assert count_images > 0, "Image ingestion failed: No vectors found."

        # 3. Retrieval
        log("RETRIEVAL", "Initializing FlexibleRetriever...")
        retriever = FlexibleRetriever(
            client=client_code,
            collection_name=collection_name,
            swe_images_collection="swe_images",
            images_client=client_images
        )

        query = "Fix the broken function"
        log("RETRIEVAL", f"Executing retrieval for query: '{query}'")
        
        # Test Hybrid Retrieval (Text + Image)
        # We need to mock _fetch_visual_description to return something relevant if we want to test fusion
        # Or we rely on the fact that we ingested a mock VLM description
        
        # Let's force a visual mode that uses the image we ingested
        # The instance_id must match what we ingested: "test-instance-1"
        
        results = retriever.retrieve(
            query, 
            instance_id="test-instance-1",
            strategy=["bm25", "jina"],
            visual_mode="fusion", # This should fetch the VLM description from DB
            top_k=3
        )
        
        log("RETRIEVAL", "Results obtained:")
        for i, res in enumerate(results['retrieved_documents']):
            print(f"  {i+1}. [Score: {res.get('score', 'N/A'):.4f}] {res.get('filepath', 'Unknown')}")
            print(f"     Snippet: {res.get('text', '')[:100]}...")
            
        assert len(results['retrieved_documents']) > 0, "Retrieval failed: No documents returned."
        
        # 4. Generation (RAG Pipeline)
        log("GENERATION", "Initializing RAGPipeline...")
        
        # Mock vLLM
        mock_vllm = MagicMock()
        mock_vllm.generate.return_value = [MagicMock(
            metrics=MagicMock(arrival_time=0, first_token_time=0, finished_time=0),
            outputs=[MagicMock(text="Fixed code\n```python\ndef broken_function():\n    x = 1 / 1\n```", token_ids=[1]*10)]
        )]
        
        # Patch HybridRetriever to avoid loading files during RAGPipeline init
        with patch('rag.pipeline.HybridRetriever') as MockRetriever:
            MockRetriever.return_value = MagicMock()
            pipeline = RAGPipeline(retriever_type='hybrid', vllm=mock_vllm)
            
        pipeline.retriever = retriever # Inject our test retriever
        
        log("GENERATION", "Running pipeline.query...")
        pipeline_result = pipeline.query(
            query, 
            mode="code_gen",
            retrieval_token_limit=1000
        )
        
        log("GENERATION", "Pipeline Result:")
        print(f"  Answer: {pipeline_result['answer']}")
        print(f"  Metrics: {pipeline_result['metrics']}")
        
        # Verify Prompt contains retrieved context
        # We can't easily check the exact prompt string inside pipeline without mocking prompt_builder or inspecting internal state
        # But we can check if 'prompt_tokens' > 0
        assert pipeline_result['metrics']['prompt_tokens'] > 0, "Prompt tokens should be > 0"
        
        print("\n" + "="*80)
        log("SUCCESS", "🎉 End-to-End verification passed! The system is healthy.")
        print("="*80)

    except Exception as e:
        log("FAILURE", f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    run_e2e_test()
