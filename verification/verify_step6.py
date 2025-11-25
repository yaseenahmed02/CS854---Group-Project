import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.pipeline import RAGPipeline

def test_code_gen():
    print("Testing RAGPipeline code_gen mode...")
    
    # Create a dummy file to patch
    dummy_file = "dummy_test_file.py"
    original_content = "def hello():\n    print('Hello World')\n"
    with open(dummy_file, "w") as f:
        f.write(original_content)
        
    try:
        # Patch HybridRetriever to avoid loading files
        with patch('rag.pipeline.HybridRetriever') as MockRetriever:
            # Setup mock instance
            mock_retriever_instance = MockRetriever.return_value
            mock_retriever_instance.retrieve.return_value = {
                'retrieved_documents': [{
                    'text': original_content,
                    'path': os.path.abspath(dummy_file),
                    'metadata': {'path': os.path.abspath(dummy_file)}
                }]
            }
            
            # Initialize pipeline
            pipeline = RAGPipeline(retriever_type='hybrid', vllm_url='http://mock')
            
            # Ensure pipeline uses our mock (it should because we patched the class used in __init__)
            # But RAGPipeline assigns self.retriever = HybridRetriever(...)
            # So self.retriever IS mock_retriever_instance
        
        # Mock vLLM response (the "fixed" file)
        fixed_content = "def hello():\n    print('Hello Universe')\n"
        pipeline._call_vllm = MagicMock(return_value={
            'text': f"Here is the fix:\n```python\n{fixed_content}```",
            'tokens_generated': 10
        })
        
        # Execute query
        result = pipeline.query("Fix the greeting", mode="code_gen")
        
        print("\n=== Result Answer (Diff) ===")
        print(result['answer'])
        
        # Verify diff
        expected_diff_snippet = "-    print('Hello World')"
        expected_diff_snippet_2 = "+    print('Hello Universe')"
        
        if expected_diff_snippet in result['answer'] and expected_diff_snippet_2 in result['answer']:
            print("\nSUCCESS: Diff generated correctly.")
        else:
            print("\nFAILURE: Diff does not contain expected changes.")
            
    finally:
        # Cleanup
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

if __name__ == "__main__":
    test_code_gen()
