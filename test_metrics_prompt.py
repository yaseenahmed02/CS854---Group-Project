
import sys
import os
from rag.pipeline import RAGPipeline

# Mock retriever
class MockRetriever:
    def retrieve(self, query, top_k=5, **kwargs):
        return {
            'retrieved_documents': [{'text': 'doc1', 'payload': {'text': 'doc1'}}],
            'results': []
        }
    def get_statistics(self):
        return {}

def test_prompt_metrics():
    print("Testing prompt metrics...")
    pipeline = RAGPipeline(
        retriever=MockRetriever(),
        llm_provider="mock"
    )
    
    result = pipeline.query("test query", mode="code_gen")
    metrics = result['metrics']
    
    if 'input_prompt_text' in metrics:
        print("SUCCESS: input_prompt_text found in metrics.")
    
    if 'retrieved_file_paths' in metrics:
        print("SUCCESS: retrieved_file_paths found in metrics.")
        print(f"Retrieved paths: {metrics['retrieved_file_paths']}")
    else:
        print("FAILURE: retrieved_file_paths NOT found in metrics.")

if __name__ == "__main__":
    test_prompt_metrics()
