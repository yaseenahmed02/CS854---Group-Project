import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.getcwd())

from rag.pipeline import RAGPipeline

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Mock Retriever
        self.mock_retriever = MagicMock()
        self.mock_retriever.retrieve.return_value = {
            'retrieved_documents': [
                {'text': 'def foo(): pass', 'metadata': {'rel_path': 'foo.py'}},
                {'text': 'class Bar: pass', 'metadata': {'rel_path': 'bar.py'}}
            ]
        }
        
        # Initialize Pipeline with mock LLM
        self.pipeline = RAGPipeline(
            retriever=self.mock_retriever,
            llm_provider="mock",
            top_k=2
        )

    def test_metrics_calculation(self):
        query = "Fix the bug in foo"
        
        # Run Query
        result = self.pipeline.query(
            query=query,
            mode="code_gen",
            visual_input_mode="vlm_desc",
            vlm_descs=["A screenshot of the error"]
        )
        
        metrics = result['metrics']
        
        print("\n=== Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
        # Assertions
        self.assertIn('prompt_template_tokens', metrics)
        self.assertIn('issue_text_tokens', metrics)
        self.assertIn('vlm_tokens', metrics)
        self.assertIn('retrieved_tokens', metrics)
        self.assertIn('total_input_prompt_tokens', metrics)
        
        # Check values are calculated (not 0 unless empty)
        self.assertGreater(metrics['issue_text_tokens'], 0)
        self.assertGreater(metrics['vlm_tokens'], 0)
        self.assertGreater(metrics['retrieved_tokens'], 0)
        
        # Check Sum Property (Allowing for small discrepancy if any, but logic should hold)
        # total = template + issue + vlm + retrieved
        calculated_sum = (
            metrics['prompt_template_tokens'] + 
            metrics['issue_text_tokens'] + 
            metrics['vlm_tokens'] + 
            metrics['retrieved_tokens']
        )
        
        self.assertEqual(metrics['total_input_prompt_tokens'], calculated_sum)

if __name__ == '__main__':
    unittest.main()
