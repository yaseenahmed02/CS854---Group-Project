import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.pipeline import RAGPipeline

class TestMultimodalPrompt(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_retriever.retrieve.return_value = {
            'retrieved_documents': [{'text': 'def foo(): pass', 'metadata': {'path': 'foo.py'}}]
        }
        
        # Initialize pipeline with mock LLM provider to avoid API calls
        # But we want to test _generate_openai logic or at least the prompt construction before it calls _generate
        # The prompt construction happens inside query.
        # We can mock _generate_openai and check what it was called with.
        
        self.pipeline = RAGPipeline(
            retriever=self.mock_retriever,
            llm_provider="openai", # We want to trigger the multimodal logic which is generic but _generate_openai handles list
            llm_model="gpt-4o"
        )
        
        # Mock the client to avoid real init
        self.pipeline.openai_client = MagicMock()
        
        # Mock _generate_openai to capture prompt
        self.pipeline._generate_openai = MagicMock(return_value={'text': 'Fixed', 'tokens_generated': 10})

    def test_text_only_mode(self):
        """Test legacy text-only mode (vlm_desc)."""
        vlm_context = [{'description': 'A bug', 'image_url': 'http://example.com/img.png'}]
        
        # Must explicitly set visual_input_mode="vlm_desc" now that default is "vlm_desc_url_image_file"
        self.pipeline.query(
            query="Fix bug",
            mode="code_gen",
            vlm_context=vlm_context,
            visual_input_mode="vlm_desc"
        )
        
        # Check call args
        args, _ = self.pipeline._generate_openai.call_args
        prompt = args[0]
        
        self.assertIsInstance(prompt, str)
        self.assertIn("Visual Context (from images):", prompt)
        self.assertIn("Image 1: A bug", prompt)
        self.assertNotIn("http://example.com/img.png", prompt) # URL should not be in text prompt

    def test_url_mode(self):
        """Test vlm_desc_url mode."""
        vlm_context = [{'description': 'A bug', 'image_url': 'http://example.com/img.png'}]
        
        self.pipeline.query(
            query="Fix bug",
            mode="code_gen",
            vlm_context=vlm_context,
            visual_input_mode="vlm_desc_url"
        )
        
        args, _ = self.pipeline._generate_openai.call_args
        prompt = args[0]
        
        self.assertIsInstance(prompt, list)
        self.assertEqual(len(prompt), 2) # Text + 1 Image
        
        # Check Text Part
        self.assertEqual(prompt[0]['type'], 'text')
        self.assertIn("Visual Context (from images):", prompt[0]['text']) # Desc should be included
        
        # Check Image Part
        self.assertEqual(prompt[1]['type'], 'image_url')
        self.assertEqual(prompt[1]['image_url']['url'], 'http://example.com/img.png')

    def test_file_mode(self):
        """Test image_file mode."""
        vlm_context = [{'description': 'A bug', 'image_base64': 'SGVsbG8='}]
        
        self.pipeline.query(
            query="Fix bug",
            mode="code_gen",
            vlm_context=vlm_context,
            visual_input_mode="image_file"
        )
        
        args, _ = self.pipeline._generate_openai.call_args
        prompt = args[0]
        
        self.assertIsInstance(prompt, list)
        
        # Check Text Part
        self.assertEqual(prompt[0]['type'], 'text')
        self.assertNotIn("Visual Context (from images):", prompt[0]['text']) # Desc NOT included in image_file mode?
        # Wait, my logic was: if "vlm_desc" in visual_input_mode...
        # "image_file" does NOT contain "vlm_desc". So desc should be absent.
        
        # Check Image Part
        self.assertEqual(prompt[1]['type'], 'image_url')
        self.assertEqual(prompt[1]['image_url']['url'], 'data:image/png;base64,SGVsbG8=')

    def test_all_mode(self):
        """Test vlm_desc_url_image_file mode."""
        vlm_context = [{
            'description': 'A bug', 
            'image_url': 'http://example.com/img.png',
            'image_base64': 'SGVsbG8='
        }]
        
        self.pipeline.query(
            query="Fix bug",
            mode="code_gen",
            vlm_context=vlm_context,
            visual_input_mode="vlm_desc_url_image_file"
        )
        
        args, _ = self.pipeline._generate_openai.call_args
        prompt = args[0]
        
        self.assertIsInstance(prompt, list)
        self.assertEqual(len(prompt), 3) # Text + URL + File
        
        # Check Text Part
        self.assertIn("Visual Context (from images):", prompt[0]['text'])
        
        # Check Image Parts (Order depends on implementation loop)
        # Implementation:
        # if "url" ... append
        # if "image_file" ... append
        
        self.assertEqual(prompt[1]['type'], 'image_url')
        self.assertEqual(prompt[1]['image_url']['url'], 'http://example.com/img.png')
        
        self.assertEqual(prompt[2]['type'], 'image_url')
        self.assertEqual(prompt[2]['image_url']['url'], 'data:image/png;base64,SGVsbG8=')

    def test_default_mode(self):
        """Test that default mode is vlm_desc_url_image_file."""
        vlm_context = [{
            'description': 'A bug', 
            'image_url': 'http://example.com/img.png',
            'image_base64': 'SGVsbG8='
        }]
        
        # Do NOT specify visual_input_mode
        self.pipeline.query(
            query="Fix bug",
            mode="code_gen",
            vlm_context=vlm_context
        )
        
        args, _ = self.pipeline._generate_openai.call_args
        prompt = args[0]
        
        self.assertIsInstance(prompt, list)
        self.assertEqual(len(prompt), 3) # Should be Text + URL + File by default

if __name__ == '__main__':
    unittest.main()
