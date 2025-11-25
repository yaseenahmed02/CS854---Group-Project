import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

# Mock vllm before importing pipeline
sys.modules['vllm'] = MagicMock()
from rag.pipeline import RAGPipeline

class TestTokenLimit(unittest.TestCase):
    def setUp(self):
        # Mock vLLM
        self.mock_vllm = MagicMock()
        self.mock_vllm.generate.return_value = [MagicMock(outputs=[MagicMock(token_ids=[1, 2, 3], text="Fixed code")])]
        self.mock_vllm.generate.return_value[0].metrics.arrival_time = 0
        self.mock_vllm.generate.return_value[0].metrics.first_token_time = 0
        self.mock_vllm.generate.return_value[0].metrics.finished_time = 0

        # Initialize Pipeline with mock vLLM
        # We mock the retriever initialization to avoid loading real embeddings
        with patch('rag.pipeline.HybridRetriever') as MockRetriever:
            self.mock_retriever_instance = MagicMock()
            MockRetriever.return_value = self.mock_retriever_instance
            
            # Use real tiktoken
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            print("Using real tiktoken")
            
            with patch('rag.pipeline.tiktoken.encoding_for_model', return_value=self.tokenizer):
                self.pipeline = RAGPipeline(vllm=self.mock_vllm, retriever_type='hybrid')

    def test_token_limit_enforcement(self):
        # Setup mock retrieved docs
        # Create docs with known token counts
        # "word " is usually 1 token in cl100k_base
        text_100 = "word " * 100
        doc1 = {"text": text_100, "id": "1"} 
        doc2 = {"text": text_100, "id": "2"} 
        doc3 = {"text": text_100, "id": "3"} 
        
        # Verify token counts
        tokens_1 = len(self.tokenizer.encode(text_100))
        print(f"\nDocument 1 tokens: {tokens_1}")
        
        self.mock_retriever_instance.retrieve.return_value = {
            "retrieved_documents": [doc1, doc2, doc3]
        }
        
        # Test with limit that allows only 1 doc
        print(f"Testing limit: {tokens_1 + 10}")
        result = self.pipeline.query("fix bug", mode="code_gen", retrieval_token_limit=tokens_1 + 10)
        
        # Check how many docs were used. 
        # The pipeline doesn't explicitly return "used_docs", but it returns "retrieved_documents".
        # However, the logic we added modifies which docs are used for CONTEXT.
        # It doesn't necessarily filter the 'retrieved_documents' in the result output (which usually reflects what the retriever found).
        # Wait, let's check my implementation.
        
        # In my implementation:
        # selected_docs = []
        # ... logic ...
        # if selected_docs: top_doc = selected_docs[0]
        
        # The implementation I wrote mainly affects `top_doc` selection for the `target_file_content`.
        # It DOES NOT explicitly update `retrieved_docs` in the result dictionary to reflect ONLY the used ones.
        # It uses `retrieved_docs` (the original list) for the result.
        
        # BUT, the prompt construction uses `target_file_content` which comes from `top_doc`.
        # And `top_doc` comes from `selected_docs`.
        
        # Wait, if I have multiple docs, usually RAG puts ALL of them in the prompt.
        # My implementation for `code_gen` mode in `pipeline.py` was:
        # "Naively take the top document as the target file"
        
        # It seems the current `code_gen` mode ONLY uses the TOP document (single file) to rewrite.
        # It does NOT put all retrieved docs into the context as "Reference".
        # It assumes the top doc IS the file to fix.
        
        # So, if I have a token limit, and I filter `selected_docs`, 
        # `top_doc` will be `selected_docs[0]`.
        # If `selected_docs` is empty (limit too small), it falls back to `retrieved_docs[0]`.
        
        # This means my token limit implementation for `code_gen` might be slightly redundant if it only uses ONE doc anyway.
        # UNLESS `code_gen` mode is supposed to use multiple docs?
        
        # Let's re-read `pipeline.py` `code_gen` section.
        # It says:
        # target_file_content = ... top_doc ...
        # user_prompt = ... Target File ... {target_file_content} ...
        
        # It seems it ONLY uses one file.
        # So "token limit" in this context might mean "Skip this document if it's too large"?
        # OR "Find the first document that fits"?
        
        # If the requirement "Implement a token counter/limit parameter for the retrieval step, to create a condition comparable to the original SWE-bench RAG experiment"
        # implies we want to fit AS MANY docs as possible into context...
        # Then the `code_gen` prompt construction needs to be updated to support MULTIPLE docs.
        
        # Currently `pipeline.py` `code_gen` seems designed for "Single File Edit".
        # If I want to support "Context from multiple files", I need to change how the prompt is built.
        
        # However, the user request didn't explicitly ask to change the prompt strategy to multi-file, 
        # but "comparable to the original SWE-bench RAG experiment" implies standard RAG where you stuff context.
        
        # If the current pipeline only uses 1 file, then "token limit" effectively means "Max size of that 1 file".
        
        # Let's verify what I implemented.
        # I implemented:
        # for doc in retrieved_docs:
        #   if current + doc_len <= limit: keep
        #   else: break
        
        # And then:
        # top_doc = selected_docs[0]
        
        # So if the first doc is 20k tokens and limit is 13k.
        # It will NOT be selected.
        # `selected_docs` will be empty (if only 1 doc or all are large).
        # Then it falls back to `retrieved_docs[0]`.
        
        # So effectively, my implementation currently DOES NOTHING if the first doc is too big, because of the fallback.
        # AND it does nothing if the first doc fits, because it only uses the first doc.
        
        # I need to fix this.
        # 1. The fallback should probably not exist if we strictly want to enforce limit.
        # 2. OR, more likely, we want to include MULTIPLE docs in the prompt if they fit.
        
        # Let's assume for `code_gen` in this specific project (which seems to be "Rewrite File"), 
        # maybe we only want the file to be edited.
        # But if we are doing RAG, we might want other files as context.
        
        # Let's look at `pipeline.py` again.
        # `mode="qa"` uses `self.prompt_builder.build_prompt(query, retrieved_docs)`.
        # `mode="code_gen"` constructs a custom prompt with `Target File`.
        
        # If I want to enforce token limit for `qa` mode, I need to pass it to `prompt_builder` or filter `retrieved_docs` before passing.
        # My change in `pipeline.py` was inside `if mode == "code_gen":`.
        # I MISSED `qa` mode!
        
        # And for `code_gen`, if it only uses 1 file, the limit is less useful unless we change it to use multiple.
        
        # I should probably:
        # 1. Apply the filtering to `retrieved_docs` BEFORE the mode check, so it applies to BOTH modes.
        # 2. Update `code_gen` to potentially use multiple docs? Or at least respect the filtering.
        
        # Let's adjust the test to verify this behavior (or lack thereof) and then fix the code.
        # Test with limit that allows only 1 doc
        # 100 tokens per doc. Limit 150. Should get 1 doc.
        result = self.pipeline.query("fix bug", mode="code_gen", retrieval_token_limit=150)
        
        # Check that we only got 1 document back in the result
        self.assertEqual(len(result['retrieved_documents']), 1)
        self.assertEqual(result['retrieved_documents'][0]['id'], "1")
        
        # Test with limit that allows 2 docs
        # Limit 250. Should get 2 docs.
        result = self.pipeline.query("fix bug", mode="code_gen", retrieval_token_limit=250)
        self.assertEqual(len(result['retrieved_documents']), 2)
        self.assertEqual(result['retrieved_documents'][0]['id'], "1")
        self.assertEqual(result['retrieved_documents'][1]['id'], "2")

        # Test with limit that allows 0 docs (too small)
        # Limit 50. Should get 0 docs.
        result = self.pipeline.query("fix bug", mode="code_gen", retrieval_token_limit=50)
        self.assertEqual(len(result['retrieved_documents']), 0)


    def test_dummy(self):
        pass

if __name__ == '__main__':
    unittest.main()
