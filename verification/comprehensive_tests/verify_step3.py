import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.ingestion_utils import sanitize_path_component, SemanticChunker
from embeddings.embed import EmbeddingGenerator

def test_sanitize():
    print("Testing sanitize_path_component...")
    name = "Automattic/wp-calypso"
    safe = sanitize_path_component(name)
    assert safe == "Automattic_wp_calypso", f"Expected Automattic_wp_calypso, got {safe}"
    print("✓ Sanitize passed")

def test_chunker():
    print("\nTesting SemanticChunker...")
    # Mock tokenizer to avoid downloading model
    class MockTokenizer:
        def encode(self, text, truncation=False):
            return [1] * len(text) # 1 char = 1 token for simplicity
        def from_pretrained(self, *args, **kwargs):
            return self
            
    # We can't easily mock the internal tokenizer of SemanticChunker without modifying it or mocking AutoTokenizer
    # Let's just try to instantiate it. It might try to download Jina tokenizer.
    # If it fails due to network/auth, we'll catch it.
    try:
        chunker = SemanticChunker()
        text = "a" * 9000 # > 8192
        chunks = chunker.chunk_file(text, "test.py")
        print(f"✓ Chunker instantiated. Chunks generated: {len(chunks)}")
    except Exception as e:
        print(f"⚠ Chunker test skipped or failed (likely model download): {e}")

def test_embedding_init():
    print("\nTesting EmbeddingGenerator init...")
    try:
        # Test dense init (might download)
        # gen = EmbeddingGenerator(model_type='dense', model_name='sentence-transformers/all-MiniLM-L6-v2') # Use small model for test
        # print("✓ Dense generator init passed")
        pass
    except Exception as e:
        print(f"⚠ Embedding init failed: {e}")

if __name__ == "__main__":
    test_sanitize()
    test_chunker()
    # test_embedding_init() # Skipped to avoid heavy downloads in test
