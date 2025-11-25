"""
Vector Retriever
Dense retrieval using embeddings and cosine similarity.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
from typing import List, Dict, Any, Tuple
from embeddings.embed import EmbeddingGenerator
from utils.timer import Timer


class VectorRetriever:
    """Dense vector-based retrieval using embeddings."""

    def __init__(self,
                 embeddings_dir: str,
                 chunks_file: str,
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize vector retriever.

        Args:
            embeddings_dir: Directory containing embeddings and metadata
            chunks_file: Path to chunks JSON file
            model_name: SentenceTransformer model name (for query encoding)
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.chunks_file = Path(chunks_file)

        # Load embedding generator for query encoding
        self.embedding_generator = EmbeddingGenerator(model_name=model_name, model_type='dense')

        # Load embeddings and metadata
        self.embeddings, self.metadata = self._load_embeddings()

        # Load chunks
        self.chunks = self._load_chunks()

        print(f"Vector retriever initialized with {len(self.embeddings)} embeddings")

    def _load_embeddings(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings and metadata from disk."""
        embeddings, metadata = self.embedding_generator.load_embeddings(
            str(self.embeddings_dir)
        )
        return embeddings, metadata

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return chunks

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve top-k most relevant documents for query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            Dictionary containing retrieved documents and metrics
        """
        timer = Timer()
        timer.start()

        # Encode query
        query_embedding = self.embedding_generator.embed_query(query)

        # Compute similarities
        similarities = self.embedding_generator.compute_similarity(
            query_embedding,
            self.embeddings
        )

        # Get top-k results
        top_indices, top_scores = self.embedding_generator.get_top_k(
            similarities,
            k=top_k
        )

        # Retrieve chunks
        retrieved_chunks = []
        for idx, score in zip(top_indices, top_scores):
            chunk = self.chunks[idx].copy()
            chunk['retrieval_score'] = float(score)
            chunk['rank'] = len(retrieved_chunks) + 1
            retrieved_chunks.append(chunk)

        retrieval_time_ms = timer.stop()

        # Calculate total tokens (approximate: 1 token ≈ 4 characters)
        total_chars = sum(len(chunk['text']) for chunk in retrieved_chunks)
        total_tokens = total_chars // 4

        result = {
            'query': query,
            'retrieved_documents': retrieved_chunks,
            'num_retrieved': len(retrieved_chunks),
            'retrieval_time_ms': retrieval_time_ms,
            'total_tokens': total_tokens,
            'method': 'vector_only'
        }

        return result

    def batch_retrieve(self,
                      queries: List[str],
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of documents per query

        Returns:
            List of retrieval result dictionaries
        """
        results = []
        for query in queries:
            result = self.retrieve(query, top_k)
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retriever statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_embeddings': len(self.embeddings),
            'embedding_dimension': self.embeddings.shape[1],
            'total_chunks': len(self.chunks),
            'model': self.embedding_generator.model_name
        }
        return stats


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vector_retriever.py <query>")
        print("Example: python vector_retriever.py 'How does authentication work?'")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])

    # Initialize retriever
    retriever = VectorRetriever(
        embeddings_dir='data/processed/embeddings',
        chunks_file='data/processed/chunks.json'
    )

    # Retrieve documents
    result = retriever.retrieve(query, top_k=5)

    # Print results
    print(f"\nQuery: {result['query']}")
    print(f"Retrieved: {result['num_retrieved']} documents")
    print(f"Time: {result['retrieval_time_ms']:.2f}ms")
    print(f"Total tokens: {result['total_tokens']}\n")

    for doc in result['retrieved_documents']:
        print(f"[Rank {doc['rank']}] Score: {doc['retrieval_score']:.4f}")
        print(f"Document: {doc['document_id']} (chunk {doc['chunk_index']})")
        print(f"Text preview: {doc['text'][:200]}...")
        print()
