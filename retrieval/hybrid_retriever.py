"""
Hybrid Retriever
Combines BM25 sparse retrieval with dense vector retrieval.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from embeddings.embed import EmbeddingGenerator
from utils.timer import Timer


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (sparse) and dense vectors.

    Scoring: final_score = alpha * bm25_score + (1 - alpha) * vector_score
    """

    def __init__(self,
                 embeddings_dir: str,
                 chunks_file: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 alpha: float = 0.5):
        """
        Initialize hybrid retriever.

        Args:
            embeddings_dir: Directory containing embeddings and metadata
            chunks_file: Path to chunks JSON file
            model_name: SentenceTransformer model name
            alpha: Weight for BM25 scores (1-alpha for vector scores)
                   0.5 = equal weight, 0.0 = vector only, 1.0 = BM25 only
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.chunks_file = Path(chunks_file)
        self.alpha = alpha

        # Load embedding generator for query encoding
        self.embedding_generator = EmbeddingGenerator(text_model=model_name)

        # Load embeddings and metadata
        self.embeddings, self.metadata = self._load_embeddings()

        # Load chunks
        self.chunks = self._load_chunks()

        # Build BM25 index
        self.bm25_index = self._build_bm25_index()

        print(f"Hybrid retriever initialized with {len(self.embeddings)} documents")
        print(f"BM25-Vector weighting: α={self.alpha} (BM25), {1-self.alpha} (Vector)")

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

    def _build_bm25_index(self) -> BM25Okapi:
        """Build BM25 index from chunks."""
        print("Building BM25 index...")

        # Tokenize all chunks (simple whitespace tokenization)
        tokenized_corpus = [
            self._tokenize(chunk['text'])
            for chunk in self.chunks
        ]

        # Build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)

        print(f"BM25 index built with {len(tokenized_corpus)} documents")
        return bm25

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens (lowercased, alphanumeric)
        """
        # Simple tokenization: lowercase, split on whitespace, keep alphanumeric
        tokens = text.lower().split()
        tokens = [
            ''.join(c for c in token if c.isalnum() or c in ['_', '-'])
            for token in tokens
        ]
        return [t for t in tokens if t]

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve top-k documents using hybrid BM25 + vector scoring.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            Dictionary containing retrieved documents and metrics
        """
        timer = Timer()
        timer.start()

        # 1. BM25 retrieval
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_scores_norm = bm25_scores / max_bm25

        # 2. Vector retrieval
        query_embedding = self.embedding_generator.embed_query(query)
        vector_scores = self.embedding_generator.compute_similarity(
            query_embedding,
            self.embeddings
        )

        # Vector scores are already normalized (cosine similarity in [-1, 1])
        # Shift to [0, 1]
        vector_scores_norm = (vector_scores + 1) / 2

        # 3. Combine scores
        hybrid_scores = (
            self.alpha * bm25_scores_norm +
            (1 - self.alpha) * vector_scores_norm
        )

        # 4. Get top-k results
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        top_scores = hybrid_scores[top_indices]

        # 5. Retrieve chunks
        retrieved_chunks = []
        for idx, score in zip(top_indices, top_scores):
            chunk = self.chunks[idx].copy()
            chunk['retrieval_score'] = float(score)
            chunk['bm25_score'] = float(bm25_scores_norm[idx])
            chunk['vector_score'] = float(vector_scores_norm[idx])
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
            'method': 'hybrid',
            'alpha': self.alpha
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
            'model': self.embedding_generator.text_model_name,
            'alpha': self.alpha,
            'retrieval_method': 'hybrid (BM25 + Vector)'
        }
        return stats


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hybrid_retriever.py <query> [alpha]")
        print("Example: python hybrid_retriever.py 'validateToken function' 0.5")
        sys.exit(1)

    query = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Initialize retriever
    retriever = HybridRetriever(
        embeddings_dir='data/processed/embeddings',
        chunks_file='data/processed/chunks.json',
        alpha=alpha
    )

    # Retrieve documents
    result = retriever.retrieve(query, top_k=5)

    # Print results
    print(f"\nQuery: {result['query']}")
    print(f"Retrieved: {result['num_retrieved']} documents")
    print(f"Time: {result['retrieval_time_ms']:.2f}ms")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Alpha (BM25 weight): {result['alpha']}\n")

    for doc in result['retrieved_documents']:
        print(f"[Rank {doc['rank']}] Hybrid Score: {doc['retrieval_score']:.4f}")
        print(f"  BM25: {doc['bm25_score']:.4f} | Vector: {doc['vector_score']:.4f}")
        print(f"  Document: {doc['document_id']} (chunk {doc['chunk_index']})")
        print(f"  Text: {doc['text'][:150]}...")
        print()
