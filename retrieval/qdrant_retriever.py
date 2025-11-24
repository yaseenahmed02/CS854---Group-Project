"""
Qdrant Retriever
Vector retrieval using Qdrant persistent storage.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from embeddings.embed import EmbeddingGenerator
from utils.timer import Timer


class QdrantRetriever:
    """
    Vector retrieval using Qdrant.
    """

    def __init__(self,
                 qdrant_path: str,
                 chunks_file: str,
                 collection_name: str = "documents",
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Qdrant retriever.

        Args:
            qdrant_path: Path to Qdrant database directory
            chunks_file: Path to chunks JSON file
            collection_name: Name of Qdrant collection
            model_name: SentenceTransformer model name (must match embeddings)
        """
        self.qdrant_path = Path(qdrant_path)
        self.chunks_file = Path(chunks_file)
        self.collection_name = collection_name

        # Initialize Qdrant client (loads existing database)
        print(f"Loading Qdrant database from {self.qdrant_path}")
        self.client = QdrantClient(path=str(self.qdrant_path))

        # Load embedding generator for query encoding
        self.embedding_generator = EmbeddingGenerator(text_model=model_name)

        # Load chunks for reference
        self.chunks = self._load_chunks()

        # Verify collection exists
        self._verify_collection()

        print(f"Qdrant retriever initialized")
        print(f"Collection: {self.collection_name}")
        print(f"Total documents: {len(self.chunks)}")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return chunks

    def _verify_collection(self):
        """Verify that the collection exists in Qdrant."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            raise ValueError(
                f"Collection '{self.collection_name}' not found in Qdrant database. "
                f"Available collections: {collection_names}"
            )
        
        # Get collection info
        collection_info = self.client.get_collection(self.collection_name)
        print(f"Collection info: {collection_info.points_count} points, "
              f"vector size: {collection_info.config.params.vectors.size}")

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve top-k documents using vector similarity.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            Dictionary containing retrieved documents and metrics
        """
        timer = Timer()
        timer.start()

        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)

        # Search in Qdrant
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        )

        # Process results    
        retrieved_chunks = []
        print(f"Type of search_results: {type(search_results)}")
        for result in search_results:
            result_data = result[1]
            for (rank, hit) in enumerate(result_data, start=1):
                chunk_index = hit.payload.get('chunk_index')
                chunk = self.chunks[chunk_index].copy()
                chunk["retrieval_score"] = hit.score
                chunk["rank"] = rank
                retrieved_chunks.append(chunk)

        retrieval_time_ms = timer.stop()

        # Calculate total tokens (approximate: 1 token ≈ 4 characters)
        total_chars = sum(len(chunk.get('text', '')) for chunk in retrieved_chunks)
        total_tokens = total_chars // 4

        result = {
            'query': query,
            'retrieved_documents': retrieved_chunks,
            'num_retrieved': len(retrieved_chunks),
            'retrieval_time_ms': retrieval_time_ms,
            'total_tokens': total_tokens,
            'method': 'qdrant_vector'
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
        collection_info = self.client.get_collection(self.collection_name)
        
        stats = {
            'total_points': collection_info.points_count,
            'embedding_dimension': collection_info.config.params.vectors.size,
            'total_chunks': len(self.chunks),
            'model': self.embedding_generator.text_model_name,
            'collection_name': self.collection_name,
            'retrieval_method': 'qdrant_vector',
            'qdrant_path': str(self.qdrant_path)
        }
        return stats


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python qdrant_retriever.py <query> [top_k]")
        print("Example: python qdrant_retriever.py 'validateToken function' 5")
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # Initialize retriever
    retriever = QdrantRetriever(
        qdrant_path='data/processed/embeddings/qdrant_db',
        chunks_file='data/processed/chunks.json',
        collection_name='embeddings'
    )

    # Retrieve documents
    result = retriever.retrieve(query, top_k=top_k)

    # Print results
    print(f"\nQuery: {result['query']}")
    print(f"Retrieved: {result['num_retrieved']} documents")
    print(f"Time: {result['retrieval_time_ms']:.2f}ms")
    print(f"Total tokens: {result['total_tokens']}\n")

    for doc in result['retrieved_documents']:
        print(f"[Rank {doc['rank']}] Score: {doc['retrieval_score']:.4f}")
        print(f"  Document: {doc['document_id']} (chunk {doc['chunk_index']})")
        print(f"  Text: {doc.get('text', '')[:150]}...")
        print()