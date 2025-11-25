"""
Embeddings Module
Generates embeddings for text, code, and images using SentenceTransformers and CLIP.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from utils.timer import Timer, measure_time


class EmbeddingGenerator:
    """Generate embeddings for multimodal documents."""

    def __init__(self,
                 text_model: str = 'all-MiniLM-L6-v2',
                 device: str = 'cpu'):
        """
        Initialize embedding generator.

        Args:
            text_model: SentenceTransformer model name for text/code
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        self.text_model_name = text_model

        print(f"Loading text embedding model: {text_model}")
        self.text_model = SentenceTransformer(text_model, device=device)

        # TODO: Load CLIP model for image embeddings when needed
        # from transformers import CLIPProcessor, CLIPModel
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> tuple:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Tuple of (embeddings_array, metadata_list)
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")

        with measure_time("Embedding generation") as timing:
            # Extract text from chunks
            texts = [chunk['text'] for chunk in chunks]

            # Generate embeddings
            embeddings = self.text_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for cosine similarity
            )

            # Create metadata for each embedding
            metadata = []
            for i, chunk in enumerate(chunks):
                meta = {
                    'chunk_id': chunk['chunk_id'],
                    'document_id': chunk['document_id'],
                    'document_type': chunk['document_type'],
                    'chunk_index': chunk['chunk_index'],
                    'embedding_dim': embeddings.shape[1]
                }
                metadata.append(meta)

        print(f"Generated {len(embeddings)} embeddings "
              f"(dim={embeddings.shape[1]}) "
              f"in {timing['elapsed_ms']:.2f}ms")

        return embeddings, metadata

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query string

        Returns:
            Query embedding vector
        """
        embedding = self.text_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding

    def save_embeddings(self,
                        embeddings: np.ndarray,
                        metadata: List[Dict[str, Any]],
                        output_dir: str):
        """
        Save embeddings and metadata to disk.

        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings as numpy array
        embeddings_file = output_path / "text_embeddings.npy"
        np.save(embeddings_file, embeddings)
        print(f"Saved embeddings to {embeddings_file}")

        # Save metadata as JSON
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")

        # Save embedding info
        info = {
            'model': self.text_model_name,
            'num_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'normalized': True
        }
        info_file = output_path / "embedding_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        print(f"Saved embedding info to {info_file}")

    def load_embeddings(self, input_dir: str) -> tuple:
        """
        Load embeddings and metadata from disk.

        Args:
            input_dir: Directory containing embedding files

        Returns:
            Tuple of (embeddings_array, metadata_list)
        """
        input_path = Path(input_dir)

        # Load embeddings
        embeddings_file = input_path / "text_embeddings.npy"
        embeddings = np.load(embeddings_file)
        print(f"Loaded embeddings from {embeddings_file} (shape={embeddings.shape})")

        # Load metadata
        metadata_file = input_path / "metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} metadata entries")

        return embeddings, metadata

    def compute_similarity(self,
                          query_embedding: np.ndarray,
                          doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding vector (1D)
            doc_embeddings: Document embeddings matrix (2D)

        Returns:
            Similarity scores array
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        if query_embedding.ndim == 1:
            similarities = np.dot(doc_embeddings, query_embedding)
        else:
            similarities = np.dot(doc_embeddings, query_embedding.T).flatten()

        return similarities

    def get_top_k(self,
                  similarities: np.ndarray,
                  k: int = 5) -> tuple:
        """
        Get top-k indices and scores.

        Args:
            similarities: Similarity scores array
            k: Number of top results to return

        Returns:
            Tuple of (top_indices, top_scores)
        """
        # Get top-k indices (highest scores first)
        top_indices = np.argsort(similarities)[::-1][:k]
        top_scores = similarities[top_indices]

        return top_indices, top_scores


def embed_corpus(chunks_file: str, output_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
    """
    Standalone function to embed corpus chunks.

    Args:
        chunks_file: Path to chunks JSON file
        output_dir: Output directory for embeddings
        model_name: SentenceTransformer model name
    """
    # Load chunks
    print(f"Loading chunks from {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Generate embeddings
    generator = EmbeddingGenerator(text_model=model_name)
    embeddings, metadata = generator.embed_chunks(chunks)

    # Save embeddings
    generator.save_embeddings(embeddings, metadata, output_dir)

    print(f"\n✓ Embedding generation complete!")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python embed.py <chunks_file> <output_dir> [model_name]")
        print("Example: python embed.py data/processed/chunks.json data/processed/embeddings")
        sys.exit(1)

    chunks_file = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else 'all-MiniLM-L6-v2'

    embed_corpus(chunks_file, output_dir, model_name)
