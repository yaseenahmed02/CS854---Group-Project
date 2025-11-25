"""
Embeddings Module
Generates embeddings for text, code, and images using SentenceTransformers and other models.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
import torch
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from utils.timer import Timer, measure_time

class EmbeddingGenerator:
    """Generate embeddings for multimodal documents."""

    def __init__(self,
                 model_type: str = 'dense',
                 model_name: str = 'jinaai/jina-embeddings-v2-base-code',
                 device: str = 'cpu'):
        """
        Initialize embedding generator.

        Args:
            model_type: 'dense', 'sparse_splade', or 'sparse_bgem3'
            model_name: HuggingFace model name
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        self.model_type = model_type
        self.model_name = model_name

        print(f"Loading {model_type} embedding model: {model_name}")
        
        if model_type == 'dense':
            self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
            self.model.max_seq_length = 8192
            
        elif model_type == 'sparse_splade':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.to(device)
            
        elif model_type == 'sparse_bgem3':
            try:
                from FlagEmbedding import BGEM3FlagModel
                self.model = BGEM3FlagModel(model_name, use_fp16=False, device=device)
            except ImportError:
                print("FlagEmbedding not found. Please install it to use BGE-M3 sparse.")
                raise
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> tuple:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Tuple of (embeddings, metadata_list)
            embeddings can be np.ndarray (dense) or List[Dict] (sparse)
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")

        with measure_time("Embedding generation") as timing:
            texts = [chunk['text'] for chunk in chunks]
            
            if self.model_type == 'dense':
                embeddings = self.model.encode(
                    texts,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            elif self.model_type == 'sparse_splade':
                embeddings = self._generate_splade(texts)
            elif self.model_type == 'sparse_bgem3':
                embeddings = self._generate_bgem3_sparse(texts)
            
            # Create metadata
            metadata = []
            for i, chunk in enumerate(chunks):
                meta = {
                    'chunk_id': chunk['chunk_id'],
                    'document_id': chunk['document_id'],
                    'document_type': chunk['document_type'],
                    'chunk_index': chunk['chunk_index'],
                    'model_type': self.model_type
                }
                metadata.append(meta)

        print(f"Generated {len(embeddings)} embeddings in {timing['elapsed_ms']:.2f}ms")
        return embeddings, metadata

    def embed_query(self, query: str) -> Union[np.ndarray, Dict[str, float]]:
        """Generate embedding for a query."""
        if self.model_type == 'dense':
            return self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        elif self.model_type == 'sparse_splade':
            return self._generate_splade([query])[0]
        elif self.model_type == 'sparse_bgem3':
            return self._generate_bgem3_sparse([query])[0]

    def _generate_splade(self, texts: List[str]) -> List[Dict[str, float]]:
        """Generate SPLADE sparse embeddings."""
        # Simplified SPLADE implementation
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # SPLADE logic: max(log(1 + relu(logits))) * attention_mask
        inter = torch.log1p(torch.relu(logits))
        token_max = torch.max(inter * inputs["attention_mask"].unsqueeze(-1), dim=1).values
        
        results = []
        for i in range(token_max.shape[0]):
            # Get non-zero weights
            indices = token_max[i].nonzero().flatten()
            weights = token_max[i][indices]
            
            # Convert to token: weight dict
            sparse_vec = {}
            for idx, weight in zip(indices, weights):
                token = self.tokenizer.decode([idx])
                sparse_vec[token] = float(weight)
            results.append(sparse_vec)
            
        return results

    def _generate_bgem3_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        """Generate BGE-M3 lexical weights (sparse)."""
        # Using FlagEmbedding
        output = self.model.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        
        # output['lexical_weights'] is a list of dicts (token -> weight)
        if isinstance(output, dict) and 'lexical_weights' in output:
             return output['lexical_weights']
        
        # Fallback if structure is different (older versions?)
        return output

    def save_embeddings(self, embeddings: Any, metadata: List[Dict[str, Any]], output_dir: str):
        """Save embeddings to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.model_type == 'dense':
            np.save(output_path / "embeddings.npy", embeddings)
        else:
            # Save sparse as JSON or Pickle
            import pickle
            with open(output_path / "embeddings.pkl", 'wb') as f:
                pickle.dump(embeddings, f)

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_embeddings(self, input_dir: str) -> tuple:
        """
        Load embeddings and metadata from disk.
        
        Args:
            input_dir: Directory containing embeddings and metadata
            
        Returns:
            Tuple of (embeddings, metadata_list)
        """
        input_path = Path(input_dir)
        
        # Load metadata
        with open(input_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            
        # Load embeddings
        if (input_path / "embeddings.npy").exists():
            embeddings = np.load(input_path / "embeddings.npy")
        elif (input_path / "embeddings.pkl").exists():
            import pickle
            with open(input_path / "embeddings.pkl", 'rb') as f:
                embeddings = pickle.load(f)
        else:
            raise FileNotFoundError(f"No embeddings found in {input_dir}")
            
        return embeddings, metadata

def embed_corpus(chunks_file: str, output_dir: str, model_type: str, model_name: str):
    """Standalone function to embed corpus."""
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)

    generator = EmbeddingGenerator(model_type=model_type, model_name=model_name)
    embeddings, metadata = generator.embed_chunks(chunks)
    generator.save_embeddings(embeddings, metadata, output_dir)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python embed.py <chunks_file> <output_dir> <model_type> [model_name]")
        sys.exit(1)

    chunks_file = sys.argv[1]
    output_dir = sys.argv[2]
    model_type = sys.argv[3]
    model_name = sys.argv[4] if len(sys.argv) > 4 else 'jinaai/jina-embeddings-v2-base-code'

    embed_corpus(chunks_file, output_dir, model_type, model_name)
