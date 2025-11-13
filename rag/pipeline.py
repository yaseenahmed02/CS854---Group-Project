"""
RAG Pipeline
End-to-end retrieval-augmented generation pipeline with vLLM integration.
"""

import requests
import json
from typing import Dict, Any, Optional
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
from rag.prompt_builder import PromptBuilder
from utils.timer import Timer


class RAGPipeline:
    """End-to-end RAG pipeline with vLLM backend."""

    def __init__(self,
                 retriever_type: str = 'hybrid',
                 embeddings_dir: str = 'data/processed/embeddings',
                 chunks_file: str = 'data/processed/chunks.json',
                 vllm_url: str = 'http://localhost:8000',
                 top_k: int = 5,
                 alpha: float = 0.5):
        """
        Initialize RAG pipeline.

        Args:
            retriever_type: 'vector' or 'hybrid'
            embeddings_dir: Directory with embeddings
            chunks_file: Path to chunks JSON
            vllm_url: vLLM server URL
            top_k: Number of documents to retrieve
            alpha: Hybrid retrieval weight (BM25 vs vector)
        """
        self.retriever_type = retriever_type
        self.vllm_url = vllm_url.rstrip('/')
        self.top_k = top_k

        # Initialize retriever
        if retriever_type == 'vector':
            self.retriever = VectorRetriever(
                embeddings_dir=embeddings_dir,
                chunks_file=chunks_file
            )
        elif retriever_type == 'hybrid':
            self.retriever = HybridRetriever(
                embeddings_dir=embeddings_dir,
                chunks_file=chunks_file,
                alpha=alpha
            )
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(max_context_length=4096)

        print(f"RAG Pipeline initialized with {retriever_type} retrieval")

    def query(self,
             query: str,
             max_tokens: int = 256,
             temperature: float = 0.7) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieve + generate.

        Args:
            query: User query
            max_tokens: Max tokens to generate
            temperature: LLM temperature

        Returns:
            Dictionary with results and metrics
        """
        # 1. Retrieval stage
        retrieval_timer = Timer()
        retrieval_timer.start()

        retrieval_result = self.retriever.retrieve(query, top_k=self.top_k)
        retrieved_docs = retrieval_result['retrieved_documents']

        retrieval_time_ms = retrieval_timer.stop()

        # 2. Prompt building
        prompt_result = self.prompt_builder.build_prompt(query, retrieved_docs)
        prompt = prompt_result['prompt']
        prompt_tokens = prompt_result['estimated_tokens']

        # 3. LLM generation
        generation_timer = Timer()
        generation_timer.start()

        llm_response = self._call_vllm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        generation_time_ms = generation_timer.stop()

        # 4. Compile results
        result = {
            'query': query,
            'answer': llm_response.get('text', ''),
            'retrieved_documents': retrieved_docs,
            'num_retrieved': len(retrieved_docs),
            'metrics': {
                'retrieval_time_ms': retrieval_time_ms,
                'generation_time_ms': generation_time_ms,
                'total_time_ms': retrieval_time_ms + generation_time_ms,
                'prompt_tokens': prompt_tokens,
                'generated_tokens': llm_response.get('tokens_generated', 0),
                'total_tokens': prompt_tokens + llm_response.get('tokens_generated', 0)
            },
            'retrieval_method': self.retriever_type,
            'llm_response': llm_response
        }

        return result

    def _call_vllm(self,
                   prompt: str,
                   max_tokens: int = 256,
                   temperature: float = 0.7) -> Dict[str, Any]:
        """
        Call vLLM generation endpoint.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            LLM response dictionary
        """
        # TODO: Update endpoint based on actual vLLM API
        # This is a generic implementation that may need adjustment

        endpoint = f"{self.vllm_url}/v1/completions"

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            # Extract text from response
            # Format may vary based on vLLM version
            if 'choices' in result and len(result['choices']) > 0:
                text = result['choices'][0].get('text', '')
            else:
                text = result.get('text', '')

            return {
                'text': text.strip(),
                'tokens_generated': result.get('usage', {}).get('completion_tokens', 0),
                'raw_response': result
            }

        except requests.exceptions.RequestException as e:
            # Return error information if vLLM is not available
            return {
                'text': f'[vLLM Error: {str(e)}]',
                'tokens_generated': 0,
                'error': str(e)
            }

    def batch_query(self,
                   queries: list,
                   max_tokens: int = 256,
                   temperature: float = 0.7) -> list:
        """
        Process multiple queries.

        Args:
            queries: List of query strings
            max_tokens: Max tokens per generation
            temperature: LLM temperature

        Returns:
            List of result dictionaries
        """
        results = []
        for query in queries:
            result = self.query(query, max_tokens, temperature)
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Statistics dictionary
        """
        retriever_stats = self.retriever.get_statistics()
        stats = {
            'retriever_type': self.retriever_type,
            'vllm_url': self.vllm_url,
            'top_k': self.top_k,
            **retriever_stats
        }
        return stats


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <query> [retriever_type]")
        print("Example: python pipeline.py 'How does caching work?' hybrid")
        sys.exit(1)

    query = sys.argv[1]
    retriever_type = sys.argv[2] if len(sys.argv) > 2 else 'hybrid'

    # Initialize pipeline
    pipeline = RAGPipeline(
        retriever_type=retriever_type,
        embeddings_dir='data/processed/embeddings',
        chunks_file='data/processed/chunks.json',
        vllm_url='http://localhost:8000',
        top_k=5
    )

    # Execute query
    print(f"\nQuery: {query}")
    print(f"Retrieval method: {retriever_type}\n")

    result = pipeline.query(query)

    # Print results
    print("=== ANSWER ===")
    print(result['answer'])
    print("\n=== METRICS ===")
    metrics = result['metrics']
    print(f"Retrieval time: {metrics['retrieval_time_ms']:.2f}ms")
    print(f"Generation time: {metrics['generation_time_ms']:.2f}ms")
    print(f"Total time: {metrics['total_time_ms']:.2f}ms")
    print(f"Prompt tokens: {metrics['prompt_tokens']}")
    print(f"Generated tokens: {metrics['generated_tokens']}")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"\n=== RETRIEVED DOCUMENTS ({result['num_retrieved']}) ===")
    for doc in result['retrieved_documents'][:3]:
        print(f"[{doc['rank']}] {doc['document_id']} (score: {doc['retrieval_score']:.4f})")
