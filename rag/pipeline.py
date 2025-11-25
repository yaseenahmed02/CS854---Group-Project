import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import json
import difflib
from typing import Dict, Any, Optional
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
from rag.prompt_builder import PromptBuilder
from utils.timer import Timer
from utils.patch_cleaner import extract_diff


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
             temperature: float = 0.7,
             mode: str = "qa") -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieve + generate.

        Args:
            query: User query
            max_tokens: Max tokens to generate
            temperature: LLM temperature
            mode: 'qa' or 'code_gen'

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
        if mode == "code_gen":
            # For code gen, we need a specific prompt
            # And we assume the retrieved docs contain the file to be patched
            # Let's find the most relevant file from docs
            target_file_content = ""
            target_file_path = ""
            
            if retrieved_docs:
                # Naively take the top document as the target file
                # In a real scenario, we might want to let the LLM decide or pass it explicitly
                top_doc = retrieved_docs[0]
                # If we have the full file content in 'text' or can load it
                # The retrieved doc 'text' might be a chunk.
                # If we want to rewrite the ENTIRE file, we need the full content.
                # Let's check if 'path' is in metadata and load it if possible.
                # If not, we rely on the chunk (which might be partial).
                # For this exercise, let's assume we try to load from disk if path exists.
                
                doc_path = top_doc.get('path') or top_doc.get('metadata', {}).get('path')
                if doc_path and Path(doc_path).exists():
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            target_file_content = f.read()
                        target_file_path = doc_path
                    except:
                        target_file_content = top_doc.get('text', '')
                else:
                    target_file_content = top_doc.get('text', '')
            
            system_prompt = (
                "You are a Senior Software Engineer. Your task is to fix the issue described in the query.\n"
                "Rewrite the entire file to fix the issue. Output the full, valid file content enclosed in markdown code blocks.\n"
                "Do not output diffs. Do not output explanations outside the code block unless necessary.\n"
            )
            
            user_prompt = (
                f"Issue: {query}\n\n"
                f"Target File ({target_file_path}):\n"
                f"```\n{target_file_content}\n```\n\n"
                "Please provide the fixed file content."
            )
            
            prompt = f"{system_prompt}\n\n{user_prompt}"
            prompt_tokens = len(prompt) // 4 # Estimate
            
        else:
            # Standard QA prompt
            prompt_result = self.prompt_builder.build_prompt(query, retrieved_docs)
            prompt = prompt_result['prompt']
            prompt_tokens = prompt_result['estimated_tokens']

        # 3. LLM generation
        generation_timer = Timer()
        generation_timer.start()

        llm_response = self._call_vllm(
            prompt,
            max_tokens=max_tokens if mode == "qa" else 4096, # Allow more tokens for code gen
            temperature=temperature
        )

        generation_time_ms = generation_timer.stop()
        
        answer = llm_response.get('text', '')
        
        # 4. Post-processing for code_gen
        final_answer = answer
        if mode == "code_gen" and target_file_content:
            # Extract code
            generated_code = extract_diff(answer)
            
            # Calculate unified diff
            original_lines = target_file_content.splitlines(keepends=True)
            generated_lines = generated_code.splitlines(keepends=True)
            
            diff = difflib.unified_diff(
                original_lines,
                generated_lines,
                fromfile=f"a/{Path(target_file_path).name}" if target_file_path else "a/original",
                tofile=f"b/{Path(target_file_path).name}" if target_file_path else "b/modified",
                lineterm=""
            )
            
            diff_text = "".join(diff)
            if diff_text:
                final_answer = diff_text
            else:
                final_answer = "No changes detected or diff generation failed."

        # 5. Compile results
        result = {
            'query': query,
            'answer': final_answer,
            'raw_llm_output': answer if mode == "code_gen" else None,
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
            'llm_response': llm_response,
            'mode': mode
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
                timeout=120 # Increased timeout for code gen
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
        print("Usage: python pipeline.py <query> [retriever_type] [mode]")
        print("Example: python pipeline.py 'Fix the bug' hybrid code_gen")
        sys.exit(1)

    query = sys.argv[1]
    retriever_type = sys.argv[2] if len(sys.argv) > 2 else 'hybrid'
    mode = sys.argv[3] if len(sys.argv) > 3 else 'qa'

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
    print(f"Retrieval method: {retriever_type}")
    print(f"Mode: {mode}\n")

    result = pipeline.query(query, mode=mode)

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
