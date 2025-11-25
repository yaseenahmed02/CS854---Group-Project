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
from vllm import LLM, SamplingParams
from utils.patch_cleaner import extract_diff
import tiktoken


class RAGPipeline:
    """End-to-end RAG pipeline with vLLM backend."""

    def __init__(self,
                 vllm: LLM = None, 
                 retriever_type: str = 'hybrid',
                 embeddings_dir: str = 'data/processed/embeddings',
                 chunks_file: str = 'data/processed/chunks.json',
                 top_k: int = 5,
                 alpha: float = 0.5,
                 tokenizer_name: str = "gpt-4o"):
        """
        Initialize RAG pipeline.

        Args:
            vllm: Pre-initialized vLLM instance
            retriever_type: 'vector' or 'hybrid'
            embeddings_dir: Directory with embeddings
            chunks_file: Path to chunks JSON
            top_k: Number of documents to retrieve
            top_k: Number of documents to retrieve
            alpha: Hybrid retrieval weight (BM25 vs vector)
            tokenizer_name: Name of the tokenizer to use for counting tokens
        """
        self.retriever_type = retriever_type
        if vllm is None:
            self.vllm = LLM(model="meta-llama/Meta-Llama-3-8B")
        else:
            self.vllm = vllm
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
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(tokenizer_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {tokenizer_name}. Fallback to cl100k_base. Error: {e}")
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                self.tokenizer = None
                print("Error: Could not load tiktoken.")

        print(f"RAG Pipeline initialized with {retriever_type} retrieval")

    def query(self,
             query: str,
             max_tokens: int = 256,
             temperature: float = 0.7,
             mode: str = "qa",
             retrieval_token_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieve + generate.

        Args:
            query: User query
            max_tokens: Max tokens to generate
            temperature: LLM temperature
            max_tokens: Max tokens to generate
            temperature: LLM temperature
            mode: 'qa' or 'code_gen'
            retrieval_token_limit: Optional limit on tokens from retrieved documents

        Returns:
            Dictionary with results and metrics
        """
        # 1. Retrieval stage
        retrieval_timer = Timer()
        retrieval_timer.start()

        retrieval_result = self.retriever.retrieve(query, top_k=self.top_k)
        retrieved_docs = retrieval_result['retrieved_documents']

        retrieval_time_ms = retrieval_timer.stop()

        # 1.5 Apply Token Limit (if set)
        if retrieval_token_limit and self.tokenizer and retrieved_docs:
            selected_docs = []
            current_tokens = 0
            for doc in retrieved_docs:
                content = doc.get('text', '')
                # Estimate or count tokens
                doc_tokens = len(self.tokenizer.encode(content))
                
                if current_tokens + doc_tokens <= retrieval_token_limit:
                    selected_docs.append(doc)
                    current_tokens += doc_tokens
                else:
                    # Stop adding if we exceed limit
                    break
            
            # Update retrieved_docs to only contain selected ones
            retrieved_docs = selected_docs

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
            if self.tokenizer:
                prompt_tokens = len(self.tokenizer.encode(prompt))
            else:
                prompt_tokens = len(prompt) // 4 # Estimate
            
        else:
            # Standard QA prompt
            prompt_result = self.prompt_builder.build_prompt(query, retrieved_docs)
            prompt = prompt_result['prompt']
            if self.tokenizer:
                prompt_tokens = len(self.tokenizer.encode(prompt))
            else:
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
            LLM response
        """
        # TODO: Update endpoint based on actual vLLM API
        # This is a generic implementation that may need adjustment

        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.9)
        responses = self.vllm.generate(prompt, sampling_params=sampling_params)

        arrival_time = responses[0].metrics.arrival_time
        first_token_time = responses[0].metrics.first_token_time
        finished_time = responses[0].metrics.finished_time
        e2e = finished_time - arrival_time
        ttft = first_token_time - arrival_time

        print(f"arrival_time: {arrival_time} \n first_token_time: {first_token_time} \n finished_time: {finished_time}")

        return {
            'text': responses[0].outputs[0].text,
            'tokens_generated': len(responses[0].outputs[0].token_ids),
            'ttft_ms': ttft,
            'e2e_ms': e2e
            # 'raw_response': [response for response in responses] # could not include raw_resoponse as outout type (RequestOutput) is not JSON serializable
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
