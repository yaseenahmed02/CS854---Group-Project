import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import json
import difflib
import os
import time
from openai import OpenAI
from typing import Dict, Any, Optional, List, Union
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
                 retriever: Any,
                 llm_model: str = "gpt-4o-2024-08-06",
                 llm_provider: str = "openai",
                 vllm_base_url: str = "http://localhost:8000/v1",
                 top_k: int = 5):
        """
        Initialize RAG Pipeline.
        
        Args:
            retriever: Initialized retriever instance
            llm_model: Model name to use
            llm_provider: 'openai', 'vllm', or 'mock'
            vllm_base_url: Base URL for vLLM API
            top_k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.vllm_base_url = vllm_base_url
        self.top_k = top_k
        self.tokenizer = None
        
        # Initialize Clients
        self.openai_client = None
        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                print("Warning: OPENAI_API_KEY not found. Fallback to mock.")
                self.llm_provider = "mock"
        
        # Initialize Tokenizer (for counting)
        # We prefer tiktoken for its speed and handling of long sequences without warnings
        import tiktoken
        try:
            if self.llm_provider == "openai":
                self.tokenizer = tiktoken.encoding_for_model(self.llm_model)
            else:
                # Fallback to cl100k_base (GPT-4) encoding for general counting
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback to transformers if tiktoken fails
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            except:
                self.tokenizer = None
                print("Error: Could not load any tokenizer.")

        print(f"RAG Pipeline initialized with {self.llm_provider} LLM and {self.retriever.__class__.__name__} retrieval")

    def query(self,
             query: str,
             max_tokens: int = 16384,
             temperature: float = 0.7,
             mode: str = "qa",
             retrieval_token_limit: Optional[int] = None,
             total_token_limit: Optional[int] = None,
             vlm_descs: Optional[List[str]] = None,
             vlm_context: Optional[List[Dict[str, Any]]] = None,
             visual_input_mode: str = "vlm_desc_url_image_file") -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieve + generate.

        Args:
            query: User query
            max_tokens: Max tokens to generate
            temperature: LLM temperature
            mode: 'qa' or 'code_gen'
            retrieval_token_limit: Optional limit on tokens from retrieved documents
            total_token_limit: Optional limit on total input tokens
            vlm_descs: Legacy list of descriptions
            vlm_context: List of dicts with visual context (desc, url, base64)
            visual_input_mode: 'vlm_desc', 'vlm_desc_url', 'image_file', 'vlm_desc_url_image_file'

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
        final_retrieval_limit = retrieval_token_limit

        # Use vlm_context if provided, otherwise fallback to vlm_descs
        if vlm_context:
            vlm_descs_list = [item['description'] for item in vlm_context]
            vlm_time_ms = sum(item.get('generation_time_ms', 0) for item in vlm_context)
        else:
            vlm_descs_list = vlm_descs if vlm_descs else []
            vlm_time_ms = 0

        if total_token_limit and self.tokenizer:
            # Estimate base tokens
            system_prompt_tokens = 100 # Rough estimate
            issue_tokens = len(self.tokenizer.encode(query))
            
            vlm_tokens = 0
            # Only count VLM text tokens if we are including descriptions
            if "vlm_desc" in visual_input_mode:
                for desc in vlm_descs_list:
                    vlm_tokens += len(self.tokenizer.encode(desc))
            
            base_tokens = system_prompt_tokens + issue_tokens + vlm_tokens
            retrieval_budget = total_token_limit - base_tokens
            
            if retrieval_budget < 0:
                print(f"Warning: Base tokens ({base_tokens}) exceed total limit ({total_token_limit}). Retrieval budget is 0.")
                retrieval_budget = 0
            
            print(f"Token Limit: Total={total_token_limit}, Base={base_tokens}, Retrieval Budget={retrieval_budget}")
            
            if final_retrieval_limit:
                final_retrieval_limit = min(final_retrieval_limit, retrieval_budget)
            else:
                final_retrieval_limit = retrieval_budget

        if final_retrieval_limit is not None and self.tokenizer and retrieved_docs:
            selected_docs = []
            current_tokens = 0
            for doc in retrieved_docs:
                content = doc.get('text', '')
                if not content:
                    payload = doc.get('payload', {})
                    if isinstance(payload, dict):
                        content = payload.get('text', '')
                    else:
                        content = getattr(payload, 'text', '')
                
                doc_tokens = len(self.tokenizer.encode(content))
                if current_tokens + doc_tokens <= final_retrieval_limit:
                    selected_docs.append(doc)
                    current_tokens += doc_tokens
                else:
                    break
            retrieved_docs = selected_docs
            print(f"Applied retrieval token limit: {final_retrieval_limit}. Retained {len(retrieved_docs)} docs.")

        # 2. Prompt building
        prompt_content = None # Can be str or list
        
        if mode == "code_gen":
            target_file_content = ""
            target_file_path = ""
            
            if retrieved_docs:
                top_doc = retrieved_docs[0]
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
            
            user_prompt_text = (
                f"Issue: {query}\n\n"
                f"Target File ({target_file_path}):\n"
                f"```\n{target_file_content}\n```\n\n"
            )
            
            # Add VLM Descriptions if requested
            if "vlm_desc" in visual_input_mode and vlm_descs_list:
                user_prompt_text += "\n\nVisual Context (from images):\n"
                for i, desc in enumerate(vlm_descs_list):
                    user_prompt_text += f"Image {i+1}: {desc}\n"
            
            user_prompt_text += "Please provide the fixed file content."
            
            full_text_prompt = f"{system_prompt}\n\n{user_prompt_text}"
            
            # Construct Multimodal Prompt if needed
            if visual_input_mode == "vlm_desc":
                # Legacy text-only mode
                prompt_content = full_text_prompt
            else:
                # Multimodal mode
                prompt_content = [{"type": "text", "text": full_text_prompt}]
                
                if vlm_context:
                    for item in vlm_context:
                        # Add URL if requested
                        if "url" in visual_input_mode and item.get("image_url"):
                            prompt_content.append({
                                "type": "image_url",
                                "image_url": {"url": item["image_url"]}
                            })
                        
                        # Add File (Base64) if requested
                        if "image_file" in visual_input_mode and item.get("image_base64"):
                            # Assuming JPEG/PNG, but base64 usually doesn't have header in our ingest?
                            # ingest_images_to_qdrant.py: base64.b64encode(response.content).decode('utf-8')
                            # We need to add data URI scheme
                            b64_data = item["image_base64"]
                            prompt_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_data}"}
                            })
            
            prompt = prompt_content # Assign to generic name
            
            if self.tokenizer and isinstance(prompt, str):
                prompt_tokens = len(self.tokenizer.encode(prompt))
            else:
                prompt_tokens = 0 # Hard to count for multimodal list without specific logic
            
        else:
            # Standard QA prompt
            prompt_result = self.prompt_builder.build_prompt(query, retrieved_docs)
            prompt = prompt_result['prompt']
            if self.tokenizer:
                prompt_tokens = len(self.tokenizer.encode(prompt))
            else:
                prompt_tokens = prompt_result['estimated_tokens']

        # 3. Generate Answer
        generation_start = time.time()
        
        if mode == "code_gen":
            # Dispatch to provider
            if self.llm_provider == "mock":
                llm_response = self._generate_mock(prompt, max_tokens=max_tokens)
            elif self.llm_provider == "openai":
                llm_response = self._generate_openai(prompt, max_tokens=max_tokens)
            elif self.llm_provider == "vllm":
                llm_response = self._generate_vllm(prompt, max_tokens=max_tokens)
            else:
                print(f"Unknown provider {self.llm_provider}, using mock.")
                llm_response = self._generate_mock(prompt, max_tokens=max_tokens)
                
            answer = llm_response.get('text', '')
            final_answer = extract_diff(answer) # Kept as utility function call
        else:
            # QA Mode (legacy/simple)
            final_answer = "QA mode not fully implemented with new providers."
            llm_response = {}

        generation_time_ms = (time.time() - generation_start) * 1000       
        
        # 4. Post-processing for code_gen (moved into the if mode == "code_gen" block above)
        # final_answer = answer # This line is now handled by the dispatch logic
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
        
        # Calculate detailed metrics
        retrieved_tokens = 0
        if self.tokenizer:
            for doc in retrieved_docs:
                 # Re-extract content logic
                 content = doc.get('text', '')
                 if not content:
                     payload = doc.get('payload', {})
                     if isinstance(payload, dict):
                         content = payload.get('text', '')
                     else:
                         content = getattr(payload, 'text', '')
                 retrieved_tokens += len(self.tokenizer.encode(content))
        
        # Metric Renaming and Calculation
        issue_text_tokens = issue_tokens if total_token_limit else 0
        vlm_tokens_count = vlm_tokens if total_token_limit else 0
        
        # Recalculate total input based on components as requested
        # Note: This might differ slightly from actual prompt_tokens if system prompt is excluded here
        # But user requested: total_input_prompt_tokens = vlm_tokens + issue_tokens + retrieved_tokens
        # We should probably include system_prompt_tokens if we want it to be accurate to "prompt_tokens"
        # However, strictly following user formula:
        total_input_prompt_tokens = vlm_tokens_count + issue_text_tokens + retrieved_tokens
        
        output_generated_tokens = llm_response.get('tokens_generated', 0)
        total_io_tokens = total_input_prompt_tokens + output_generated_tokens
        
        total_retrieval_time = vlm_time_ms + retrieval_time_ms
        total_io_time_ms = total_retrieval_time + generation_time_ms

        result = {
            'query': query,
            'answer': final_answer,
            'raw_llm_output': answer if mode == "code_gen" else None,
            'retrieved_documents': retrieved_docs,
            'num_retrieved': len(retrieved_docs),
            'metrics': {
                # Order requested by user
                # instance_id and experiment_id are added in run_experiments.py
                'num_images': len(vlm_descs_list),
                'vlm_generation_time_ms': vlm_time_ms,
                'retrieval_time_ms': retrieval_time_ms,
                'total_retrieval_time_ms': total_retrieval_time,
                'generation_time_ms': generation_time_ms,
                'total_io_time_ms': total_io_time_ms,
                
                'input_total_token_limit': total_token_limit,
                'issue_text_tokens': issue_text_tokens,
                'vlm_tokens': vlm_tokens_count,
                'retrieved_tokens': retrieved_tokens,
                'total_input_prompt_tokens': total_input_prompt_tokens,
                'output_generated_tokens': output_generated_tokens,
                'total_io_tokens': total_io_tokens
            },
            'retrieval_method': getattr(self.retriever, 'retrieval_strategy', 'flexible'),
            'llm_response': llm_response,
            'mode': mode
        }

        return result

    # Removed _call_vllm as it's replaced by _generate_vllm and other provider methods
    # def _call_vllm(self,
    #                prompt: str,
    #                max_tokens: int = 256,
    #                temperature: float = 0.7) -> Dict[str, Any]:
    #     """
    #     Call vLLM generation endpoint.

    #     Args:
    #         prompt: Input prompt
    #         max_tokens: Max tokens to generate
    #         temperature: Sampling temperature

    #     Returns:
    #         LLM response
    #     """
    #     # TODO: Update endpoint based on actual vLLM API
    #     # This is a generic implementation that may need adjustment

    #     sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.9)
    #     responses = self.vllm.generate(prompt, sampling_params=sampling_params)

    #     arrival_time = responses[0].metrics.arrival_time
    #     first_token_time = responses[0].metrics.first_token_time
    #     finished_time = responses[0].metrics.finished_time
    #     e2e = finished_time - arrival_time
    #     ttft = first_token_time - arrival_time

    #     print(f"arrival_time: {arrival_time} \n first_token_time: {first_token_time} \n finished_time: {finished_time}")

    #     return {
    #         'text': responses[0].outputs[0].text,
    #         'tokens_generated': len(responses[0].outputs[0].token_ids),
    #         'ttft_ms': ttft,
    #         'e2e_ms': e2e
    #         # 'raw_response': [response for response in responses] # could not include raw_resoponse as outout type (RequestOutput) is not JSON serializable
    #     }

    def _generate_openai(self, prompt: Union[str, List[Dict]], max_tokens: int = 16384) -> Dict[str, Any]:
        """Generate response using OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful software engineer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=max_tokens
            )
            text = response.choices[0].message.content
            tokens_generated = response.usage.completion_tokens
            return {
                "text": text,
                "tokens_generated": tokens_generated
            }
        except Exception as e:
            print(f"OpenAI generation failed: {e}")
            return {"text": "", "tokens_generated": 0}

    def _generate_vllm(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Generate response using vLLM API."""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.llm_model, # Use configured model name
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        
        try:
            response = requests.post(f"{self.vllm_base_url}/completions", headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            text = result['choices'][0]['text']
            # vLLM usage might differ, check response structure
            tokens_generated = len(text.split()) # Rough estimate if usage not provided
            if 'usage' in result:
                tokens_generated = result['usage'].get('completion_tokens', tokens_generated)
                
            return {
                "text": text,
                "tokens_generated": tokens_generated
            }
        except Exception as e:
            print(f"vLLM generation failed: {e}")
            return {"text": "", "tokens_generated": 0}

    def _generate_mock(self, prompt: str, max_tokens: int = 10) -> Dict[str, Any]:
        """Return a mock response for testing."""
        return {
            "text": "```diff\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-foo\n+bar\n```",
            "tokens_generated": 10
        }

    def batch_query(self,
                   queries: list,
                   max_tokens: int = 16384,
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
            'retriever_type': getattr(self.retriever, 'retrieval_strategy', 'flexible'),
            'llm_provider': self.llm_provider, # Changed from vllm_url
            'llm_model': self.llm_model, # Added llm_model
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
