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
             visual_input_mode: str = "vlm_desc_url_image_file",
             instance_id: Optional[str] = None) -> Dict[str, Any]:
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
            instance_id: Optional instance ID for context-aware retrieval

        Returns:
            Dictionary with results and metrics
        """
        # 1. Retrieval stage
        retrieval_timer = Timer()
        retrieval_timer.start()

        # Pass instance_id if available and if retriever supports it
        # We check if retrieve accepts kwargs or instance_id
        # For now, we assume our OfflineRetriever needs it, but FlexibleRetriever might not.
        # We can pass it as a kwarg if the signature allows, or check signature.
        # But to be safe and simple, let's try passing it if it's not None.
        # Actually, FlexibleRetriever.retrieve signature is: retrieve(query, instance_id, strategy, ...)
        # So we should pass it.
        
        if instance_id:
             retrieval_result = self.retriever.retrieve(query, top_k=self.top_k, instance_id=instance_id)
        else:
             retrieval_result = self.retriever.retrieve(query, top_k=self.top_k)
        retrieved_docs = retrieval_result['retrieved_documents']

        retrieval_time_ms = retrieval_timer.stop()

        # 1.5 Apply Token Limit (if set)
        final_retrieval_limit = retrieval_token_limit

        # Use vlm_context if provided, otherwise fallback to vlm_descs
        if vlm_context:
            vlm_descs_list = [item['vlm_description'] for item in vlm_context]
            vlm_time_ms = sum(item.get('vlm_generation_time_ms', 0) for item in vlm_context)
        else:
            vlm_descs_list = vlm_descs if vlm_descs else []
            vlm_time_ms = 0

        # Always calculate token counts for metrics
        if self.tokenizer:
            # Estimate base tokens
            system_prompt_tokens = 100 # Rough estimate
            issue_tokens = len(self.tokenizer.encode(query))
            
            vlm_tokens = 0
            # Only count VLM text tokens if we are including descriptions
            if "vlm_desc" in visual_input_mode:
                for desc in vlm_descs_list:
                    vlm_tokens += len(self.tokenizer.encode(desc))
            
            base_tokens = system_prompt_tokens + issue_tokens + vlm_tokens
            
            if total_token_limit:
                retrieval_budget = total_token_limit - base_tokens
                
                if retrieval_budget < 0:
                    print(f"Warning: Base tokens ({base_tokens}) exceed total limit ({total_token_limit}). Retrieval budget is 0.")
                    retrieval_budget = 0
                
                print(f"Token Limit: Total={total_token_limit}, Base={base_tokens}, Retrieval Budget={retrieval_budget}")
                
                if final_retrieval_limit:
                    final_retrieval_limit = min(final_retrieval_limit, retrieval_budget)
                else:
                    final_retrieval_limit = retrieval_budget
        else:
            issue_tokens = 0
            vlm_tokens = 0

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
            
            # Construct Issue Text
            issue_text = query
            if "vlm_desc" in visual_input_mode and vlm_descs_list:
                issue_text += "\n\nVisual Context (from images):\n"
                for i, desc in enumerate(vlm_descs_list):
                    issue_text += f"Image {i+1}: {desc}\n"

            # Construct Code Context
            code_context = ""
            for doc in retrieved_docs:
                payload = doc.get('payload', {})
                # Try to get rel_path, fallback to filepath/path
                rel_path = doc.get('metadata', {}).get('rel_path') or payload.get('rel_path')
                if not rel_path:
                    raw_path = doc.get('path') or doc.get('metadata', {}).get('path') or payload.get('filepath')
                    if raw_path:
                        rel_path = Path(raw_path).name
                    else:
                        rel_path = "unknown_file"
                
                content = doc.get('text') or payload.get('text', '')
                
                code_context += f"[start of {rel_path}]\n{content}\n[end of {rel_path}]\n"

            prompt_text = f"""You will be provided with a partial code base and an issue statement explaining a problem to resolve.
<issue>
{issue_text}
</issue>

<code>
{code_context}
</code>

Here is an example of a patch file. It consists of changes to the code base. It specifies the file names, the line numbers of each change, and the removed and added lines. A single patch file can contain changes to multiple files.

<patch>
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
 def euclidean(a, b):
-    while b:
-        a, b = b, a % b
-    return a
+    if b == 0:
+        return a
+    return euclidean(b, a % b)
 
 def bresenham(x0, y0, x1, y1):
+    points = []
+    dx = abs(x1 - x0)
+    dy = abs(y1 - y0)
+    sx = 1 if x0 < x1 else -1
+    sy = 1 if y0 < y1 else -1
+    err = dx - dy
+    
     while True:
-        plot(x0, y0)
+        points.append((x0, y0))
         if x0 == x1 and y0 == y1:
             break
-        e2 = 2 * err
-        if e2 > -dy:
-            err -= dy
-            x0 += sx
-        if e2 < dx:
-            err += dx
-            y0 += sy
+        e2 = 2 * err
+        if e2 > -dy:
+            err -= dy
+            x0 += sx
+        if e2 < dx:
+            err += dx
+            y0 += sy
+    return points
+</patch>

I need you to solve the provided issue by generating a single patch file that I can apply directly to this repository using git apply. Please respond with a single patch file in the format shown above.

Respond below: 
"""
            
            # Construct Multimodal Prompt if needed
            if visual_input_mode == "vlm_desc":
                # Legacy text-only mode
                prompt_content = prompt_text
            else:
                # Multimodal mode
                prompt_content = [{"type": "text", "text": prompt_text}]
                
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
                prompt_tokens = 0 
            
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
            
            # Extract patch from <patch> tags
            import re
            patch_match = re.search(r'<patch>(.*?)</patch>', answer, re.DOTALL)
            if patch_match:
                final_answer = patch_match.group(1).strip()
            else:
                # Fallback: try to find markdown code block
                code_block_match = re.search(r'```(?:diff)?\n(.*?)```', answer, re.DOTALL)
                if code_block_match:
                    final_answer = code_block_match.group(1).strip()
                else:
                    # Fallback: just return raw answer
                    final_answer = answer.strip()

            # Clean markdown fences if present (extra safety for all cases)
            final_answer = re.sub(r'^```\w*\n', '', final_answer)
            final_answer = re.sub(r'\n```$', '', final_answer)
            
            # Ensure patch ends with newline (git apply requirement)
            if final_answer and not final_answer.endswith('\n'):
                final_answer += '\n'
        else:
            # QA Mode (legacy/simple)
            final_answer = "QA mode not fully implemented with new providers."
            llm_response = {}

        generation_time_ms = (time.time() - generation_start) * 1000       
        
        # 5. Compile results
        
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
        issue_text_tokens = issue_tokens
        vlm_tokens_count = vlm_tokens
        
        # Calculate total input prompt tokens from the actual prompt used
        if self.tokenizer and isinstance(prompt, str):
             total_input_prompt_tokens = len(self.tokenizer.encode(prompt))
        elif self.tokenizer and isinstance(prompt, list):
             # For multimodal list prompts, we can't easily count "tokens" in the same way.
             # We will approximate by summing text parts.
             total_input_prompt_tokens = 0
             for part in prompt:
                 if isinstance(part, dict) and part.get('type') == 'text':
                     total_input_prompt_tokens += len(self.tokenizer.encode(part['text']))
        else:
             total_input_prompt_tokens = prompt_tokens # Fallback to earlier estimate
        
        # Calculate prompt template tokens
        # total = template + issue + vlm + retrieved
        # template = total - (issue + vlm + retrieved)
        prompt_template_tokens = total_input_prompt_tokens - (issue_text_tokens + vlm_tokens_count + retrieved_tokens)
        
        # Ensure non-negative (can happen due to token merging differences)
        if prompt_template_tokens < 0:
            prompt_template_tokens = 0
            # Adjust total to match sum if we force template to 0? 
            # Or just accept the discrepancy. Let's accept discrepancy but report 0 for template.
        
        output_generated_tokens = llm_response.get('tokens_generated', 0)
        total_io_tokens = total_input_prompt_tokens + output_generated_tokens
        
        # Extract search_time_ms if available (pure search time)
        search_time_ms = retrieval_result.get('search_time_ms', 0)
        search_time_breakdown = retrieval_result.get('search_time_breakdown', {})
        
        # If search_time_ms is available, we use it as the "retrieval_time_ms" (pure search)
        # Otherwise we fall back to the measured retrieval_timer (which includes overhead)
        final_retrieval_time_ms = search_time_ms if search_time_ms > 0 else retrieval_time_ms
        
        total_retrieval_time = vlm_time_ms + final_retrieval_time_ms
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
                # 'num_images': len(vlm_descs_list), # Added in run_experiments.py if visual_mode != none
                # 'vlm_generation_time_ms': vlm_time_ms, # Added in run_experiments.py if visual_mode != none
                'retrieval_time_ms': final_retrieval_time_ms, # Pure search time if available
                # Granular search times are added in run_experiments.py based on active strategies
                'total_retrieval_time_ms': total_retrieval_time,
                'generation_time_ms': generation_time_ms,
                'total_io_time_ms': total_io_time_ms,
                
                # Extra debug metric

                
                'input_total_token_limit': total_token_limit,
                'issue_text_tokens': issue_text_tokens,
                'vlm_tokens': vlm_tokens_count,
                'retrieved_tokens': retrieved_tokens,
                'prompt_template_tokens': prompt_template_tokens,
                'total_input_prompt_tokens': total_input_prompt_tokens,
                'output_generated_tokens': output_generated_tokens,
                'total_io_tokens': total_io_tokens,
                'input_prompt_text': prompt if isinstance(prompt, str) else json.dumps(prompt),
                'retrieved_file_paths': [
                    (doc.get('metadata', {}).get('rel_path') or 
                     doc.get('payload', {}).get('rel_path') or 
                     Path(doc.get('path') or doc.get('metadata', {}).get('path') or doc.get('payload', {}).get('filepath') or "unknown").name)
                    for doc in retrieved_docs
                ]
            },
            'retrieval_method': getattr(self.retriever, 'retrieval_strategy', 'flexible'),
            'llm_response': llm_response,
            'mode': mode,
            # Debug/Verification Details
            'prompt': prompt,
            'issue_text': issue_text,
            'code_context': code_context,
            'vlm_descs': vlm_descs_list,
            'vlm_descs': vlm_descs_list,
            'search_time_breakdown': search_time_breakdown, # Pass for granular metrics in run_experiments
            'num_images': len(vlm_descs_list), # Pass for metrics in run_experiments
            'vlm_generation_time_ms': vlm_time_ms, # Pass for metrics in run_experiments
            'visual_embedding_time_ms': retrieval_result.get('visual_embedding_time_ms', 0.0) # Pass for metrics in run_experiments
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
            "text": "<patch>\ndiff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-foo\n+bar\n</patch>",
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
