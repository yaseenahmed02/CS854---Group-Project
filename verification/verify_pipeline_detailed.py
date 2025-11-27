import argparse
import os
import sys
import json
from pathlib import Path
from datasets import load_dataset

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingestion_engine import IngestionEngine
from offline_experiment_runner import OfflineExperimentRunner, OfflineRetriever
from rag.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Run detailed verification for a single instance")
    parser.add_argument("--repo", required=True, help="Repository name (e.g., markedjs/marked)")
    parser.add_argument("--version", required=True, help="Version (e.g., 1.2.3)")
    parser.add_argument("--instance_id", required=True, help="Instance ID (e.g., markedjs__marked-1821)")
    parser.add_argument("--split", default="dev", help="Dataset split")
    parser.add_argument("--mock_vlm", action="store_true", help="Use mock VLM")
    parser.add_argument("--mock_llm", action="store_true", help="Use mock LLM")
    parser.add_argument("--token_limit", type=int, default=None, help="Total token limit for input")
    
    args = parser.parse_args()
    
    print(f"Running verification for {args.instance_id} ({args.repo} {args.version})...")
    
    # 1. Load Instance
    print("Loading dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=args.split)
    instance = next((item for item in dataset if item["instance_id"] == args.instance_id), None)
    
    if not instance:
        print(f"Error: Instance {args.instance_id} not found in {args.split} split.")
        return

    print("Instance found.")
    print(f"Problem Statement:\n{instance['problem_statement']}\n")
    
    # 2. Initialize Ingestion (to ensure data exists)
    # We assume data is already ingested or we are using existing data.
    # If not, we might need to run ingestion here, but let's assume it's done.
    
    # 3. Initialize Pipeline Components
    print("Initializing pipeline...")
    
    # Setup Clients
    from qdrant_client import QdrantClient
    from utils.ingestion_utils import sanitize_path_component
    
    safe_repo = sanitize_path_component(args.repo)
    safe_version = sanitize_path_component(args.version)
    db_path = f"data/qdrant/qdrant_data_{safe_repo}_{safe_version}"
    collection_name = f"{safe_repo}_{safe_version}"
    
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} does not exist.")
        return

    client = QdrantClient(path=db_path)
    issues_client = QdrantClient(path="data/qdrant/qdrant_data_swe_bench_issues")
    images_client = QdrantClient(path="data/qdrant/qdrant_data_swe_bench_images")

    # Setup Retriever
    retriever = OfflineRetriever(
        client=client,
        collection_name=collection_name,
        issues_client=issues_client,
        images_client=images_client
    )
    
    # Monkey patch retrieve to inject default strategy/visual mode for this test
    # This mimics what OfflineExperimentRunner does
    original_retrieve = retriever.retrieve
    retriever.retrieve = lambda q, top_k, instance_id=None: original_retrieve(
        q, 
        instance_id=instance_id or args.instance_id, 
        strategy=["jina", "bm25"], # Default hybrid strategy
        visual_mode="fusion",      # Default visual mode
        top_k=top_k, 
        repo=args.repo, 
        version=args.version
    )
    
    # Setup Pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_provider="mock" if args.mock_llm else "openai",
        llm_model="gpt-4o",
        vllm_base_url=None
    )
    
    # 4. Run Pipeline
    print("Executing pipeline...")
    
    # Determine visual mode
    visual_mode = "vlm_desc" # Default
    
    # Retrieve VLM descriptions (simulating what OfflineExperimentRunner does)
    # Actually, RAGPipeline handles this if we pass vlm_context or if we let it handle it.
    # But OfflineExperimentRunner usually prepares vlm_context.
    # Let's use the retriever to get VLM descriptions first to pass them in, 
    # or rely on RAGPipeline to fetch them if we don't pass them?
    # RAGPipeline.query doesn't fetch VLM descriptions itself unless we pass them or it has logic.
    # In run_experiments.py, it does:
    # vlm_context = retriever.retrieve_visuals(...)
    # Let's do that.
    
    vlm_context = []
    if instance.get("image_assets"):
        print("Retrieving visual context...")
        # We need to extract image URLs/paths from instance to retrieve their descriptions
        # The retriever.retrieve_visuals method might expect something specific.
        # Let's check OfflineRetriever.retrieve_visuals... it doesn't exist?
        # FlexibleRetriever has it?
        # Actually, in run_experiments.py:
        # vlm_context = []
        # if "vlm" in visual_input_mode:
        #    # It calls ingestion or retrieves from Qdrant?
        #    # Actually, it seems it relies on pre-computed descriptions in Qdrant.
        #    # The pipeline.query takes vlm_context.
        pass

    # For now, let's run query and see what happens. 
    # We need to pass the instance_id to the retriever?
    # OfflineRetriever.retrieve takes instance_id.
    
    result = pipeline.query(
        query=instance["problem_statement"],
        mode="code_gen",
        visual_input_mode=visual_mode,
        instance_id=args.instance_id,
        total_token_limit=args.token_limit,
        # We need to pass arguments that help retrieval
        # The retrieve method signature: retrieve(query, instance_id, ...)
        # But pipeline.query calls retriever.retrieve(query, top_k)
        # It doesn't pass instance_id!
        # Wait, OfflineRetriever needs instance_id to find relevant images?
        # Or does it use the query?
        # Let's check OfflineRetriever.retrieve signature.
    )
    
    # Wait, RAGPipeline.query calls self.retriever.retrieve(query, top_k=self.top_k)
    # It does NOT pass instance_id.
    # This might be a limitation if OfflineRetriever needs it.
    # Let's check OfflineRetriever.retrieve.
    
    # If OfflineRetriever needs instance_id, we might need to hack it or pass it in query?
    # Actually, for visual retrieval, we usually need the instance_id to look up the images for that instance.
    # If RAGPipeline doesn't support passing instance_id, we might have a problem with visual retrieval integration
    # unless we pass it via kwargs or something.
    
    # Let's assume for now we just run it and see.
    
    print("\n" + "="*80)
    print("VERIFICATION LOG")
    print("="*80)
    
    print(f"\n[Problem Statement]\n{instance['problem_statement']}")
    
    print(f"\n[VLM Descriptions]")
    if result.get('vlm_descs'):
        for i, desc in enumerate(result['vlm_descs']):
            print(f"Image {i+1}: {desc}")
    else:
        print("No VLM descriptions used/found.")
        
    print(f"\n[Issue Text in Prompt]\n{result.get('issue_text', 'N/A')}")
    
    print(f"\n[Code Context in Prompt]\n{result.get('code_context', 'N/A')}")
    
    print(f"\n[Full Prompt Sent to LLM]")
    print("-" * 40)
    prompt_content = result.get('prompt')
    if isinstance(prompt_content, list):
        # Multimodal prompt
        for item in prompt_content:
            print(f"Type: {item['type']}")
            if item['type'] == 'text':
                print(item['text'])
            elif item['type'] == 'image_url':
                print(f"[Image URL: {item['image_url']['url'][:50]}...]")
    else:
        print(prompt_content)
    print("-" * 40)
    
    print(f"\n[LLM Result]")
    answer = result.get('answer', '')
    # Clean markdown fences if present (extra safety)
    import re
    answer = re.sub(r'^```\w*\n', '', answer.strip())
    answer = re.sub(r'\n```$', '', answer.strip())
    print(answer)
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
