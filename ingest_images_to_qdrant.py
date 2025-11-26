import sys
import os
import time
import uuid
import json
import requests
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.getcwd())

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from datasets import load_dataset
from openai import OpenAI

from embeddings.embed import EmbeddingGenerator

# Load environment variables
load_dotenv()

def setup_qdrant(db_path: str = "data/qdrant/qdrant_data_swe_images") -> tuple[QdrantClient, str]:
    """
    Setup Qdrant client and collection for images.
    """
    print(f"Initializing Qdrant at {db_path}...")
    client = QdrantClient(path=db_path)
    
    collection_name = "swe_images"
    
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if not exists:
        print(f"Creating collection {collection_name}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE) # Jina-v2-base-code
        )
    else:
        print(f"Collection {collection_name} already exists.")
        
    return client, collection_name

def generate_vlm_description(client: OpenAI, image_url: str, issue_text: str, model_name: str = "gpt-4o-2024-08-06") -> str:
    """
    Generate a technical description of the image using GPT-4o.
    """
    system_prompt = """You are a Senior Front-End Engineer and UI/UX Specialist.
Your goal is to analyze a screenshot provided in a software issue report and "technically reverse-engineer" it into a search query.
Focus on:
1. Visual Symptom Analysis: Describe exactly what is wrong (misalignment, wrong color, broken icon, etc.).
2. UI Component Identification: Identify the likely React/Vue/HTML components involved (e.g., "SidebarNav", "ButtonPrimary", "GridContainer").
3. Technical Keywords: Infer the CSS properties or logic likely causing the issue (e.g., "z-index", "overflow: hidden", "flex-wrap").

Output a concise, dense technical description that can be used to search the codebase."""

    user_prompt = f"""Issue Context: {issue_text[:500]}...

Please analyze the attached screenshot and provide the technical reverse-engineering description."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            },
                        },
                    ],
                },
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating VLM description: {e}")
        return f"Error analyzing image: {issue_text[:100]}"

import requests
import base64

def download_and_encode_image(url: str) -> Optional[str]:
    """
    Download image from URL and convert to base64 string.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        else:
            print(f"Failed to download image {url}: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None

def ingest_images(limit: int = None, mock_vlm: bool = False, store_image: bool = False, split: str = "test", repo_filter: str = None, version_filter: str = None, vlm_model: str = "gpt-4o-2024-08-06"):
    """
    Ingest images from SWE-bench Multimodal.
    """
    # 1. Setup Qdrant
    client, collection_name = setup_qdrant()
    
    # 2. Setup OpenAI
    if not mock_vlm:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found. Switching to mock VLM mode.")
            mock_vlm = True
        else:
            openai_client = OpenAI(api_key=api_key)
    
    # 3. Load Dataset
    print(f"Loading SWE-bench Multimodal dataset ({split} split)...")
    try:
        # Using the huggingface dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 
    
    # 4. Initialize Embedding Model
    print("Initializing Jina embedding model...")
    dense_gen = EmbeddingGenerator(model_type='dense', model_name='jinaai/jina-embeddings-v2-base-code')
    
    points = []
    count = 0
    
    print("Processing instances...")
    for instance in dataset:
        if limit and count >= limit:
            break
            
        instance_id = instance.get('instance_id')
        repo = instance.get('repo')
        version = instance.get('version')
        
        # Apply filters
        if repo_filter and repo != repo_filter:
            continue
        if version_filter and version != version_filter:
            continue
            
        print(f"DEBUG: Found matching instance {instance_id}. Repo: {repo}, Version: {version}")
            
        problem_statement = instance.get('problem_statement', '')
        
        # Extract images
        images_raw = instance.get('image_assets')
        images = []
        
        if images_raw:
            import json
            import ast
            
            if isinstance(images_raw, str):
                try:
                    # Try JSON first
                    data = json.loads(images_raw)
                    if isinstance(data, dict):
                        for key, urls in data.items():
                            if isinstance(urls, list):
                                images.extend(urls)
                except:
                    try:
                        # Try literal eval
                        data = ast.literal_eval(images_raw)
                        if isinstance(data, dict):
                            for key, urls in data.items():
                                if isinstance(urls, list):
                                    images.extend(urls)
                    except:
                        print(f"Warning: Could not parse image_assets for {instance_id}")
            elif isinstance(images_raw, dict):
                 for key, urls in images_raw.items():
                    if isinstance(urls, list):
                        images.extend(urls)
            elif isinstance(images_raw, list):
                images = images_raw
        
        if not images:
            print(f"DEBUG: No images found for {instance_id} after parsing. Raw: {images_raw}")
            continue
            
        for img in images:
            # img might be a dict with 'url' or just a string
            image_url = img if isinstance(img, str) else img.get('url')
            
            if not image_url:
                continue
                
            print(f"Processing image for {instance_id}...")
            
            # Generate VLM Description
            start_time = time.time()
            if mock_vlm:
                vlm_desc = f"Mock VLM description for {instance_id}: Visual bug in sidebar navigation."
            else:
                vlm_desc = generate_vlm_description(openai_client, image_url, problem_statement, model_name=vlm_model)
            end_time = time.time()
            vlm_time_ms = (end_time - start_time) * 1000
            
            # Embed Description
            embedding = dense_gen.embed_query(vlm_desc)
            
            # Optional: Download and store image
            image_base64 = None
            if store_image:
                print(f"Downloading image from {image_url}...")
                image_base64 = download_and_encode_image(image_url)
            
            # Create Point
            point_id = str(uuid.uuid4())
            
            point = models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "instance_id": instance_id,
                    "image_url": image_url,
                    "vlm_description": vlm_desc,
                    "repo": repo,
                    "version": version,
                    "original_issue": problem_statement[:200],
                    "image_base64": image_base64,
                    "vlm_model": "mock" if mock_vlm else vlm_model,
                    "vlm_generation_time_ms": vlm_time_ms
                }
            )
            points.append(point)
            count += 1
            
            # Upsert batch
            if len(points) >= 10:
                client.upsert(collection_name=collection_name, points=points)
                print(f"Upserted batch of {len(points)} images.")
                points = []
                
            if limit and count >= limit:
                break

    # Upsert remaining
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Upserted final batch of {len(points)} images.")

    print(f"Ingestion complete. Total images processed: {count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--mock", action="store_true", help="Use mock VLM instead of OpenAI API")
    parser.add_argument("--store-image", action="store_true", help="Download and store image base64 in Qdrant")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (dev/test)")
    parser.add_argument("--repo", type=str, default=None, help="Filter by repository name")
    parser.add_argument("--version", type=str, default=None, help="Filter by version")
    parser.add_argument("--vlm_model", type=str, default="gpt-4o-2024-08-06", help="VLM model to use (default: gpt-4o-2024-08-06)")
    args = parser.parse_args()
    
    ingest_images(limit=args.limit, mock_vlm=args.mock, store_image=args.store_image, split=args.split, repo_filter=args.repo, version_filter=args.version, vlm_model=args.vlm_model)
