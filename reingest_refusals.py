
import os
import time
import uuid
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from dotenv import load_dotenv
from embeddings.embed import EmbeddingGenerator

# Add project root to path
sys.path.insert(0, os.getcwd())

load_dotenv()

def generate_vlm_description(client: OpenAI, image_url: str, issue_text: str) -> str:
    system_prompt = """You are a Senior Front-End Engineer and UI/UX Specialist.
Your goal is to analyze a screenshot provided in a software issue report and "technically reverse-engineer" it into a search query.
Focus on:
1. Visual Symptom Analysis: Describe exactly what is wrong (misalignment, wrong color, broken icon, etc.).
2. UI Component Identification: Identify the likely React/Vue/HTML components involved.
3. Technical Keywords: Infer the CSS properties or logic likely causing the issue.

Output a concise, dense technical description that can be used to search the codebase.
IMPORTANT: If the image is a chart or abstract graphic, analyze it technically as a data visualization component. Do NOT refuse to analyze it."""

    user_prompt = f"""Issue Context: {issue_text}

Please analyze the provided image and generate the technical reverse-engineering description."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    ],
                },
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating VLM description: {e}")
        return "Error analyzing image"

def reingest_refusals():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    dense_gen = EmbeddingGenerator(model_type='dense', model_name='jinaai/jina-embeddings-v2-base-code')
    
    refusal_phrases = [
        "I'm sorry", 
        "I cannot", 
        "I can't help", 
        "I am sorry",
        "unable to analyze",
        "cannot analyze",
        "unable to identify",
        "Error analyzing image"
    ]
    
    print("Scanning for refusals to re-ingest...")
    
    # Scroll all points
    points = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False
        )
        points.extend(batch)
        if next_offset is None:
            break
            
    refusal_points = []
    for point in points:
        desc = point.payload.get('vlm_description', '').lower()
        if any(phrase.lower() in desc for phrase in refusal_phrases):
            refusal_points.append(point)
            
    print(f"Found {len(refusal_points)} refusals to process.")
    
    updated_count = 0
    
    for point in refusal_points:
        url = point.payload.get('image_url')
        print(f"\nProcessing {point.id} ({url})...")
        
        issue_text = point.payload.get('original_issue')
        
        # Generate New Description
        print("Generating new description...")
        new_desc = generate_vlm_description(openai_client, url, issue_text)
        
        # Check for Refusal again
        if any(phrase.lower() in new_desc.lower() for phrase in refusal_phrases):
            print(f"FAILURE: New description is still a refusal:\n{new_desc[:100]}...")
            # We skip updating if it failed again, or we could force it? 
            # Let's skip to avoid overwriting with another refusal, but maybe we should try again?
            # For now, just report it.
            continue
            
        print("SUCCESS: Valid description generated.")
        
        # Update Point
        new_embedding = dense_gen.embed_query(new_desc)
        
        new_payload = point.payload
        new_payload['vlm_description'] = new_desc
        new_payload['vlm_tokens'] = len(new_desc.split())
        
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point.id,
                    vector=new_embedding.tolist(),
                    payload=new_payload
                )
            ]
        )
        updated_count += 1
        print("Point updated.")
        
    print(f"\nUpdate complete. {updated_count}/{len(refusal_points)} images updated.")
    
    total_points = client.count(collection_name=collection_name).count
    print(f"Total points in collection: {total_points}")

if __name__ == "__main__":
    reingest_refusals()
