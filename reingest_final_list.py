
import os
import time
import base64
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from dotenv import load_dotenv
from embeddings.embed import EmbeddingGenerator

load_dotenv()

# List of targets. 
# For Chart.js and p5.js, we identify by URL suffix.
# For markedjs, we identify by instance_id and use local file.

TARGETS = [
    # Chart.js
    {"suffix": "test/fixtures/core.scale/border-behind-elements.png", "type": "url"},
    {"suffix": "test/fixtures/core.scale/grid-lines-scriptable.png", "type": "url"},
    {"suffix": "test/fixtures/core.scale/grid-lines-index-axis-y.png", "type": "url"},
    {"suffix": "test/fixtures/core.scale/grid-lines-index-axis-x.png", "type": "url"},
    {"suffix": "test/fixtures/controller.bar/horizontal-borders.png", "type": "url"},
    {"suffix": "test/fixtures/controller.line/point-style.png", "type": "url"},
    {"suffix": "test/fixtures/controller.bar/borderRadius/border-radius.png", "type": "url"},
    {"suffix": "test/fixtures/controller.line/clip/default-y-max.png", "type": "url"},
    {"suffix": "test/fixtures/controller.line/pointBorderWidth/value.png", "type": "url"},
    {"suffix": "test/fixtures/controller.bar/borderSkipped/vertical.png", "type": "url"},
    {"suffix": "test/fixtures/controller.bar/borderSkipped/horizontal.png", "type": "url"},
    {"suffix": "test/fixtures/controller.line/clip/default.png", "type": "url"},
    
    # p5.js
    {"suffix": "206325019-7879028e-85f1-4de3-81bb-896e06fd53ab.png", "type": "url"},
    
    # markedjs (Manual)
    {"instance_id": "markedjs__marked-1683", "type": "manual", "path": "/Users/talesmp/.gemini/antigravity/brain/6dafeaae-2362-40b1-8af6-a849d46ff6c0/uploaded_image_1764916499535.png"}
]

REFUSAL_PHRASES = [
    "I'm sorry", "I cannot", "I can't help", "I am sorry",
    "unable to analyze", "cannot analyze", "unable to identify",
    "Error analyzing image"
]

def generate_vlm_description(client: OpenAI, image_input: dict, issue_text: str) -> tuple[str, float]:
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

    content = [{"type": "text", "text": user_prompt}]
    
    if "url" in image_input:
        content.append({"type": "image_url", "image_url": {"url": image_input["url"], "detail": "high"}})
    elif "base64" in image_input:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_input['base64']}", "detail": "high"}})

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=4096,
        )
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        return response.choices[0].message.content, duration_ms
    except Exception as e:
        print(f"Error generating description: {e}")
        return "Error analyzing image", 0.0

def reingest_final_list():
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    dense_gen = EmbeddingGenerator(model_type='dense', model_name='jinaai/jina-embeddings-v2-base-code')
    
    print(f"Processing {len(TARGETS)} targets...")
    
    # Pre-fetch all points to search locally (more efficient than many scrolls)
    print("Fetching all points for lookup...")
    all_points = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False
        )
        all_points.extend(batch)
        if next_offset is None:
            break
            
    updated_count = 0
    
    for target in TARGETS:
        point_to_update = None
        
        # Find the point
        if target["type"] == "url":
            suffix = target["suffix"]
            for p in all_points:
                url = p.payload.get("image_url", "")
                if url.endswith(suffix):
                    point_to_update = p
                    break
            if not point_to_update:
                print(f"WARNING: Could not find point for suffix: {suffix}")
                continue
                
        elif target["type"] == "manual":
            instance_id = target["instance_id"]
            # For markedjs, we know the URL is the invalid one, so we search by instance_id
            # and pick the one that looks like the demo URL (or just the only one if unique)
            candidates = [p for p in all_points if p.payload.get("instance_id") == instance_id]
            if not candidates:
                print(f"WARNING: Could not find point for instance: {instance_id}")
                continue
            # Assuming the invalid one is the target. 
            # In previous steps we saw it has URL 'https://marked.js.org/demo/...'
            for p in candidates:
                if "marked.js.org" in p.payload.get("image_url", ""):
                    point_to_update = p
                    break
            if not point_to_update:
                 # Fallback: take the first one if only one
                 if len(candidates) == 1:
                     point_to_update = candidates[0]
                 else:
                     print(f"WARNING: Multiple points for {instance_id}, unsure which to update.")
                     continue

        print(f"\nProcessing {point_to_update.id}...")
        issue_text = point_to_update.payload.get('original_issue')
        
        # Prepare Image Input
        image_input = {}
        if target["type"] == "url":
            image_input["url"] = point_to_update.payload.get("image_url")
            print(f"Using URL: {image_input['url']}")
        elif target["type"] == "manual":
            print(f"Using manual file: {target['path']}")
            with open(target["path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                image_input["base64"] = b64
        
        # Generate
        print("Generating description...")
        desc, time_ms = generate_vlm_description(openai_client, image_input, issue_text)
        
        # Check Refusal
        if any(phrase.lower() in desc.lower() for phrase in REFUSAL_PHRASES):
            print(f"FAILURE: Generated description is a refusal:\n{desc[:100]}...")
            continue
            
        print(f"SUCCESS. Time: {time_ms:.2f}ms. Tokens: {len(desc.split())}")
        
        # Update
        new_embedding = dense_gen.embed_query(desc)
        new_payload = point_to_update.payload
        new_payload['vlm_description'] = desc
        new_payload['vlm_tokens'] = len(desc.split())
        new_payload['vlm_generation_time_ms'] = time_ms
        
        if target["type"] == "manual":
             new_payload['image_base64'] = image_input["base64"]
        
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_to_update.id,
                    vector=new_embedding.tolist(),
                    payload=new_payload
                )
            ]
        )
        updated_count += 1
        print("Point updated.")

    print(f"\nTotal updated: {updated_count}/{len(TARGETS)}")

if __name__ == "__main__":
    reingest_final_list()
