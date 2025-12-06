
import os
import base64
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from dotenv import load_dotenv
from embeddings.embed import EmbeddingGenerator

load_dotenv()

def ingest_manual_screenshot():
    image_path = "/Users/talesmp/.gemini/antigravity/brain/6dafeaae-2362-40b1-8af6-a849d46ff6c0/uploaded_image_1764916499535.png"
    target_instance_id = "markedjs__marked-1683"
    target_url_substring = "marked.js.org/demo" # To identify the specific point to update
    
    client = QdrantClient(path="data/qdrant/qdrant_data_swe_images")
    collection_name = "swe_images"
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    dense_gen = EmbeddingGenerator(model_type='dense', model_name='jinaai/jina-embeddings-v2-base-code')
    
    print(f"Reading image from {image_path}...")
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
    # 1. Find the target point
    print(f"Finding point for instance {target_instance_id}...")
    res = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="instance_id",
                    match=models.MatchValue(value=target_instance_id)
                )
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    points, _ = res
    
    target_point = None
    for point in points:
        if target_url_substring in point.payload.get('image_url', ''):
            target_point = point
            break
            
    if not target_point:
        print("FAILURE: Could not find the specific point with the invalid URL.")
        return

    print(f"Found point: {target_point.id}")
    issue_text = target_point.payload.get('original_issue')

    # 2. Generate Description
    print("Generating VLM description...")
    system_prompt = """You are a Senior Front-End Engineer and UI/UX Specialist.
    Your goal is to analyze a screenshot provided in a software issue report and "technically reverse-engineer" it into a search query.
    Focus on:
    1. Visual Symptom Analysis: Describe exactly what is wrong (misalignment, wrong color, broken icon, etc.).
    2. UI Component Identification: Identify the likely React/Vue/HTML components involved.
    3. Technical Keywords: Infer the CSS properties or logic likely causing the issue.

    Output a concise, dense technical description that can be used to search the codebase."""

    user_prompt = f"""Issue Context: {issue_text}

    Please analyze the provided image and generate the technical reverse-engineering description."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            },
                        },
                    ],
                },
            ],
            max_tokens=4096,
        )
        description = response.choices[0].message.content
        print("Description generated successfully.")
    except Exception as e:
        print(f"Error generating description: {e}")
        return

    # 3. Update Point
    print("Updating Qdrant point...")
    new_embedding = dense_gen.embed_query(description)
    
    new_payload = target_point.payload
    new_payload['vlm_description'] = description
    new_payload['vlm_tokens'] = len(description.split())
    new_payload['image_base64'] = base64_image # Update the base64 with the valid image
    # We keep the original URL as a reference, or should we mark it as manual?
    # Let's keep it but maybe add a note? For now, just update content.
    
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=target_point.id,
                vector=new_embedding.tolist(),
                payload=new_payload
            )
        ]
    )
    print("Point updated successfully.")

if __name__ == "__main__":
    ingest_manual_screenshot()
