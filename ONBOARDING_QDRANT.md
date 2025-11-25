# Onboarding Guide: Qdrant Multimodal RAG

**Welcome to the Qdrant Multimodal RAG project!** This guide will help you get set up and running with our advanced retrieval system for SWE-bench.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Ingestion](#data-ingestion)
4. [Running Retrieval](#running-retrieval)
5. [Running Benchmarks](#running-benchmarks)
6. [Project Structure](#project-structure)

---

## Prerequisites

Before you begin, ensure you have:

*   **Python 3.10+** installed.
*   **OpenAI API Key**: Required for generating VLM descriptions of images.
*   **Qdrant**: The system uses a local Qdrant instance (embedded or server).
*   **vLLM (Optional)**: Required for the full RAG generation step, but you can run retrieval benchmarks without it.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd CS854---Group-Project
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory:
    ```bash
    OPENAI_API_KEY=sk-your-key-here
    ```

---

## Data Ingestion

The system needs two types of data: **Code** and **Images**.

### 1. Ingesting Code
This process chunks the codebase, generates embeddings (Dense, SPLADE, BGE), and stores them in Qdrant.

```bash
# Example: Ingesting the 'grommet' repository
python ingest_code_to_qdrant.py \
  --repo_path /path/to/grommet \
  --repo_name grommet/grommet \
  --repo_version 2.10.0
```

### 2. Ingesting Images (Multimodal)
This process loads the SWE-bench Multimodal dataset, uses GPT-4o to generate technical descriptions of issue screenshots, and stores them in the `swe_images` collection.

```bash
# Ingest test split (default)
python ingest_images_to_qdrant.py --limit 100

# Ingest dev split
python ingest_images_to_qdrant.py --split dev --limit 50

# Optional: Store raw image base64 in Qdrant
python ingest_images_to_qdrant.py --store-image
```

---

## Running Retrieval

You can test the retrieval logic directly using the `FlexibleRetriever`.

**Example Script:**
```python
from qdrant_client import QdrantClient
from retrieval.flexible_retriever import FlexibleRetriever

# Connect to DB
client = QdrantClient(path="./qdrant_data_grommet_grommet_2_10_0")
images_client = QdrantClient(path="./qdrant_data_swe_images")

# Init Retriever
retriever = FlexibleRetriever(
    client=client,
    collection_name="grommet_grommet_2_10_0",
    swe_images_collection="swe_images",
    images_client=images_client
)

# Run Multimodal Query
results = retriever.retrieve(
    query="Fix the sidebar overflow issue",
    instance_id="grommet__grommet-123", # Needed to find associated images
    strategy=["jina", "bm25"],          # Hybrid Text Search
    visual_mode="fusion"                # Fuse with Visual Search
)

print(results['retrieved_documents'])
```

---

## Running Benchmarks

We have a comprehensive benchmark suite to evaluate different retrieval strategies.

### 1. Run the Experiments
This script runs 13 different configurations (e.g., `text_bm25`, `multimodal_fusion_jina`) and saves the results.

```bash
# Run all experiments on the test split
python benchmark/run_experiments.py

# Run on dev split with a limit (good for testing)
python benchmark/run_experiments.py --split dev --limit 5

# Run with mock vLLM (if you don't have a GPU server)
python benchmark/run_experiments.py --mock
```

### 2. Evaluate Results
Use the official SWE-bench harness to score the generated patches.

```bash
python -m swebench.harness.run_evaluation \
    --predictions_path results/ \
    --dataset_name princeton-nlp/SWE-bench_Multimodal \
    --split test
```

### 3. Measure Recall
To rigorously evaluate the retrieval performance (checking if the retrieved files match the actual modified files in the solution), run the recall analysis script.

```bash
python benchmark/measure_recall.py
```
This will output a summary table and save detailed statistics to `results/recall_analysis.csv`.

---

## Project Structure

*   **`ingest_code_to_qdrant.py`**: Main script for code ingestion.
*   **`ingest_images_to_qdrant.py`**: Main script for image ingestion.
*   **`benchmark/run_experiments.py`**: Orchestrates the 13-way experimental study.
*   **`retrieval/`**:
    *   `flexible_retriever.py`: Core logic for dynamic strategy selection.
    *   `hybrid_retriever.py`: Implements RRF fusion.
*   **`embeddings/`**:
    *   `embed.py`: Wrappers for Jina, SPLADE, and BGE-M3 models.
*   **`rag/`**:
    *   `pipeline.py`: End-to-end RAG (Retrieval + Generation).
*   **`verification/`**: Helper scripts to verify system integrity.
