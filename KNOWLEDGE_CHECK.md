## Phase 1: Ingestion (Preparing the Data)

Before the system can answer queries, it must index the codebase and the visual assets (screenshots) from the issue reports.

### Step 1: Code Ingestion

* [x] **Script**: `ingest_code_to_qdrant.py`
* [x] **Loading**: It reads the target repository (e.g., `grommet/grommet`) from the disk.
* [x] **Chunking**: The **SemanticChunker** (`utils/ingestion_utils.py`) splits the code into manageable pieces (approx. 8192 tokens), trying to respect function/class boundaries using tree-sitter.
* [x] **Embedding**: For each chunk, **EmbeddingGenerator** (`embeddings/embed.py`) creates three types of vectors:
    * [x] **Dense**: `jina-embeddings-v2-base-code` (captures semantic meaning).
    * [x] **Sparse (Neural)**: `Splade_PP_en_v1` (captures learned keywords).
    * [x] **Sparse (Multilingual)**: `bge-m3` (captures long-context keywords).
* [x] **Storage**: These vectors + metadata (file path, code content) are saved into a local **Qdrant** collection specific to that repo version (e.g., `qdrant_data_grommet_grommet_2_10_0`).

### Step 2: Multimodal Ingestion

* [x] **Script**: `ingest_images_to_qdrant.py`
* [x] **Loading**: It loads the SWE-bench Multimodal dataset and extracts images associated with issues.
* [x] **VLM Analysis**: It sends the image + issue text to **GPT-4o**. The system prompt instructs GPT-4o to act as a "Senior Front-End Engineer" and reverse-engineer the UI bug into a technical search query (e.g., "identifying the Sidebar component has overflow: hidden").
* [x] **Embedding**: This generated text description is embedded using the Dense model (**Jina**).
* [x] **Storage**: The vector + description is saved into a global **Qdrant** collection called `swe_images`. (Optional: Image file stored as Base64 if requested).

## Phase 2: Retrieval (Finding the Right Context)

When a user asks a question (or the benchmark runs), the **FlexibleRetriever** (`retrieval/flexible_retriever.py`) finds the most relevant code.

### Step 3: Dynamic Retrieval Strategies

The retriever is stateless and accepts a `strategy` argument at runtime. It supports:

* [x] **Text-Only Retrieval**:
    * [x] **BM25**: Builds an on-the-fly index from Qdrant documents for traditional keyword search.
    * [x] **Sparse (SPLADE/BGE)**: Finds code with matching "learned" keywords.
    * [x] **Dense (Jina)**: Finds code conceptually similar to the query.
    * [x] **Hybrid**: Combines the above (e.g., Jina + SPLADE) using **Reciprocal Rank Fusion (RRF)** to get the best of both worlds.
* [x] **Multimodal Retrieval**:
    * [x] **Visual-Only**: Ignores the user's text and searches only using the VLM's technical description.
    * [x] **Augment**: Fetches the **VLM** description for the issue and appends it to the text query (e.g., "Fix the sidebar... [VLM: Sidebar component has z-index issue]").
    * [x] **Fusion**: Executes two separate searches—one for the user's text and one for the VLM's description—and fuses the results.

## Phase 3: Generation (Creating the Patch)

Once relevant code files are found, the system generates a fix.

### Step 4: RAG Pipeline

* [x] **Class**: `RAGPipeline` (`rag/pipeline.py`)
* [x] **Prompting**: It constructs a prompt for the **LLM** (vLLM server).
    * [x] **System Prompt**: "You are a Senior Software Engineer... Rewrite the entire file to fix the issue."
    * [x] **User Prompt**: Contains the Issue Text + The Content of the Retrieved File.
* [x] **Generation**: The LLM generates the full, corrected file content.
* [x] **Post-Processing**:
    * [x] **Extraction**: `utils.patch_cleaner.extract_diff` pulls the code block from the LLM's markdown response.
    * [x] **Diffing**: The system computes a **Unified Diff** between the original file and the LLM's generated file. This diff is the final output.

## Phase 4: Evaluation (Benchmarking)

* [x] **Script**: `benchmark/run_experiments.py` This script orchestrates the entire study.
* [x] **Configuration**: Defines the 13 experiments (e.g., `text_bm25`, `multimodal_fusion_jina`).
* [x] **Execution Loop**:
    * [x] Iterates through the SWE-bench dataset.
    * [x] For each instance, it initializes the **FlexibleRetriever** with the specific experiment configuration (e.g., `strategies=['jina', 'bge'], visual_mode='fusion'`).
    * [x] Runs the **RAGPipeline** to generate a patch.
* [x] **Output**: Saves the generated patches to `results/{experiment_id}_predictions.json`.
* [x] **Scoring**: These JSON files are then fed into the official **SWE-bench** harness to verify if the patches actually pass the unit tests.