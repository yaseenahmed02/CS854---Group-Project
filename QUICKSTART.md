# Quick Start Guide

Get the RAG evaluation framework running in 5 minutes.

## Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
./setup.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Load and chunk the corpus
4. Generate embeddings

## Option 2: Manual Setup

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Prepare Corpus

```bash
# All-in-one preparation
python3 prepare_corpus.py
```

Or step-by-step:

```bash
# Load and chunk documents
python3 -c "
from utils.file_loader import FileLoader
from utils.chunker import Chunker

loader = FileLoader('data/raw')
documents = loader.load_all_documents()

chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_documents(documents)
chunker.save_chunks(chunks, 'data/processed/chunks.json')
"

# Generate embeddings
python3 embeddings/embed.py data/processed/chunks.json data/processed/embeddings
```

## Quick Tests

### Test Retrieval

**Vector-only:**
```bash
python3 retrieval/vector_retriever.py "How does authentication work?"
```

**Hybrid (BM25 + Vector):**
```bash
python3 retrieval/hybrid_retriever.py "validateToken function" 0.5
```

### Run Benchmark (No vLLM Required)

```bash
python3 benchmark/run_benchmark.py --methods vector hybrid
```

Note: Without vLLM, the LLM generation will fail gracefully, but retrieval metrics will still be collected.

## With vLLM (Full Pipeline)

### 1. Start vLLM Server

```bash
# Example: Llama-3 8B
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B \
  --port 8000
```

### 2. Test RAG Pipeline

```bash
python3 rag/pipeline.py "What is the cache hit ratio target?" hybrid
```

### 3. Run Full Benchmark

```bash
python3 benchmark/run_benchmark.py \
  --methods vector hybrid \
  --top-k 5 \
  --alpha 0.5 \
  --vllm-url http://localhost:8000
```

## Verify Installation

```bash
# Check corpus stats
python3 -c "
from utils.chunker import Chunker
chunks = Chunker().load_chunks('data/processed/chunks.json')
print(f'Loaded {len(chunks)} chunks')
"

# Check embeddings
python3 -c "
from embeddings.embed import EmbeddingGenerator
gen = EmbeddingGenerator()
emb, meta = gen.load_embeddings('data/processed/embeddings')
print(f'Loaded {len(emb)} embeddings (dim={emb.shape[1]})')
"
```

## Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'sentence_transformers'`**
```bash
pip install sentence-transformers
```

**Issue: First embedding generation is slow**
- This is normal! SentenceTransformers downloads the model on first use (~90MB)
- Subsequent runs will be fast

**Issue: vLLM connection error**
- Benchmark works without vLLM (retrieval-only mode)
- Check vLLM is running: `curl http://localhost:8000/health`

**Issue: Out of memory during embedding**
- Use a smaller model: `all-MiniLM-L6-v2` (default, 384 dim)
- Or use CPU: The code defaults to CPU

## Next Steps

1. Review results in `benchmark/results/`
2. Modify queries in `benchmark/workload.json`
3. Add your own documents to `data/raw/`
4. Adjust chunking parameters in `prepare_corpus.py`
5. Tune hybrid alpha in benchmark (0.0-1.0)

## Expected Output

After running the benchmark, you should see:

```
=== VECTOR RETRIEVAL METRICS ===
Total queries: 10
Retrieval Latency:
  P50: 12.34ms
  P95: 18.56ms
...

=== HYBRID RETRIEVAL METRICS ===
Total queries: 10
Retrieval Latency:
  P50: 15.67ms
  P95: 21.34ms
...

VECTOR vs HYBRID COMPARISON
  ✓ Hybrid is 12.8% FASTER (p50 latency)
  ✓ Hybrid uses 20.0% FEWER tokens
```

Files created:
- `benchmark/results/benchmark_results_<timestamp>.json`
- `benchmark/results/benchmark_metrics_<timestamp>.csv`

Happy benchmarking!
