# Composite Evaluation Framework for Multimodal RAG

A modular evaluation framework for measuring system-level performance differences between vector-only (dense) and hybrid (sparse+dense) retrieval in multimodal RAG pipelines.

## Project Overview

### Motivation

This project addresses a critical gap in existing multimodal RAG benchmarks:

**Existing benchmarks (SWE-Bench M, RepoBench, ArtifactsBench) measure task completion quality but ignore:**
- Retrieval precision impact on system-level performance
- KV-cache memory utilization
- LLM generation latency (p50/p99)
- Prompt token count and throughput

### Hypothesis

**Retrieval precision directly affects system-level LLM performance in vLLM.**

Specifically:
- **Vector-only retrieval ("sloppy")** retrieves large, irrelevant, high-token documents → inflates KV-cache → increases latency → reduces throughput
- **Hybrid retrieval (BM25 + dense vectors)** retrieves smaller, lexically-precise documents → reduces KV-cache load → improves latency and throughput

This is a **systems-performance and information retrieval precision problem**, not an agentic benchmark problem.

## Architecture

### Three-Pillar Evaluation Framework

```
┌─────────────────────────────────────────────────────────┐
│  1. MULTIMODAL SYNTHETIC CORPUS                         │
│     • Text documentation (MD files)                     │
│     • Code snippets (Python, JavaScript)                │
│     • Images (system diagrams, UI mockups)              │
├─────────────────────────────────────────────────────────┤
│  2. DIFFERENTIATED QUERY WORKLOAD                       │
│     • Semantic-dominant queries (CodeQueries-like)      │
│     • Keyword-dominant queries (function names, errors) │
│     • Hybrid queries (realistic SWE-Bench M-like)       │
├─────────────────────────────────────────────────────────┤
│  3. SYSTEM PERFORMANCE MEASUREMENT                      │
│     • Retrieval time                                    │
│     • Prompt token count                                │
│     • LLM end-to-end latency (p50/p95/p99)             │
│     • Throughput                                        │
│     • (Future) vLLM KV-cache profiling                  │
│     • (Future) GenAI-Perf + CEBench integration         │
└─────────────────────────────────────────────────────────┘
```

### Components

**Retrieval Systems:**
- `VectorRetriever` - Dense vector search only (cosine similarity)
- `HybridRetriever` - BM25 (sparse) + Vector (dense) with configurable weighting

**RAG Pipeline:**
- Document chunking with overlap
- Embedding generation (SentenceTransformers)
- Prompt construction
- vLLM integration (HTTP API)

**Benchmarking:**
- Workload: 10 queries (3 semantic, 3 keyword, 4 hybrid)
- Metrics: latency percentiles, token counts, throughput
- Comparison: Vector vs Hybrid performance deltas

## Directory Structure

```
.
├── data/
│   ├── raw/
│   │   ├── sample_texts/          # Markdown documentation
│   │   ├── sample_code/           # Python, JavaScript files
│   │   └── sample_images/         # System diagrams, UI mockups
│   └── processed/
│       ├── chunks.json            # Chunked documents
│       └── embeddings/
│           ├── text_embeddings.npy
│           └── metadata.json
│
├── retrieval/
│   ├── vector_retriever.py        # Dense vector search
│   └── hybrid_retriever.py        # BM25 + vector hybrid
│
├── embeddings/
│   └── embed.py                   # SentenceTransformers + CLIP
│
├── rag/
│   ├── pipeline.py                # End-to-end RAG pipeline
│   └── prompt_builder.py          # Prompt construction
│
├── benchmark/
│   ├── workload.json              # 10 evaluation queries
│   ├── run_benchmark.py           # Benchmark runner
│   ├── metrics.py                 # Metrics calculation
│   └── results/                   # Output directory
│
├── utils/
│   ├── chunker.py                 # Document chunking
│   ├── file_loader.py             # Corpus loading
│   └── timer.py                   # Performance timing
│
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `sentence-transformers` - Text/code embeddings
- `rank-bm25` - BM25 sparse retrieval
- `numpy` - Numerical operations
- `requests` - vLLM HTTP client
- `torch` - PyTorch backend

### 2. Generate Corpus Embeddings

**Step 1: Load and chunk documents**

```bash
python -c "
from utils.file_loader import FileLoader
from utils.chunker import Chunker

loader = FileLoader('data/raw')
documents = loader.load_all_documents()
print(f'Loaded {len(documents)} documents')

chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_documents(documents)
chunker.save_chunks(chunks, 'data/processed/chunks.json')
print(f'Created {len(chunks)} chunks')
"
```

**Step 2: Generate embeddings**

```bash
python embeddings/embed.py data/processed/chunks.json data/processed/embeddings
```

This will:
- Load chunks from JSON
- Generate embeddings using `all-MiniLM-L6-v2`
- Save embeddings to `data/processed/embeddings/`

### 3. Start vLLM Server (Optional)

For full RAG pipeline testing, start a vLLM server:

```bash
# Example: Llama-3 8B
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B \
  --port 8000
```

**Note:** The benchmark can run without vLLM (retrieval-only mode) for initial testing.

## Usage

### Quick Test: Retrieval Only

**Vector retrieval:**
```bash
python retrieval/vector_retriever.py "How does authentication work?"
```

**Hybrid retrieval:**
```bash
python retrieval/hybrid_retriever.py "validateToken function" 0.5
```

### Full RAG Pipeline

```bash
python rag/pipeline.py "What is the cache hit ratio target?" hybrid
```

### Run Full Benchmark

**Compare vector vs hybrid:**
```bash
python benchmark/run_benchmark.py --methods vector hybrid --top-k 5 --alpha 0.5
```

**Parameters:**
- `--methods` - Retrieval methods to test (vector, hybrid)
- `--top-k` - Number of documents to retrieve (default: 5)
- `--alpha` - Hybrid BM25 weight (0.0 = vector only, 1.0 = BM25 only, 0.5 = balanced)
- `--vllm-url` - vLLM server URL (default: http://localhost:8000)

**Output:**
- JSON results: `benchmark/results/benchmark_results_<timestamp>.json`
- CSV metrics: `benchmark/results/benchmark_metrics_<timestamp>.csv`
- Console summary with latency percentiles and token statistics

## Benchmark Workload

The evaluation includes **10 carefully designed queries**:

### Semantic Queries (3)
- "How can I improve the performance of my application's data access layer?"
- "What security measures should I implement to protect user sessions?"
- "Explain the strategy for handling temporary data in a distributed system"

*Expected behavior: Vector retrieval should perform well (semantic understanding)*

### Keyword Queries (3)
- "validateToken"
- "AUTH_002"
- "allkeys-lru"

*Expected behavior: Hybrid retrieval should excel (BM25 lexical matching)*

### Hybrid Queries (4)
- "How does the JWT token expiration work in the authentication system?"
- "What is the cache hit ratio target and why is it important?"
- "Explain how the APIClient class handles retry logic with exponential backoff"
- "What are the connection pool settings for the database and what are the best practices?"

*Expected behavior: Demonstrates combined semantic + lexical retrieval*

## Metrics Collected

### Retrieval Metrics
- Retrieval time (ms)
- Number of documents retrieved
- Total tokens in retrieved context

### Generation Metrics
- Generation time (ms)
- Prompt tokens
- Generated tokens
- Total tokens

### Latency Percentiles
- p50, p95, p99 for retrieval, generation, and end-to-end latency

### Comparison Metrics
- Delta between vector and hybrid methods
- Percentage improvement/degradation
- Token reduction analysis

## Example Output

```
=== VECTOR RETRIEVAL METRICS ===
Total queries: 10

Retrieval Latency:
  P50: 12.34ms
  P95: 18.56ms
  Mean: 13.21ms

End-to-End Latency:
  P50: 456.78ms
  P95: 623.45ms

Token Statistics:
  Mean tokens/query: 1234.5

=== HYBRID RETRIEVAL METRICS ===
Total queries: 10

Retrieval Latency:
  P50: 15.67ms
  P95: 21.34ms

End-to-End Latency:
  P50: 398.12ms
  P95: 501.23ms

Token Statistics:
  Mean tokens/query: 987.3

VECTOR vs HYBRID COMPARISON
  ✓ Hybrid is 12.8% FASTER (p50 latency)
  ✓ Hybrid uses 20.0% FEWER tokens
```

## Future Extensions

This baseline framework is designed to be extended with:

### 1. Real-World Corpus
- Integrate SWE-Bench M dataset
- Add real GitHub issues, PRs, and screenshots
- Scale to 1000+ documents

### 2. Advanced Metrics
- vLLM KV-cache profiling (`/metrics` endpoint)
- Prefill vs decode latency breakdown
- Memory utilization tracking

### 3. Load Testing
- GenAI-Perf integration for concurrent queries
- Throughput measurement (queries/second)
- Cost-effectiveness analysis (CEBench)

### 4. Retrieval Quality Metrics
- Precision@k
- Mean Reciprocal Rank (MRR)
- NDCG
- Ground-truth relevance judgments

### 5. Multimodal Embeddings
- CLIP for image embeddings
- Visual question answering
- Diagram-to-code retrieval

## Design Principles

- **Modular:** Each component (chunker, retriever, pipeline) is independent
- **Minimal dependencies:** Only essential libraries, no heavy frameworks
- **Extensible:** Clear interfaces for adding new retrievers, metrics, workloads
- **Reproducible:** Fixed random seeds, versioned dependencies
- **Well-documented:** Comprehensive docstrings and examples

## Limitations (Current Baseline)

- Small synthetic corpus (5 documents)
- Simple chunking strategy (fixed size)
- Basic tokenization for BM25
- No GPU optimization
- No caching layer
- vLLM integration requires manual setup

## Contributing

To extend this framework:

1. **Add new retrieval methods:** Implement in `retrieval/` following the interface pattern
2. **Add new metrics:** Extend `MetricsCalculator` in `benchmark/metrics.py`
3. **Add new queries:** Update `benchmark/workload.json`
4. **Add new corpus:** Place files in `data/raw/` and re-run chunking/embedding

## References

- **SWE-Bench M:** Multimodal software engineering benchmark
- **RepoBench:** Repository-scale code completion benchmark
- **vLLM:** Fast LLM inference engine with KV-cache optimization
- **BM25:** Classic sparse retrieval algorithm
- **SentenceTransformers:** State-of-the-art text embeddings

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue.

---

**Built for CS854 - Model Serving**
**University of Waterloo - Fall 2025**
