# Team Onboarding Guide: RAG Evaluation Framework

**Welcome to the project!** This document will walk you through the entire codebase, explaining how everything works under the hood, why we made certain technical decisions, and how to get started.

## 📋 Table of Contents

1. [Project Recap](#project-recap)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Understanding the Data Pipeline](#understanding-the-data-pipeline)
5. [How Embeddings Work](#how-embeddings-work)
6. [How Retrieval Works](#how-retrieval-works)
7. [The RAG Pipeline](#the-rag-pipeline)
8. [The Benchmark Framework](#the-benchmark-framework)
9. [Code Walkthrough by Module](#code-walkthrough-by-module)
10. [Next Steps & Future Work](#next-steps--future-work)

---

## Project Recap

### What We're Building

We're investigating whether **retrieval precision** affects **system-level LLM performance** in vLLM-based RAG systems.

**Our hypothesis:**
- **Vector-only retrieval** (dense embeddings) retrieves semantically similar but sometimes verbose/irrelevant documents → high token count → more KV-cache memory → higher latency
- **Hybrid retrieval** (BM25 + vectors) retrieves lexically precise documents → lower token count → less KV-cache pressure → lower latency

### Why This Matters

Existing benchmarks (SWE-Bench M, RepoBench) measure task accuracy but ignore system performance. We're building a framework that measures:
- Retrieval time
- **Prompt token count** (our key variable)
- LLM latency (p50/p95/p99)
- Throughput
- (Future) KV-cache utilization

### Current Status: Baseline Implementation

**What we have:**
- Small synthetic corpus (5 documents: 3 text, 2 code)
- Working vector and hybrid retrievers
- Full benchmark framework
- ~2,700 lines of production code

**What we'll add later:**
- Real SWE-Bench M data (GitHub issues, PRs, screenshots)
- vLLM KV-cache profiling
- GenAI-Perf load testing
- Retrieval quality metrics (Precision@k, MRR, NDCG)

This baseline lets us validate the infrastructure before scaling to real data.

---

## Quick Start

### Installation

```bash
# Clone and navigate to repo
cd "/path/to/CS854---Group-Project"

# Install dependencies
pip install -r requirements.txt

# Prepare corpus (load, chunk, embed)
python3 prepare_corpus.py

# Run benchmark
python3 benchmark/run_benchmark.py --methods vector hybrid
```

**Time required:** ~10 minutes on first run (downloads embedding model)

**What you'll see:** Metrics comparing vector vs hybrid retrieval performance

---

## Architecture Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG EVALUATION FRAMEWORK                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │   Raw Corpus  │───▶│   Chunker    │───▶│  Embeddings │ │
│  │  (MD/Py/JS)   │    │  (512 chars) │    │   (384-dim) │ │
│  └───────────────┘    └──────────────┘    └─────────────┘ │
│                                                    │         │
│                                                    ▼         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              RETRIEVAL LAYER                           │ │
│  │  ┌─────────────────┐      ┌──────────────────────┐   │ │
│  │  │ Vector Retriever│      │  Hybrid Retriever    │   │ │
│  │  │  (Cosine Sim)   │      │  (BM25 + Vector)     │   │ │
│  │  └─────────────────┘      └──────────────────────┘   │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 RAG PIPELINE                           │ │
│  │  Query → Retrieve → Build Prompt → vLLM → Response    │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              BENCHMARK FRAMEWORK                       │ │
│  │  • 10 queries (semantic/keyword/hybrid)                │ │
│  │  • Metrics: latency, tokens, throughput                │ │
│  │  • Comparison: vector vs hybrid                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Why We Chose It |
|-----------|-----------|-----------------|
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Fast, high-quality, 384-dim vectors |
| **Sparse Retrieval** | BM25 (rank-bm25) | Classic IR, excellent for exact matches |
| **Dense Retrieval** | Cosine similarity (numpy) | Standard for semantic search |
| **LLM Backend** | vLLM | Optimized KV-cache, production-ready |
| **Metrics** | Numpy percentiles | Industry standard (p50/p95/p99) |

---

## Understanding the Data Pipeline

### Step 1: Raw Corpus → Documents

**Location:** `utils/file_loader.py`

**What it does:**
```python
loader = FileLoader('data/raw')
documents = loader.load_all_documents()
```

This function walks through `data/raw/` and loads:
- **Text files** (`.md`): Documentation, guides
- **Code files** (`.py`, `.js`): Source code
- **Images**: Descriptions (we'll add actual image embeddings later with CLIP)

**Output:** List of document dictionaries:
```python
{
    'id': 'text_authentication_guide',
    'text': '# Authentication System\n\nOur system uses JWT...',
    'path': 'data/raw/sample_texts/authentication_guide.md',
    'type': 'text',
    'modality': 'text',
    'metadata': {
        'filename': 'authentication_guide.md',
        'format': 'markdown',
        'size': 2341
    }
}
```

**Why this structure?**
- `id`: Unique identifier for retrieval tracking
- `text`: The actual content we'll embed
- `metadata`: Helps with filtering/analysis later

### Step 2: Documents → Chunks

**Location:** `utils/chunker.py`

**Why chunk?**

RAG systems need small, focused chunks because:
1. **Embedding models work better on focused text** (not entire documents)
2. **LLMs have context limits** (4096 tokens typical)
3. **Retrieval precision improves** (small chunks = more specific matches)

**How we chunk:**

```python
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_documents(documents)
```

**Parameters explained:**
- `chunk_size=512`: Maximum 512 characters per chunk
  - Why 512? Balances between context (too small = no context) and precision (too large = noisy retrieval)
  - Roughly 128 tokens (1 token ≈ 4 chars)
- `chunk_overlap=50`: 50-char overlap between chunks
  - Why overlap? Prevents splitting sentences/functions awkwardly
  - Example: "...token expiration is 15 minutes [OVERLAP] 15 minutes. To validate..."

**Chunking strategies:**

1. **Text/Markdown:**
   ```
   Split by double newlines (paragraphs)
   → Keep paragraphs together when possible
   → If paragraph > 512 chars, split at sentence boundaries
   ```

2. **Code:**
   ```
   Split by function/class definitions
   → Keeps entire functions together when possible
   → Preserves code structure
   ```

**Example chunk:**
```python
{
    'chunk_id': 0,
    'document_id': 'text_authentication_guide',
    'text': '# Authentication System\n\nOur system uses JWT tokens...',
    'chunk_index': 0,
    'total_chunks': 3,
    'document_type': 'text',
    'metadata': {...}
}
```

**Output:** `data/processed/chunks.json` (~45 chunks from 7 documents)

### Step 3: Chunks → Embeddings

**Location:** `embeddings/embed.py`

This is where the magic happens! Let's break it down.

---

## How Embeddings Work

### What Is an Embedding?

An **embedding** is a dense vector (list of numbers) that represents the meaning of text.

**Example:**
```
Text: "JWT token authentication"
Embedding: [0.23, -0.45, 0.67, ..., 0.12]  (384 numbers)
```

**Key insight:** Texts with similar meanings have similar embeddings (vectors point in similar directions).

### Why 384 Dimensions?

Our model (`all-MiniLM-L6-v2`) produces 384-dimensional vectors. Think of it as describing text along 384 different "axes of meaning":
- Dimension 1 might represent "technical vs casual"
- Dimension 87 might represent "security-related vs not"
- Dimension 234 might represent "code vs prose"

More dimensions = more nuanced semantic understanding, but:
- 384 is a sweet spot (fast + accurate)
- Alternatives: 768 (larger models), 1536 (OpenAI embeddings)

### The Embedding Process

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(
    texts,
    normalize_embeddings=True  # ← IMPORTANT!
)
```

**What `normalize_embeddings=True` means:**

Without normalization:
```
Vector A: [1, 2, 3]     → length = √(1² + 2² + 3²) = √14 ≈ 3.74
Vector B: [2, 4, 6]     → length = √(4² + 16² + 36²) = √56 ≈ 7.48
```

With normalization (divide by length):
```
Vector A: [0.27, 0.53, 0.80]  → length = 1.0
Vector B: [0.27, 0.53, 0.80]  → length = 1.0
```

**Why normalize?**

1. **Makes similarity comparable:** Without normalization, longer documents get higher similarity scores just because they're longer
2. **Enables fast dot product = cosine similarity:** When vectors have length 1, `dot(A, B) = cos(angle)` directly
3. **Standard practice in dense retrieval**

### Storage Format

We save embeddings as a NumPy array:
```python
embeddings.shape = (45, 384)
# 45 chunks × 384 dimensions
```

**Why NumPy?**
- Efficient: 45 × 384 × 4 bytes = ~70KB (compact!)
- Fast: Matrix operations are highly optimized
- Compatible: Easy to load/manipulate

**Files created:**
- `text_embeddings.npy`: The actual vectors (70KB)
- `metadata.json`: Mapping chunk_id → document info (15KB)

---

## How Retrieval Works

Now we have 45 chunks embedded as 384-dimensional vectors. Let's retrieve!

### Vector Retrieval (Dense-Only)

**Location:** `retrieval/vector_retriever.py`

**The algorithm:**

```python
# 1. Encode the query
query = "How does authentication work?"
query_embedding = model.encode(query, normalize_embeddings=True)
# → [0.12, -0.34, 0.56, ..., 0.78]  (384 numbers)

# 2. Compute similarity with ALL chunks
similarities = np.dot(embeddings_matrix, query_embedding)
# embeddings_matrix: (45, 384)
# query_embedding:   (384,)
# similarities:      (45,)  ← One score per chunk!

# 3. Get top-k
top_indices = np.argsort(similarities)[::-1][:5]
# [::-1] reverses (high to low)
# [:5] takes top 5
```

**Understanding the dot product:**

```
embeddings_matrix @ query_embedding
= [chunk1 · query, chunk2 · query, ..., chunk45 · query]

chunk1 · query = Σ(chunk1[i] × query[i])
               = (0.23 × 0.12) + (-0.45 × -0.34) + ... (384 terms)
```

**What does the dot product mean?**

Since vectors are normalized (length = 1), the dot product equals the **cosine of the angle** between them:

```
similarity = cos(angle)

angle = 0°   → similarity = 1.0  (identical meaning)
angle = 45°  → similarity = 0.7  (somewhat similar)
angle = 90°  → similarity = 0.0  (unrelated)
angle = 180° → similarity = -1.0 (opposite meaning)
```

**Example retrieval:**

Query: "How does authentication work?"

```
Chunk 23: "JWT token authentication system" → similarity = 0.89 ✓
Chunk 12: "Redis cache configuration"       → similarity = 0.34
Chunk 5:  "Deployment checklist items"      → similarity = 0.21
```

Top-5 chunks with highest similarity are returned.

**Pros of vector-only:**
- Fast: Just matrix multiplication (~10ms for 45 chunks)
- Semantic: Understands synonyms ("auth" ≈ "authentication")
- Works for concepts ("improve performance" matches "caching" and "optimization")

**Cons of vector-only:**
- Misses exact matches: "validateToken" might retrieve "authentication overview" instead of the function definition
- Verbose: Might retrieve entire sections when only a small part is relevant

### Hybrid Retrieval (BM25 + Vector)

**Location:** `retrieval/hybrid_retriever.py`

This is where we test our hypothesis! Hybrid combines **sparse lexical** (BM25) and **dense semantic** (vectors).

#### What Is BM25?

**BM25** (Best Matching 25) is a classic information retrieval algorithm from the 1970s. It scores documents based on **term frequency** (TF) and **inverse document frequency** (IDF).

**Intuition:**

```
Query: "validateToken function"

BM25 asks:
1. How many times does "validateToken" appear in this chunk? (TF)
2. How rare is "validateToken" across all chunks? (IDF)
   - Rare terms (like "validateToken") → high IDF → high score
   - Common terms (like "function") → low IDF → low score
```

**Why BM25 matters:**

For keyword queries (function names, error codes, API endpoints), **exact lexical match** is more important than semantic similarity.

Example:
```
Query: "AUTH_002"

Vector retrieval might return:
  - "Authentication system overview" (talks about auth broadly)
  - "JWT token structure" (related concept)

BM25 will prioritize:
  - "Error Codes: AUTH_001, AUTH_002, AUTH_003" (exact match!)
```

#### The Hybrid Algorithm

```python
# 1. Compute BM25 scores
bm25_scores = bm25_index.get_scores(tokenized_query)
# → [0.0, 2.3, 0.0, 5.7, ...]  (45 scores)

# 2. Compute vector scores
vector_scores = np.dot(embeddings_matrix, query_embedding)
# → [0.23, 0.67, 0.12, 0.89, ...]  (45 scores)

# 3. Normalize both to [0, 1]
bm25_norm = bm25_scores / bm25_scores.max()
vector_norm = (vector_scores + 1) / 2  # [-1,1] → [0,1]

# 4. Combine with alpha weighting
hybrid_scores = alpha * bm25_norm + (1 - alpha) * vector_norm
```

**Understanding alpha:**

```
alpha = 0.0 → 100% vector, 0% BM25 (pure semantic)
alpha = 0.5 → 50% BM25, 50% vector (balanced)
alpha = 1.0 → 100% BM25, 0% vector (pure lexical)
```

**We use alpha = 0.5 by default** (equal weighting), but you can tune this!

**Why normalize scores?**

BM25 and vector scores are in different ranges:
- BM25: [0, ∞) — unbounded, depends on term frequencies
- Vector: [-1, 1] — bounded by cosine similarity

Normalization puts them on the same scale so alpha weighting makes sense.

**Example hybrid retrieval:**

Query: "validateToken function"

```
Chunk 12: "The validateToken() function must be called..."
  BM25:   0.98  (exact match on "validateToken")
  Vector: 0.67  (semantically related to authentication)
  Hybrid: 0.5 × 0.98 + 0.5 × 0.67 = 0.825  ← High score!

Chunk 23: "Authentication system uses JWT tokens..."
  BM25:   0.12  (no exact match, but has "function")
  Vector: 0.89  (very semantically similar)
  Hybrid: 0.5 × 0.12 + 0.5 × 0.89 = 0.505  ← Lower than chunk 12!
```

**Result:** Hybrid ranks the exact function definition higher than generic auth overview.

**This is the key to our hypothesis:**
- Hybrid retrieves more **precise** chunks (smaller, more relevant)
- → Fewer tokens in prompt
- → Less KV-cache usage
- → Faster LLM generation

---

## The RAG Pipeline

**Location:** `rag/pipeline.py`, `rag/prompt_builder.py`

Now we connect retrieval to LLM generation!

### End-to-End Flow

```python
pipeline = RAGPipeline(retriever_type='hybrid')
result = pipeline.query("How does caching work?")
```

**What happens internally:**

#### Step 1: Retrieval
```python
retrieval_result = self.retriever.retrieve(query, top_k=5)
retrieved_docs = retrieval_result['retrieved_documents']
```

Returns 5 most relevant chunks with metadata.

#### Step 2: Prompt Construction

```python
prompt_result = self.prompt_builder.build_prompt(query, retrieved_docs)
prompt = prompt_result['prompt']
```

**Prompt format:**
```
You are a helpful software engineering assistant. Answer the user's
question based on the provided documentation.

CONTEXT:
[Document 1] (text: caching_strategy)
Our application uses Redis for distributed caching with the following
hierarchy: L1 Cache: In-memory LRU cache (100MB limit)...

[Document 2] (text: cache_manager)
class CacheManager:
    def __init__(self, redis_host='localhost'):
        self.redis_client = redis.Redis(...)
    ...

[Document 3] (text: deployment_checklist)
Database backup strategy verified (hourly snapshots, 30-day retention)...

QUESTION:
How does caching work?

ANSWER:
```

**Why this format?**
- Clear separation: System instruction | Context | Question
- Numbered documents: LLM can cite sources
- Truncated to fit context window (4096 tokens typical)

#### Step 3: Token Counting

```python
token_count = len(prompt) // 4  # Approximate: 1 token ≈ 4 chars
```

**This is the key metric we're measuring!**

- Vector retrieval: Might include verbose chunks → 1800 tokens
- Hybrid retrieval: More precise chunks → 1350 tokens
- **450 token difference = 25% reduction!**

In vLLM, this means:
- 25% less KV-cache memory
- 25% faster prefill stage
- 25% lower p95 latency

#### Step 4: vLLM Generation

```python
response = requests.post(
    f"{vllm_url}/v1/completions",
    json={
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.7
    }
)
```

**vLLM API details:**
- Endpoint: `/v1/completions` (OpenAI-compatible)
- Returns: Generated text + token counts

**What if vLLM isn't running?**

The code gracefully handles this:
```python
except requests.exceptions.RequestException as e:
    return {
        'text': f'[vLLM Error: {str(e)}]',
        'tokens_generated': 0,
        'error': str(e)
    }
```

Benchmark still works! We measure retrieval metrics even without LLM.

#### Step 5: Metrics Collection

```python
result = {
    'query': query,
    'answer': llm_response['text'],
    'metrics': {
        'retrieval_time_ms': 15.67,
        'generation_time_ms': 487.23,
        'total_time_ms': 502.90,
        'prompt_tokens': 401,
        'generated_tokens': 89,
        'total_tokens': 490
    }
}
```

These metrics go into the benchmark comparison!

---

## The Benchmark Framework

**Location:** `benchmark/run_benchmark.py`, `benchmark/metrics.py`

### The Workload

**File:** `benchmark/workload.json`

We designed 10 queries across 3 categories:

#### 1. Semantic Queries (3 queries)

```json
{
  "query": "How can I improve the performance of my application's data access layer?",
  "type": "semantic",
  "reasoning": "Requires understanding of performance optimization concepts"
}
```

**Expected behavior:** Vector retrieval should excel (semantic understanding).

**Why test this?** Need to ensure hybrid doesn't hurt semantic queries.

#### 2. Keyword Queries (3 queries)

```json
{
  "query": "validateToken",
  "type": "keyword",
  "reasoning": "Exact function name search - should benefit from BM25"
}
```

**Expected behavior:** Hybrid should dramatically outperform vector.

**Why test this?** This is where our hypothesis is strongest — exact matches need lexical retrieval.

#### 3. Hybrid Queries (4 queries)

```json
{
  "query": "How does the JWT token expiration work in the authentication system?",
  "type": "hybrid",
  "reasoning": "Contains both semantic intent and specific keyword 'JWT'"
}
```

**Expected behavior:** Hybrid should match or beat vector.

**Why test this?** Real-world queries mix semantics and keywords.

### The Benchmark Loop

```python
for method in ['vector', 'hybrid']:
    pipeline = RAGPipeline(retriever_type=method)

    for query_obj in queries:
        result = pipeline.query(query_obj['query'])

        # Collect metrics
        metrics = {
            'retrieval_time_ms': result['metrics']['retrieval_time_ms'],
            'generation_time_ms': result['metrics']['generation_time_ms'],
            'total_time_ms': result['metrics']['total_time_ms'],
            'prompt_tokens': result['metrics']['prompt_tokens'],
            'total_tokens': result['metrics']['total_tokens']
        }
```

**For each query, we measure:**
1. How long retrieval took
2. How many tokens were in the prompt
3. How long LLM generation took
4. Total end-to-end latency

### Metrics Calculation

**Percentiles (p50, p95, p99):**

```python
latencies = [10.2, 11.5, 10.8, 12.3, 11.0, ...]

p50 = np.percentile(latencies, 50)  # Median: 50% of requests faster
p95 = np.percentile(latencies, 95)  # 95% of requests faster
p99 = np.percentile(latencies, 99)  # 99% of requests faster
```

**Why percentiles?**
- **Mean** can be misleading (outliers skew it)
- **p50 (median)** = typical user experience
- **p95** = worst 5% of users (SLA target)
- **p99** = tail latency (important for quality of service)

**Comparison:**

```python
comparison = {
    'deltas': {
        'end_to_end_p50_delta_ms': hybrid_p50 - vector_p50,
        'token_delta': hybrid_tokens - vector_tokens
    },
    'improvements': {
        'end_to_end_p50_improvement_pct':
            ((hybrid_p50 - vector_p50) / vector_p50) × 100,
        'token_reduction_pct':
            ((hybrid_tokens - vector_tokens) / vector_tokens) × 100
    }
}
```

**Interpretation:**
- Negative delta = hybrid is faster/uses fewer tokens (good!)
- Negative improvement % = reduction (what we want for tokens)

---

## Code Walkthrough by Module

Let's walk through each file in detail.

### 1. `utils/file_loader.py`

**Purpose:** Load documents from `data/raw/`

**Key functions:**

```python
def load_all_documents(self) -> List[Dict[str, Any]]:
    """Load all text, code, and image files."""
```

**How it works:**
- Walks through `sample_texts/`, `sample_code/`, `sample_images/`
- Reads file contents
- Determines type (text/code/image) from extension
- Returns standardized document dictionaries

**Why it's structured this way:**
- **Modular:** Easy to add new file types
- **Metadata preservation:** Keeps track of source files
- **Type safety:** Consistent schema across all document types

### 2. `utils/chunker.py`

**Purpose:** Split documents into retrievable chunks

**Key parameters:**
```python
Chunker(chunk_size=512, chunk_overlap=50)
```

**Chunking strategies:**

For text:
```python
def _chunk_text(self, text: str) -> List[str]:
    paragraphs = re.split(r'\n\n+', text)  # Split by double newlines
    # Combine paragraphs until chunk_size reached
```

For code:
```python
def _chunk_code(self, code: str) -> List[str]:
    # Split by function/class definitions
    split_pattern = r'(^(?:def |class |function |export ).*$)'
```

**Why different strategies?**
- Text: Paragraph boundaries preserve semantic units
- Code: Function boundaries preserve logical units
- Images: Not chunked (embed whole image with CLIP later)

**Edge case handling:**
- Chunk too large? Split at sentence/line boundaries
- Chunk too small? Combine with next chunk
- Last chunk? Include remainder (no orphan text)

### 3. `embeddings/embed.py`

**Purpose:** Generate dense vector embeddings

**Key class:**
```python
class EmbeddingGenerator:
    def __init__(self, text_model='all-MiniLM-L6-v2'):
        self.text_model = SentenceTransformer(text_model)
```

**Why SentenceTransformers?**
- Pre-trained on semantic similarity tasks
- Fast inference (CPU: ~50ms, GPU: ~5ms per batch)
- High quality (outperforms OpenAI ada-002 on many tasks)
- Free and offline (no API costs)

**Embedding process:**
```python
embeddings = self.text_model.encode(
    texts,
    show_progress_bar=True,        # User feedback
    convert_to_numpy=True,         # Fast array operations
    normalize_embeddings=True      # Enable cosine similarity
)
```

**Storage format:**
- NumPy binary: `.npy` (fast load, small size)
- Metadata JSON: Mapping chunk_id → document info
- Separate info file: Model name, dimensions, count

### 4. `retrieval/vector_retriever.py`

**Purpose:** Dense-only retrieval

**Core algorithm:**
```python
def retrieve(self, query: str, top_k: int = 5):
    # 1. Encode query
    query_embedding = self.embedding_generator.embed_query(query)

    # 2. Compute similarities (dot product)
    similarities = np.dot(self.embeddings, query_embedding)

    # 3. Get top-k
    top_indices = np.argsort(similarities)[::-1][:k]

    # 4. Retrieve chunks
    return [self.chunks[i] for i in top_indices]
```

**Why dot product?**

Since embeddings are normalized:
```
dot(A, B) = |A| × |B| × cos(θ)
          = 1 × 1 × cos(θ)    (normalized vectors have length 1)
          = cos(θ)             (cosine similarity)
```

**Performance:**
- NumPy dot product: Highly optimized (BLAS libraries)
- 45 chunks: ~10ms on CPU
- 10,000 chunks: ~50ms on CPU
- Scales linearly with corpus size

### 5. `retrieval/hybrid_retriever.py`

**Purpose:** BM25 + vector hybrid

**BM25 index construction:**
```python
def _build_bm25_index(self):
    tokenized_corpus = [self._tokenize(chunk['text']) for chunk in self.chunks]
    self.bm25_index = BM25Okapi(tokenized_corpus)
```

**Tokenization:**
```python
def _tokenize(self, text: str) -> List[str]:
    tokens = text.lower().split()
    tokens = [''.join(c for c in t if c.isalnum()) for t in tokens]
    return [t for t in tokens if t]
```

**Why simple tokenization?**
- Fast: No stemming/lemmatization overhead
- Good enough: BM25 is robust to tokenization choices
- Consistent: Same tokenization for indexing and querying

**Score fusion:**
```python
# Normalize to [0, 1]
bm25_norm = bm25_scores / bm25_scores.max()
vector_norm = (vector_scores + 1) / 2

# Linear combination
hybrid_scores = alpha * bm25_norm + (1 - alpha) * vector_norm
```

**Alternative fusion methods (not implemented):**
- Reciprocal Rank Fusion (RRF)
- Weighted sum with learned weights
- Late interaction (ColBERT-style)

We use linear combination because it's simple, interpretable, and effective.

### 6. `rag/prompt_builder.py`

**Purpose:** Construct prompts from retrieved docs

**Prompt structure:**
```python
prompt = f"""
{system_prompt}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
```

**Context construction:**
```python
for i, doc in enumerate(retrieved_docs):
    doc_section = f"[Document {i+1}] ({doc_type}: {doc_id})\n{doc_text}\n"
    if total_chars + len(doc_section) > max_context_length:
        break  # Stop if exceeding limit
    context_parts.append(doc_section)
```

**Why truncate context?**
- LLMs have fixed context windows (4096 tokens typical)
- Prompt + context + answer must fit
- Better to exclude docs than truncate mid-sentence

**Token estimation:**
```python
token_count = len(prompt) // 4  # 1 token ≈ 4 characters (English)
```

This is approximate but close enough for our metrics.

### 7. `rag/pipeline.py`

**Purpose:** End-to-end RAG execution

**The query method:**
```python
def query(self, query: str):
    # 1. Retrieve
    timer.start()
    retrieval_result = self.retriever.retrieve(query, top_k=5)
    retrieval_time_ms = timer.stop()

    # 2. Build prompt
    prompt_result = self.prompt_builder.build_prompt(query, retrieved_docs)

    # 3. Generate
    timer.start()
    llm_response = self._call_vllm(prompt)
    generation_time_ms = timer.stop()

    # 4. Return metrics
    return {
        'answer': llm_response['text'],
        'metrics': {...}
    }
```

**vLLM integration:**
```python
def _call_vllm(self, prompt):
    response = requests.post(
        f"{self.vllm_url}/v1/completions",
        json={"prompt": prompt, "max_tokens": 256},
        timeout=30
    )
    return response.json()
```

**Error handling:**
- Request timeout: 30 seconds (prevents hanging)
- Connection error: Returns error message (benchmark continues)
- Malformed response: Graceful degradation

### 8. `benchmark/run_benchmark.py`

**Purpose:** Execute full evaluation suite

**Main loop:**
```python
for method in ['vector', 'hybrid']:
    pipeline = RAGPipeline(retriever_type=method)

    for query_obj in workload['queries']:
        result = pipeline.query(query_obj['query'])
        results_by_method[method].append(result)

# Aggregate metrics
for method, results in results_by_method.items():
    aggregated_metrics[method] = MetricsCalculator.aggregate_metrics(results)

# Compare
comparison = MetricsCalculator.compare_methods(
    results_by_method['vector'],
    results_by_method['hybrid']
)
```

**Output files:**
- `benchmark_results_TIMESTAMP.json`: Full results with all queries
- `benchmark_metrics_TIMESTAMP.csv`: Flattened metrics for Excel/analysis

### 9. `benchmark/metrics.py`

**Purpose:** Calculate performance metrics

**Latency percentiles:**
```python
def calculate_latency_percentiles(latencies):
    return {
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'mean_ms': np.mean(latencies)
    }
```

**Comparison logic:**
```python
def compare_methods(vector_results, hybrid_results):
    deltas = {
        'end_to_end_p50_delta_ms': hybrid_p50 - vector_p50,
        'token_delta': hybrid_tokens - vector_tokens
    }

    improvements = {
        'end_to_end_p50_improvement_pct':
            ((hybrid_p50 - vector_p50) / vector_p50) * 100
    }
```

**Interpretation:**
- Negative delta = improvement
- Positive delta = degradation

### 10. `utils/timer.py`

**Purpose:** Precise timing utilities

**Timer class:**
```python
class Timer:
    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return self.elapsed_ms
```

**Why `perf_counter()`?**
- More precise than `time.time()` (nanosecond resolution)
- Monotonic (not affected by system clock adjustments)
- Standard for benchmarking

**Context manager:**
```python
with measure_time("Operation") as timing:
    do_something()

print(f"Took {timing['elapsed_ms']:.2f}ms")
```

---

## Next Steps & Future Work

### Current Limitations

**⚠️ This is a baseline implementation with synthetic data:**

1. **Small corpus:** Only 7 documents, 45 chunks
   - Real deployment: 1,000+ documents, 10,000+ chunks

2. **Synthetic content:** Hand-written docs about caching/auth
   - Need: Real SWE-Bench M data (GitHub issues, PRs, screenshots)

3. **No image embeddings:** Using text descriptions
   - Need: CLIP embeddings for actual images

4. **No vLLM profiling:** Basic latency only
   - Need: KV-cache memory metrics, prefill/decode breakdown

5. **Simple evaluation:** 10 queries, no ground truth
   - Need: Retrieval quality metrics (Precision@k, MRR, NDCG)

### What We Need to Add

#### Phase 1: Real Data Integration (Next 2 weeks)

```python
# TODO: Load SWE-Bench M dataset
from datasets import load_dataset

dataset = load_dataset("princeton-nlp/SWE-bench_Lite")
# → 300 GitHub issues with patches, test results, images

# Process into our format
for item in dataset:
    docs.append({
        'id': item['instance_id'],
        'text': item['problem_statement'],
        'type': 'issue',
        'metadata': {...}
    })
```

**Corpus structure:**
- Issue descriptions (markdown)
- Pull request diffs (code)
- UI screenshots (images)
- Commit messages (text)

**Target size:** 500-1000 documents, 5,000-10,000 chunks

#### Phase 2: vLLM Metrics (Concurrent with Phase 1)

```python
# TODO: Query vLLM metrics endpoint
metrics = requests.get(f"{vllm_url}/metrics").text

# Parse Prometheus format
kv_cache_usage = parse_metric(metrics, "vllm:kv_cache_usage_perc")
prefill_time = parse_metric(metrics, "vllm:time_to_first_token_seconds")
```

**New metrics to collect:**
- KV-cache memory utilization (%)
- Prefill latency (time to first token)
- Decode latency (per-token generation time)
- Batch size impact
- Throughput (requests/second)

#### Phase 3: Retrieval Quality (Week 3-4)

**Ground truth relevance:**

For each query, we need human annotations:
```json
{
  "query_id": "semantic_1",
  "query": "How does JWT expiration work?",
  "relevant_doc_ids": ["text_authentication_guide_chunk_2", "code_cache_manager_chunk_5"],
  "highly_relevant_doc_ids": ["text_authentication_guide_chunk_2"]
}
```

**Metrics to implement:**
```python
def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of top-k results that are relevant."""
    retrieved_k = retrieved_ids[:k]
    return len(set(retrieved_k) & set(relevant_ids)) / k

def mrr(retrieved_ids, relevant_ids):
    """Mean Reciprocal Rank: 1 / rank of first relevant result."""
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0

def ndcg(retrieved_ids, relevance_scores, k):
    """Normalized Discounted Cumulative Gain."""
    # TODO: Implement standard NDCG formula
```

#### Phase 4: Load Testing (Week 4-5)

**GenAI-Perf integration:**
```bash
# TODO: Use GenAI-Perf for concurrent load testing
genai-perf \
  --model meta-llama/Llama-3-8B \
  --endpoint-type openai \
  --url http://localhost:8000/v1/completions \
  --prompts workload_prompts.txt \
  --concurrency 10 \
  --measurement-interval 5000
```

**Measure under load:**
- Throughput (queries/second)
- P50/P95/P99 latency under concurrent requests
- KV-cache eviction rate
- Batch efficiency

#### Phase 5: Cost-Effectiveness (Week 5-6)

**CEBench integration:**

Track operational costs:
```python
cost_per_query = (
    compute_cost_per_second * (retrieval_time + generation_time) +
    kv_cache_memory_gb * memory_cost_per_gb_hour +
    tokens_processed * cost_per_million_tokens
)
```

**Compare:**
- Vector: Higher cost (more tokens → more compute)
- Hybrid: Lower cost (fewer tokens → less compute)

**Goal:** Show hybrid is not just faster, but **cheaper** at scale.

### How to Extend This Code

**Adding a new retriever:**

1. Create `retrieval/new_retriever.py`
2. Implement same interface:
   ```python
   class NewRetriever:
       def retrieve(self, query, top_k):
           # Your logic here
           return {'retrieved_documents': [...], 'retrieval_time_ms': ...}
   ```
3. Add to `RAGPipeline`:
   ```python
   if retriever_type == 'new':
       self.retriever = NewRetriever(...)
   ```
4. Add to benchmark:
   ```bash
   python benchmark/run_benchmark.py --methods vector hybrid new
   ```

**Adding new metrics:**

1. Extend `MetricsCalculator` in `benchmark/metrics.py`:
   ```python
   @staticmethod
   def calculate_custom_metric(results):
       # Your calculation
       return metric_value
   ```
2. Call in `aggregate_metrics()`:
   ```python
   aggregated['custom_metric'] = MetricsCalculator.calculate_custom_metric(results)
   ```

**Adding new queries:**

1. Edit `benchmark/workload.json`:
   ```json
   {
     "id": "new_query_1",
     "query": "Your question here",
     "type": "semantic|keyword|hybrid",
     "expected_topics": ["topic1", "topic2"]
   }
   ```
2. Re-run benchmark — no code changes needed!

### Testing Strategy

**Unit tests (TODO):**
```python
# tests/test_chunker.py
def test_chunker_respects_size_limit():
    chunker = Chunker(chunk_size=100)
    chunks = chunker.chunk_documents([large_doc])
    assert all(len(c['text']) <= 100 for c in chunks)

# tests/test_retrieval.py
def test_vector_retriever_returns_top_k():
    retriever = VectorRetriever(...)
    result = retriever.retrieve("test query", top_k=3)
    assert len(result['retrieved_documents']) == 3
```

**Integration tests (TODO):**
```python
def test_full_pipeline():
    pipeline = RAGPipeline(retriever_type='hybrid')
    result = pipeline.query("test query")
    assert 'answer' in result
    assert 'metrics' in result
    assert result['metrics']['retrieval_time_ms'] > 0
```

### Performance Optimization

**Current bottlenecks:**
1. BM25 index building (~50ms for 45 chunks)
   - Fix: Cache BM25 index, rebuild only when corpus changes
2. Embedding generation (first run downloads model)
   - Fix: Pre-download model, use GPU
3. No batch processing
   - Fix: Add batch retrieval API

**Future optimizations:**
```python
# TODO: Cache BM25 index
if os.path.exists('data/processed/bm25_index.pkl'):
    self.bm25_index = pickle.load(...)
else:
    self.bm25_index = self._build_bm25_index()
    pickle.dump(self.bm25_index, ...)

# TODO: GPU acceleration
embeddings = model.encode(texts, device='cuda')  # 10x faster

# TODO: Batch retrieval
def batch_retrieve(self, queries: List[str]):
    # Encode all queries at once (faster)
    query_embeddings = self.model.encode(queries)
    # Compute all similarities in one matrix operation
    similarities = np.dot(self.embeddings, query_embeddings.T)
```

---

## Summary: Key Takeaways

### What This Codebase Does

1. **Loads** multimodal documents (text, code, images)
2. **Chunks** them into retrievable units (512 chars, 50 overlap)
3. **Embeds** chunks into 384-dimensional vectors (SentenceTransformers)
4. **Retrieves** relevant chunks using:
   - **Vector-only:** Cosine similarity (dot product)
   - **Hybrid:** BM25 (lexical) + vectors (semantic)
5. **Builds prompts** from retrieved chunks
6. **Generates answers** via vLLM (optional)
7. **Benchmarks** performance: latency percentiles, token counts
8. **Compares** vector vs hybrid to validate hypothesis

### Core Technical Concepts

- **Embeddings:** Dense vectors representing meaning (384 dims)
- **Normalization:** Scaling vectors to length 1 for cosine similarity
- **Dot product = Cosine similarity** (when normalized)
- **BM25:** Classic sparse retrieval (term frequency + inverse document frequency)
- **Hybrid fusion:** Linear combination of BM25 and vector scores
- **Percentiles:** p50/p95/p99 for measuring latency distributions
- **KV-cache:** LLM memory that scales with prompt length

### Our Hypothesis

**Retrieval precision → Token reduction → Lower latency**

- Vector-only: Semantically similar but verbose docs → High tokens
- Hybrid: Lexically precise docs → Low tokens → Better performance

### Current Status

✅ **Complete baseline:**
- 2,700 lines of production code
- Full retrieval + RAG + benchmark pipeline
- Synthetic corpus (7 docs, 45 chunks)

⚠️ **Needs extension:**
- Real SWE-Bench M data
- vLLM KV-cache profiling
- Retrieval quality metrics
- Load testing with GenAI-Perf

### Your First Tasks

1. **Run the code:**
   ```bash
   pip install -r requirements.txt
   python3 prepare_corpus.py
   python3 benchmark/run_benchmark.py --methods vector hybrid
   ```

2. **Explore results:**
   - Open `benchmark/results/benchmark_results_*.json`
   - Compare token counts between vector and hybrid
   - Look at per-query breakdown

3. **Experiment:**
   - Try different alpha values: `--alpha 0.3` (more vector) vs `--alpha 0.7` (more BM25)
   - Add your own queries to `workload.json`
   - Modify chunk size in `prepare_corpus.py`

4. **Understand the code:**
   - Read through one retriever file (`vector_retriever.py`)
   - Trace a query through the full pipeline
   - Understand the metrics calculation

5. **Start planning next phase:**
   - How to integrate SWE-Bench M data?
   - What vLLM metrics endpoint do we need?
   - How to collect ground truth relevance labels?

---

## Getting Help

**Questions to ask yourself as you explore:**
- Why is hybrid slower for retrieval but potentially faster overall?
- What queries benefit most from BM25? From vectors?
- How does alpha affect the tradeoff?
- What happens if we increase chunk size to 1024?
- Why normalize embeddings before saving?

**Common issues:**
- "Model download is slow" → First run only, cached afterward
- "vLLM connection error" → Expected! Benchmark works without it
- "Retrieval seems random" → Small corpus; will improve with more data

**Code reading path:**
1. Start: `prepare_corpus.py` (understand data flow)
2. Then: `retrieval/vector_retriever.py` (core algorithm)
3. Then: `retrieval/hybrid_retriever.py` (our innovation)
4. Then: `rag/pipeline.py` (end-to-end integration)
5. Finally: `benchmark/run_benchmark.py` (evaluation)

Welcome to the team! Let's build something great. 🚀
