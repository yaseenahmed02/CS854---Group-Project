#!/bin/bash

# Setup script for RAG Evaluation Framework

set -e  # Exit on error

echo "=========================================="
echo "RAG Evaluation Framework Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Generate corpus
echo ""
echo "=========================================="
echo "Preparing Corpus"
echo "=========================================="

# Step 1: Load and chunk documents
echo ""
echo "Step 1: Loading and chunking documents..."
python3 -c "
from utils.file_loader import FileLoader
from utils.chunker import Chunker

print('Loading documents...')
loader = FileLoader('data/raw')
documents = loader.load_all_documents()
print(f'✓ Loaded {len(documents)} documents')

# Get statistics
stats = loader.get_statistics(documents)
print(f'  - Text files: {stats[\"by_type\"].get(\"text\", 0)}')
print(f'  - Code files: {stats[\"by_type\"].get(\"code\", 0)}')
print(f'  - Images: {stats[\"by_type\"].get(\"image\", 0)}')

print('')
print('Chunking documents...')
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_documents(documents)
chunker.save_chunks(chunks, 'data/processed/chunks.json')
print(f'✓ Created {len(chunks)} chunks')

# Get chunk statistics
chunk_stats = chunker.get_chunk_statistics(chunks)
print(f'  - Average chunk size: {chunk_stats[\"avg_chunk_size\"]:.0f} chars')
print(f'  - Min chunk size: {chunk_stats[\"min_chunk_size\"]} chars')
print(f'  - Max chunk size: {chunk_stats[\"max_chunk_size\"]} chars')
"

# Step 2: Generate embeddings
echo ""
echo "Step 2: Generating embeddings..."
python3 embeddings/embed.py data/processed/chunks.json data/processed/embeddings

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test retrieval:"
echo "   python3 retrieval/vector_retriever.py 'How does authentication work?'"
echo "   python3 retrieval/hybrid_retriever.py 'validateToken function' 0.5"
echo ""
echo "2. (Optional) Start vLLM server for full pipeline:"
echo "   python -m vllm.entrypoints.openai.api_server --model <model-name> --port 8000"
echo ""
echo "3. Run benchmark:"
echo "   python3 benchmark/run_benchmark.py --methods vector hybrid"
echo ""
