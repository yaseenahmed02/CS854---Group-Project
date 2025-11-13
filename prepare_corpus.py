#!/usr/bin/env python3
"""
Prepare Corpus Script
Loads, chunks, and embeds the corpus in one command.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("CORPUS PREPARATION PIPELINE")
    print("="*60)

    # Step 1: Load documents
    print("\n[1/3] Loading documents...")
    from utils.file_loader import FileLoader

    loader = FileLoader('data/raw')
    documents = loader.load_all_documents()

    print(f"✓ Loaded {len(documents)} documents")
    stats = loader.get_statistics(documents)
    print(f"  - Text: {stats['by_type'].get('text', 0)}")
    print(f"  - Code: {stats['by_type'].get('code', 0)}")
    print(f"  - Images: {stats['by_type'].get('image', 0)}")
    print(f"  - Total text length: {stats['total_text_length']:,} chars")

    # Step 2: Chunk documents
    print("\n[2/3] Chunking documents...")
    from utils.chunker import Chunker

    chunker = Chunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)

    print(f"✓ Created {len(chunks)} chunks")
    chunk_stats = chunker.get_chunk_statistics(chunks)
    print(f"  - Average chunk size: {chunk_stats['avg_chunk_size']:.0f} chars")
    print(f"  - Min: {chunk_stats['min_chunk_size']}, Max: {chunk_stats['max_chunk_size']}")

    # Save chunks
    chunks_file = 'data/processed/chunks.json'
    chunker.save_chunks(chunks, chunks_file)
    print(f"  - Saved to {chunks_file}")

    # Step 3: Generate embeddings
    print("\n[3/3] Generating embeddings...")
    from embeddings.embed import embed_corpus

    embed_corpus(
        chunks_file='data/processed/chunks.json',
        output_dir='data/processed/embeddings',
        model_name='all-MiniLM-L6-v2'
    )

    print("\n" + "="*60)
    print("✓ CORPUS PREPARATION COMPLETE!")
    print("="*60)
    print("\nYou can now run:")
    print("  python benchmark/run_benchmark.py")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
