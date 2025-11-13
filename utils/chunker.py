"""
Chunker
Splits documents into smaller chunks for embedding and retrieval.
"""

import json
import re
from typing import List, Dict, Any
from pathlib import Path


class Chunker:
    """Chunks documents into smaller segments with metadata."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize chunker with size parameters.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk all documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of chunk dictionaries
        """
        all_chunks = []
        chunk_id_counter = 0

        for doc in documents:
            doc_chunks = self._chunk_single_document(doc, chunk_id_counter)
            all_chunks.extend(doc_chunks)
            chunk_id_counter += len(doc_chunks)

        return all_chunks

    def _chunk_single_document(self, document: Dict[str, Any],
                                start_id: int) -> List[Dict[str, Any]]:
        """
        Chunk a single document.

        Args:
            document: Document dictionary
            start_id: Starting chunk ID

        Returns:
            List of chunks from this document
        """
        text = document.get('text', '')
        doc_type = document.get('type', 'text')

        # Use appropriate chunking strategy based on document type
        if doc_type == 'code':
            chunks = self._chunk_code(text)
        elif doc_type == 'text':
            chunks = self._chunk_text(text)
        else:
            chunks = self._chunk_text(text)

        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk = {
                'chunk_id': start_id + i,
                'document_id': document['id'],
                'text': chunk_text,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'document_type': doc_type,
                'document_path': document.get('path', ''),
                'metadata': {
                    **document.get('metadata', {}),
                    'chunk_size': len(chunk_text),
                    'is_partial': len(chunks) > 1
                }
            }
            chunk_objects.append(chunk)

        return chunk_objects

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text/markdown content using paragraph-aware splitting.

        Args:
            text: Text content to chunk

        Returns:
            List of text chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)

                # If single paragraph is too large, split it
                if len(paragraph) > self.chunk_size:
                    chunks.extend(self._split_large_text(paragraph))
                    current_chunk = ""
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _chunk_code(self, code: str) -> List[str]:
        """
        Chunk code content using function/class-aware splitting.

        Args:
            code: Code content to chunk

        Returns:
            List of code chunks
        """
        # Split by function/class definitions (simple heuristic)
        # Looks for lines starting with 'def ', 'class ', 'function ', etc.
        split_pattern = r'(^(?:def |class |function |export (?:function|class)|public class).*$)'

        parts = re.split(split_pattern, code, flags=re.MULTILINE)

        chunks = []
        current_chunk = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(current_chunk) + len(part) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(part) > self.chunk_size:
                    chunks.extend(self._split_large_text(part))
                    current_chunk = ""
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    current_chunk += "\n\n" + part
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [code]

    def _split_large_text(self, text: str) -> List[str]:
        """
        Split large text that exceeds chunk size using sliding window.

        Args:
            text: Large text to split

        Returns:
            List of chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + len(chunk)

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return chunks

    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str):
        """
        Save chunks to JSON file.

        Args:
            chunks: List of chunk dictionaries
            output_path: Path to save chunks
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load chunks from JSON file.

        Args:
            input_path: Path to chunks file

        Returns:
            List of chunk dictionaries
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about chunks.

        Args:
            chunks: List of chunks

        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {}

        chunk_sizes = [len(c['text']) for c in chunks]

        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_text_length': sum(chunk_sizes),
            'by_document_type': {}
        }

        # Count by document type
        for chunk in chunks:
            doc_type = chunk.get('document_type', 'unknown')
            stats['by_document_type'][doc_type] = \
                stats['by_document_type'].get(doc_type, 0) + 1

        return stats
