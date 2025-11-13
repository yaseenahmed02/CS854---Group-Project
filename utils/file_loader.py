"""
File Loader
Loads documents from the raw corpus directory.
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path


class FileLoader:
    """Loads and parses files from the corpus."""

    def __init__(self, base_path: str):
        """
        Initialize file loader.

        Args:
            base_path: Base directory containing raw corpus files
        """
        self.base_path = Path(base_path)

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        Load all documents from raw corpus.

        Returns:
            List of document dictionaries with metadata
        """
        documents = []

        # Load text files
        text_dir = self.base_path / "sample_texts"
        if text_dir.exists():
            documents.extend(self._load_text_files(text_dir))

        # Load code files
        code_dir = self.base_path / "sample_code"
        if code_dir.exists():
            documents.extend(self._load_code_files(code_dir))

        # Load image metadata
        image_dir = self.base_path / "sample_images"
        if image_dir.exists():
            documents.extend(self._load_image_files(image_dir))

        return documents

    def _load_text_files(self, directory: Path) -> List[Dict[str, Any]]:
        """Load text/markdown files."""
        documents = []

        for file_path in directory.glob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                doc = {
                    'id': f"text_{file_path.stem}",
                    'text': content,
                    'path': str(file_path),
                    'type': 'text',
                    'modality': 'text',
                    'metadata': {
                        'filename': file_path.name,
                        'format': 'markdown',
                        'size': len(content)
                    }
                }
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return documents

    def _load_code_files(self, directory: Path) -> List[Dict[str, Any]]:
        """Load code files (Python, JavaScript, etc.)."""
        documents = []

        # Support common code file extensions
        extensions = ['*.py', '*.js', '*.java', '*.cpp', '*.go']

        for pattern in extensions:
            for file_path in directory.glob(pattern):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Determine language from extension
                    ext_to_lang = {
                        '.py': 'python',
                        '.js': 'javascript',
                        '.java': 'java',
                        '.cpp': 'cpp',
                        '.go': 'go'
                    }
                    language = ext_to_lang.get(file_path.suffix, 'unknown')

                    doc = {
                        'id': f"code_{file_path.stem}",
                        'text': content,
                        'path': str(file_path),
                        'type': 'code',
                        'modality': 'text',
                        'metadata': {
                            'filename': file_path.name,
                            'language': language,
                            'size': len(content)
                        }
                    }
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return documents

    def _load_image_files(self, directory: Path) -> List[Dict[str, Any]]:
        """
        Load image files (or image descriptions).

        Note: For baseline, we load image descriptions from README.
        Later, actual images will be embedded using CLIP.
        """
        documents = []

        # For now, load image descriptions from README
        readme_path = directory / "README.md"
        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse image descriptions
                # This is a simple parser - in production, use proper markdown parsing
                if "system_architecture.png" in content:
                    arch_desc = self._extract_description(content, "system_architecture.png")
                    documents.append({
                        'id': 'image_system_architecture',
                        'text': arch_desc,
                        'path': str(directory / 'system_architecture.png'),
                        'type': 'image',
                        'modality': 'image',
                        'metadata': {
                            'filename': 'system_architecture.png',
                            'description': arch_desc,
                            'is_placeholder': True
                        }
                    })

                if "dashboard_mockup.png" in content:
                    dash_desc = self._extract_description(content, "dashboard_mockup.png")
                    documents.append({
                        'id': 'image_dashboard_mockup',
                        'text': dash_desc,
                        'path': str(directory / 'dashboard_mockup.png'),
                        'type': 'image',
                        'modality': 'image',
                        'metadata': {
                            'filename': 'dashboard_mockup.png',
                            'description': dash_desc,
                            'is_placeholder': True
                        }
                    })
            except Exception as e:
                print(f"Error loading image descriptions: {e}")

        # TODO: Load actual PNG files when available and embed with CLIP
        # for file_path in directory.glob("*.png"):
        #     # Load and embed image using CLIP
        #     pass

        return documents

    def _extract_description(self, content: str, image_name: str) -> str:
        """Extract image description from README content."""
        lines = content.split('\n')
        description_lines = []
        capture = False

        for line in lines:
            if image_name in line:
                capture = True
                continue
            if capture:
                if line.startswith('###') or line.startswith('##'):
                    break
                if line.strip() and not line.startswith('#'):
                    description_lines.append(line.strip())

        return ' '.join(description_lines) if description_lines else f"Image: {image_name}"

    def get_document_by_id(self, doc_id: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Retrieve document by ID.

        Args:
            doc_id: Document ID
            documents: List of documents

        Returns:
            Document dictionary or None if not found
        """
        for doc in documents:
            if doc['id'] == doc_id:
                return doc
        return None

    def get_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get corpus statistics.

        Args:
            documents: List of documents

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_documents': len(documents),
            'by_type': {},
            'by_modality': {},
            'total_text_length': 0
        }

        for doc in documents:
            # Count by type
            doc_type = doc.get('type', 'unknown')
            stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1

            # Count by modality
            modality = doc.get('modality', 'unknown')
            stats['by_modality'][modality] = stats['by_modality'].get(modality, 0) + 1

            # Total text length
            if 'text' in doc:
                stats['total_text_length'] += len(doc['text'])

        return stats
