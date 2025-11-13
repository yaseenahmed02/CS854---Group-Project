"""Setup script for RAG evaluation framework."""

from setuptools import setup, find_packages

setup(
    name="rag-eval-framework",
    version="0.1.0",
    description="Composite Evaluation Framework for Multimodal RAG",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "rank-bm25>=0.2.2",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
    ],
)
