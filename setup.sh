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

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Run the interactive pipeline:"
echo "   python3 interactive_pipeline.py"
echo ""
echo "2. Run SWE-bench evaluation:"
echo "   ./evaluate_all.sh"
echo ""
