# Quick Start Guide: Qdrant Multimodal RAG

Get the Multimodal RAG system running in 5 minutes.

## 1. Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI Key (Required for VLM)
export OPENAI_API_KEY=sk-your-key-here
```

## 2. Ingest Data

Ingest a sample repository and the multimodal dataset.

```bash
# 1. Ingest Code (Grommet repo)
python ingest_code_to_qdrant.py \
  --repo_path /path/to/grommet \
  --repo_name grommet/grommet \
  --repo_version 2.10.0

# 2. Ingest Images (Test split, limit 100)
python ingest_images_to_qdrant.py --limit 100
```

## 3. Run Pipeline (Interactive)
The easiest way to run experiments is via the wizard:
```bash
python interactive_pipeline.py
```

## 4. Run Benchmark (Manual Fast Mode)

Run the 13-way benchmark with a limit and mock LLM to verify the pipeline.

```bash
python benchmark/run_experiments.py --limit 1 --mock
```

## 4. Run Full Benchmark

Run the full study (requires vLLM running).

```bash
python benchmark/run_experiments.py
```

## 5. Evaluate Results

Score the generated patches.

```bash
# Run evaluation for all results
for file in results/*_predictions.json; do
    exp_name=$(basename "$file" _predictions.json)
    echo "Running evaluation for $exp_name..."
    python -m swebench.harness.run_evaluation \
        --predictions_path "$file" \
        --dataset_name princeton-nlp/SWE-bench_Multimodal \
        --split dev \
        --run_id "eval_$exp_name"
done
```

## 6. Measure Recall

Calculate retrieval recall against the ground truth.

```bash
python benchmark/measure_recall.py
```
