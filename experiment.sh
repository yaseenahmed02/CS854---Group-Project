# Define the path to your venv python
PYTHON_CMD="./venv/bin/python3"

for file in results/consolidated/*_predictions.json; do
  echo "Running evaluation for $(basename "$file")..."
  
  # use the variable $PYTHON_CMD instead of just 'python3'
  $PYTHON_CMD -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Multimodal \
    --split dev \
    --predictions_path "$file" \
    --max_workers 4 \
    --run_id "$(basename "$file" _predictions.json)"
done