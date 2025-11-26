#!/bin/bash

# Run evaluation for all results
for file in results/swebench_predictions/*_predictions.json; do
    if [ ! -f "$file" ]; then
        echo "No prediction files found in results/swebench_predictions/"
        exit 1
    fi
    
    exp_name=$(basename "$file" _predictions.json)
    echo "---------------------------------------------------"
    echo "Running evaluation for $exp_name..."
    echo "---------------------------------------------------"
    
    mkdir -p results/swebench_evaluation
    
    python -m swebench.harness.run_evaluation \
        --predictions_path "$file" \
        --dataset_name princeton-nlp/SWE-bench_Multimodal \
        --split dev \
        --run_id "eval_$exp_name" \
        --report_dir results/swebench_evaluation

    # Move evaluation result file (swebench outputs to root by default sometimes)
    mv "${exp_name}.eval_${exp_name}.json" results/swebench_evaluation/ 2>/dev/null || true
        
    # Move logs
    if [ -d "logs/run_evaluation" ]; then
        mkdir -p results/swebench_evaluation/logs/run_evaluation
        cp -r logs/run_evaluation/* results/swebench_evaluation/logs/run_evaluation/ 2>/dev/null || true
        rm -rf logs/run_evaluation
    fi
done
