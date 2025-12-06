import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
import numpy as np

def load_metrics(results_dir: str) -> pd.DataFrame:
    metrics_file = Path(results_dir) / "instance_metrics.json"
    if not metrics_file.exists():
        # Try checking swebench_predictions subdirectory
        metrics_file = Path(results_dir) / "swebench_predictions" / "instance_metrics.json"
        
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return pd.DataFrame()
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data)

def load_eval_reports(results_dir: str) -> pd.DataFrame:
    eval_dir = Path(results_dir) / "swebench_evaluation"
    if not eval_dir.exists():
        print(f"Evaluation directory not found: {eval_dir}")
        return pd.DataFrame()
    
    reports = []
    for file in eval_dir.glob("*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Extract experiment ID from filename (assuming format exp_id.eval_exp_id.json)
        # Or just use the one in the file if available? The file has lists of IDs.
        # Let's infer experiment_id from filename.
        # Filename: text_bm25.eval_text_bm25.json -> text_bm25
        exp_id = file.name.split('.')[0]
        
        # Create rows for each instance
        # We only know status for submitted instances
        
        # Helper to add rows
        def add_rows(ids, status):
            for instance_id in ids:
                reports.append({
                    "experiment_id": exp_id,
                    "instance_id": instance_id,
                    "status": status
                })
                
        add_rows(data.get('resolved_ids', []), 'Resolved')
        add_rows(data.get('unresolved_ids', []), 'Unresolved')
        add_rows(data.get('error_ids', []), 'Error')
        # We could also track 'submitted_ids' but status is more useful
        
    return pd.DataFrame(reports)

def generate_plots(metrics_df: pd.DataFrame, eval_df: pd.DataFrame, output_dir: str):
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Success Rate & Patch Apply Rate
    if not eval_df.empty:
        # Calculate rates per experiment
        summary = eval_df.groupby('experiment_id')['status'].value_counts(normalize=False).unstack(fill_value=0)
        summary['Total'] = summary.sum(axis=1)
        summary['Success Rate'] = (summary.get('Resolved', 0) / summary['Total']) * 100
        summary['Patch Apply Rate'] = ((summary.get('Resolved', 0) + summary.get('Unresolved', 0)) / summary['Total']) * 100
        
        # Plot Success Rate
        plt.figure(figsize=(10, 6))
        sns.barplot(x=summary.index, y=summary['Success Rate'])
        plt.title('Success Rate (% Resolved)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / "success_rate.png")
        plt.close()
        
        # Plot Patch Apply Rate
        plt.figure(figsize=(10, 6))
        sns.barplot(x=summary.index, y=summary['Patch Apply Rate'])
        plt.title('Patch Apply Rate (% Patches Applied Successfully)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / "patch_apply_rate.png")
        plt.close()

    # 2. Latency Analysis
    if not metrics_df.empty:
        # Filter for relevant columns
        latency_cols = ['retrieval_time_ms', 'generation_time_ms', 'total_io_time_ms']
        
        # Melt for plotting
        latency_df = metrics_df.melt(id_vars=['experiment_id'], value_vars=latency_cols, var_name='Metric', value_name='Time (ms)')
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=latency_df, x='experiment_id', y='Time (ms)', hue='Metric')
        plt.title('Latency Distribution by Experiment')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / "latency_distribution.png")
        plt.close()

    # 3. Token Usage
    if not metrics_df.empty:
        # Stacked Bar for Tokens
        # We want average tokens per experiment
        token_cols = ['issue_text_tokens', 'vlm_tokens', 'retrieved_tokens', 'output_generated_tokens']
        # Note: prompt_template_tokens might be missing in old data, handle gracefully
        if 'prompt_template_tokens' in metrics_df.columns:
            token_cols.insert(0, 'prompt_template_tokens')
            
        avg_tokens = metrics_df.groupby('experiment_id')[token_cols].mean()
        
        avg_tokens.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Average Token Usage per Experiment')
        plt.ylabel('Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / "token_usage.png")
        plt.close()

def generate_summary_md(metrics_df: pd.DataFrame, eval_df: pd.DataFrame, output_dir: str):
    summary_file = Path(output_dir) / "summary.md"
    
    with open(summary_file, 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        
        # Evaluation Metrics
        if not eval_df.empty:
            f.write("## Evaluation Performance\n\n")
            summary = eval_df.groupby('experiment_id')['status'].value_counts().unstack(fill_value=0)
            summary['Total'] = summary.sum(axis=1)
            summary['Success Rate (%)'] = ((summary.get('Resolved', 0) / summary['Total']) * 100).round(2)
            summary['Apply Rate (%)'] = (((summary.get('Resolved', 0) + summary.get('Unresolved', 0)) / summary['Total']) * 100).round(2)
            
            f.write(summary.to_markdown())
            f.write("\n\n")
            
        # Latency & Cost Metrics
        if not metrics_df.empty:
            f.write("## Latency Statistics (ms)\n\n")
            
            latency_cols = [
                'total_retrieval_time_ms',
                'generation_time_ms',
                'total_io_time_ms'
            ]
            # Filter cols that exist in metrics_df
            latency_cols = [c for c in latency_cols if c in metrics_df.columns]
            
            for col in latency_cols:
                f.write(f"### {col}\n\n")
                # Calculate stats
                stats = metrics_df.groupby('experiment_id')[col].agg(
                    Mean='mean',
                    P50='median',
                    P95=lambda x: x.quantile(0.95),
                    P99=lambda x: x.quantile(0.99),
                    Max='max'
                ).round(2)
                f.write(stats.to_markdown())
                f.write("\n\n")

            f.write("## Token Usage (Average)\n\n")
            token_cols = [
                'total_input_prompt_tokens',
                'output_generated_tokens'
            ]
            token_cols = [c for c in token_cols if c in metrics_df.columns]
            
            if token_cols:
                avg_tokens = metrics_df.groupby('experiment_id')[token_cols].mean().round(2)
                f.write(avg_tokens.to_markdown())
                f.write("\n\n")

        # Instance Statistics
        f.write("## Instance Statistics\n\n")
        try:
            # Load dataset (dev split for now, or try to infer/load both)
            dataset = load_dataset("princeton-nlp/SWE-bench_Multimodal", split="dev")
            
            # Filter dataset to only include instances in our results
            if not metrics_df.empty:
                result_ids = set(metrics_df['instance_id'].unique())
            elif not eval_df.empty:
                result_ids = set(eval_df['instance_id'].unique())
            else:
                result_ids = set()
                
            relevant_instances = [inst for inst in dataset if inst['instance_id'] in result_ids]
            
            if relevant_instances:
                stats = []
                for inst in relevant_instances:
                    repo = inst['repo']
                    version = inst['version']
                    
                    # Image count
                    images = inst.get('image_assets', [])
                    if isinstance(images, list):
                        num_images = len(images)
                    elif isinstance(images, dict):
                        # Flatten dict values
                        num_images = sum(len(v) if isinstance(v, list) else 1 for v in images.values())
                    else:
                        num_images = 1 if images else 0
                        
                    # Text length
                    text_len = len(inst['problem_statement'])
                    
                    stats.append({
                        'repo': repo,
                        'version': version,
                        'num_images': num_images,
                        'text_length': text_len
                    })
                
                stats_df = pd.DataFrame(stats)
                
                # Group by Repo/Version
                grouped = stats_df.groupby(['repo', 'version']).agg({
                    'num_images': ['count', 'mean'],
                    'text_length': ['mean', 'min', 'max']
                }).round(2)
                
                # Rename columns for clarity
                grouped.columns = ['# Instances', 'Avg Images', 'Avg Text Len', 'Min Text Len', 'Max Text Len']
                
                f.write(grouped.to_markdown())
                f.write("\n\n")
            else:
                f.write("No matching instances found in 'dev' split of SWE-bench Multimodal.\n")
                
        except Exception as e:
            f.write(f"Error calculating instance statistics: {e}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="Path to results directory")
    args = parser.parse_args()
    
    print(f"Analyzing results in {args.results_dir}...")
    
    metrics_df = load_metrics(args.results_dir)
    eval_df = load_eval_reports(args.results_dir)
    
    if metrics_df.empty and eval_df.empty:
        print("No data found to analyze.")
    else:
        generate_plots(metrics_df, eval_df, args.results_dir)
        generate_summary_md(metrics_df, eval_df, args.results_dir)
        print(f"Analysis complete. Check {args.results_dir}/summary.md and {args.results_dir}/plots/")
