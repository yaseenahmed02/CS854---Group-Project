import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict

# Requested order
EXPERIMENT_ORDER = [
    'text_bm25', 
    'text_bge', 
    'text_dense_jina', 
    'text_hybrid_jina_bge', 
    'multimodal_fusion_jina', 
    'multimodal_fusion_hybrid_jina_best_sparse'
]

def load_all_metrics(results_root: str) -> pd.DataFrame:
    all_metrics = []
    root_path = Path(results_root)
    
    # Iterate over all subdirectories
    for run_dir in root_path.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue
            
        # Check for metrics file in root or subdir
        metrics_file = run_dir / "instance_metrics.json"
        if not metrics_file.exists():
            metrics_file = run_dir / "swebench_predictions" / "instance_metrics.json"
            
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    # Add run_id for tracking
                    for item in data:
                        item['run_id'] = run_dir.name
                        all_metrics.append(item)
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")
                
    df = pd.DataFrame(all_metrics)
    if df.empty:
        return df
        
    # Deduplicate: keep latest timestamp for each (experiment_id, instance_id)
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=True)
        df = df.drop_duplicates(subset=['experiment_id', 'instance_id'], keep='last')
        
    return df

def load_all_eval_reports(results_root: str) -> pd.DataFrame:
    all_reports = []
    root_path = Path(results_root)
    
    for run_dir in root_path.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue
            
        eval_dir = run_dir / "swebench_evaluation"
        if not eval_dir.exists():
            continue
            
        for file in eval_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                exp_id = file.name.split('.')[0]
                
                def add_rows(ids, status):
                    for instance_id in ids:
                        all_reports.append({
                            "experiment_id": exp_id,
                            "instance_id": instance_id,
                            "status": status,
                            "run_id": run_dir.name
                        })
                        
                add_rows(data.get('resolved_ids', []), 'Resolved')
                add_rows(data.get('unresolved_ids', []), 'Unresolved')
                add_rows(data.get('error_ids', []), 'Error')
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
    df = pd.DataFrame(all_reports)
    if df.empty:
        return df
        
    # Deduplicate: We don't have timestamps in reports easily, but we can assume run_dir name is timestamp
    # Sort by run_id (assuming YYYY-MM-DD_HH-MM format sorts correctly)
    df = df.sort_values('run_id', ascending=True)
    df = df.drop_duplicates(subset=['experiment_id', 'instance_id'], keep='last')
    
    return df

def generate_aggregated_plots(metrics_df: pd.DataFrame, eval_df: pd.DataFrame, output_dir: str):
    plots_dir = Path(output_dir) / "aggregated_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # Filter for requested experiments only
    if not metrics_df.empty:
        metrics_df = metrics_df[metrics_df['experiment_id'].isin(EXPERIMENT_ORDER)]
    if not eval_df.empty:
        eval_df = eval_df[eval_df['experiment_id'].isin(EXPERIMENT_ORDER)]
        
    # 1. Success Rate
    if not eval_df.empty:
        summary = eval_df.groupby('experiment_id')['status'].value_counts(normalize=False).unstack(fill_value=0)
        summary['Total'] = summary.sum(axis=1)
        summary['Success Rate'] = (summary.get('Resolved', 0) / summary['Total']) * 100
        
        # Reindex to enforce order
        summary = summary.reindex(EXPERIMENT_ORDER)
        summary = summary.dropna(how='all').fillna(0) # Keep requested even if empty? Or drop?
        # Let's keep them to show 0 if missing
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=summary.index, y=summary['Success Rate'])
        plt.title('Success Rate (% Resolved)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / "success_rate.png")
        plt.close()
        
    # 2. Patch Apply Rate
    if not eval_df.empty:
        summary = eval_df.groupby('experiment_id')['status'].value_counts(normalize=False).unstack(fill_value=0)
        summary['Total'] = summary.sum(axis=1)
        summary['Patch Apply Rate'] = ((summary.get('Resolved', 0) + summary.get('Unresolved', 0)) / summary['Total']) * 100
        
        summary = summary.reindex(EXPERIMENT_ORDER).fillna(0)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=summary.index, y=summary['Patch Apply Rate'])
        plt.title('Patch Apply Rate (% Patches Applied Successfully)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / "patch_apply_rate.png")
        plt.close()

    # 3. Latency Distribution
    if not metrics_df.empty:
        latency_cols = ['total_retrieval_time_ms', 'generation_time_ms', 'total_io_time_ms']
        latency_df = metrics_df.melt(id_vars=['experiment_id'], value_vars=latency_cols, var_name='Metric', value_name='Time (ms)')
        
        # Enforce order in plot
        plt.figure(figsize=(14, 7))
        sns.boxplot(data=latency_df, x='experiment_id', y='Time (ms)', hue='Metric', order=EXPERIMENT_ORDER)
        plt.title('Latency Distribution by Experiment')
        plt.ylim(0, 30000)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / "latency_distribution.png")
        plt.close()

    # 4. Token Usage
    if not metrics_df.empty:
        token_cols = ['issue_text_tokens', 'vlm_tokens', 'retrieved_tokens', 'output_generated_tokens']
        if 'prompt_template_tokens' in metrics_df.columns:
            token_cols.insert(0, 'prompt_template_tokens')
            
        avg_tokens = metrics_df.groupby('experiment_id')[token_cols].mean()
        avg_tokens = avg_tokens.reindex(EXPERIMENT_ORDER).fillna(0)
        
        avg_tokens.plot(kind='bar', stacked=True, figsize=(14, 7))
        plt.title('Average Token Usage per Experiment')
        plt.ylabel('Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / "token_usage.png")
        plt.close()
        
    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    results_dir = "results"
    print(f"Aggregating results from {results_dir}...")
    
    metrics_df = load_all_metrics(results_dir)
    eval_df = load_all_eval_reports(results_dir)
    
    print(f"Loaded {len(metrics_df)} metric records and {len(eval_df)} evaluation records.")
    
    generate_aggregated_plots(metrics_df, eval_df, results_dir)
