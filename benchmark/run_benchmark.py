"""
Benchmark Runner
Executes benchmark workload and collects performance metrics.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Any
from rag.pipeline import RAGPipeline
from benchmark.metrics import MetricsCalculator
from utils.timer import Timer, PerformanceTracker
from vllm import LLM
import torch


def assert_gpu_available():
    """Ensure GPU is available for vLLM."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ No GPU detected! vLLM requires an NVIDIA GPU.\n"
            "This machine cannot run the benchmark.\n"
            "Please run this script on a CUDA-enabled GPU instance "
            "(e.g., Colab Pro, RunPod, AWS g5, Lambda GPU)."
        )

    device_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU detected: {device_name}")


class BenchmarkRunner:
    """Run benchmarks comparing vector vs hybrid retrieval."""

    def __init__(self,
                 vllm: LLM = None,
                 workload_file: str = 'benchmark/workload.json',
                 embeddings_dir: str = 'data/processed/embeddings',
                 chunks_file: str = 'data/processed/chunks.json',
                 results_dir: str = 'benchmark/results'):
        """
        Initialize benchmark runner.

        Args:
            vllm: Pre-initialized vLLM instance
            workload_file: Path to workload JSON file
            embeddings_dir: Directory with embeddings
            chunks_file: Path to chunks JSON
            results_dir: Directory to save results
        """

        # check_gpu_available() # asserting GPU availability. If no GPU, raise error and exit. Without GPU, model won't load anyway

        self.workload_file = Path(workload_file)
        self.embeddings_dir = embeddings_dir
        self.chunks_file = chunks_file
        if vllm is None:
            self.vllm = LLM(model="meta-llama/Meta-Llama-3-8B")
        else:
            self.vllm = vllm
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load workload
        with open(self.workload_file, 'r') as f:
            self.workload = json.load(f)

        self.queries = self.workload['queries']

        print(f"Loaded {len(self.queries)} queries from workload")

    def run_benchmark(self,
                     methods: List[str] = None,
                     top_k: int = 5,
                     alpha: float = 0.5) -> Dict[str, Any]:
        """
        Run full benchmark suite.

        Args:
            methods: List of methods to test ['vector', 'hybrid']
            top_k: Number of documents to retrieve
            alpha: Hybrid retrieval weight

        Returns:
            Dictionary with all results and comparisons
        """
        if methods is None:
            methods = ['vector', 'hybrid']

        print(f"\n{'='*60}")
        print(f"BENCHMARK EXECUTION")
        print(f"{'='*60}")
        print(f"Methods: {', '.join(methods)}")
        print(f"Queries: {len(self.queries)}")
        print(f"Top-k: {top_k}")
        if 'hybrid' in methods:
            print(f"Hybrid alpha: {alpha}")
        print(f"{'='*60}\n")

        results_by_method = {}

        for method in methods:
            print(f"\n--- Running {method.upper()} retrieval ---")

            # Initialize pipeline
            pipeline = RAGPipeline(
                retriever_type=method,
                embeddings_dir=self.embeddings_dir,
                chunks_file=self.chunks_file,
                vllm = self.vllm,
                top_k=top_k,
                alpha=alpha
            )

            # Run queries
            method_results = []
            for i, query_obj in enumerate(self.queries, 1):
                query = query_obj['query']
                query_id = query_obj['id']
                query_type = query_obj['type']

                print(f"[{i}/{len(self.queries)}] {query_id} ({query_type}): {query[:50]}...")

                try:
                    result = pipeline.query(query, max_tokens=256, temperature=0.7)
                    result['query_id'] = query_id
                    result['query_type'] = query_type
                    method_results.append(result)

                    # Print quick metrics
                    metrics = result['metrics']
                    print(f"  ✓ Retrieval: {metrics['retrieval_time_ms']:.1f}ms | "
                          f"Generation: {metrics['generation_time_ms']:.1f}ms | "
                          f"Tokens: {metrics['total_tokens']}")

                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    # Add error result
                    method_results.append({
                        'query': query,
                        'query_id': query_id,
                        'query_type': query_type,
                        'error': str(e),
                        'metrics': {
                            'retrieval_time_ms': 0,
                            'generation_time_ms': 0,
                            'total_time_ms': 0,
                            'prompt_tokens': 0,
                            'total_tokens': 0
                        }
                    })

            results_by_method[method] = method_results

        # Calculate metrics
        print("\n\n=== CALCULATING METRICS ===")
        aggregated_metrics = {}
        for method, results in results_by_method.items():
            aggregated_metrics[method] = MetricsCalculator.aggregate_metrics(
                results, method
            )

        # Compare methods if both vector and hybrid were run
        comparison = None
        if 'vector' in results_by_method and 'hybrid' in results_by_method:
            comparison = MetricsCalculator.compare_methods(
                results_by_method['vector'],
                results_by_method['hybrid']
            )

        # Compile final results
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'workload': self.workload['workload_name'],
            'config': {
                'methods': methods,
                'top_k': top_k,
                'alpha': alpha,
                'total_queries': len(self.queries)
            },
            'results_by_method': results_by_method,
            'aggregated_metrics': aggregated_metrics,
            'comparison': comparison
        }

        # Save results
        self._save_results(benchmark_results)

        # Print summary
        self._print_summary(benchmark_results)

        return benchmark_results

    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to JSON and CSV."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save full results as JSON
        json_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {json_file}")

        # Save metrics as CSV
        csv_file = self.results_dir / f"benchmark_metrics_{timestamp}.csv"
        self._save_csv(results, csv_file)
        print(f"✓ Saved metrics to {csv_file}")

    def _save_csv(self, results: Dict[str, Any], csv_file: Path):
        """Save metrics to CSV format."""
        rows = []

        for method, method_results in results['results_by_method'].items():
            for result in method_results:
                if 'error' in result:
                    continue

                metrics = result.get('metrics', {})
                row = {
                    'method': method,
                    'query_id': result.get('query_id', ''),
                    'query_type': result.get('query_type', ''),
                    'query': result.get('query', '')[:100],
                    'retrieval_time_ms': metrics.get('retrieval_time_ms', 0),
                    'generation_time_ms': metrics.get('generation_time_ms', 0),
                    'total_time_ms': metrics.get('total_time_ms', 0),
                    'prompt_tokens': metrics.get('prompt_tokens', 0),
                    'total_tokens': metrics.get('total_tokens', 0),
                    'num_retrieved': result.get('num_retrieved', 0)
                }
                rows.append(row)

        if rows:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    def _print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60 + "\n")

        # Print aggregated metrics for each method
        for method, metrics in results['aggregated_metrics'].items():
            print(MetricsCalculator.format_summary(metrics))
            print()

        # Print comparison if available
        if results['comparison']:
            self._print_comparison(results['comparison'])

    def _print_comparison(self, comparison: Dict[str, Any]):
        """Print method comparison."""
        print("\n" + "="*60)
        print("VECTOR vs HYBRID COMPARISON")
        print("="*60 + "\n")

        deltas = comparison['deltas']
        improvements = comparison['improvements']

        print("Latency Deltas (Hybrid - Vector):")
        print(f"  Retrieval P50: {deltas['retrieval_p50_delta_ms']:+.2f}ms "
              f"({improvements['retrieval_p50_improvement_pct']:+.1f}%)")
        print(f"  Retrieval P95: {deltas['retrieval_p95_delta_ms']:+.2f}ms")
        print(f"  Generation P50: {deltas['generation_p50_delta_ms']:+.2f}ms "
              f"({improvements['generation_p50_improvement_pct']:+.1f}%)")
        print(f"  Generation P95: {deltas['generation_p95_delta_ms']:+.2f}ms")
        print(f"  End-to-End P50: {deltas['end_to_end_p50_delta_ms']:+.2f}ms "
              f"({improvements['end_to_end_p50_improvement_pct']:+.1f}%)")
        print(f"  End-to-End P95: {deltas['end_to_end_p95_delta_ms']:+.2f}ms")
        print()
        print(f"Token Delta: {deltas['token_delta']:+.1f} "
              f"({improvements['token_reduction_pct']:+.1f}%)")
        print()

        # Interpretation
        print("Interpretation:")
        if deltas['end_to_end_p50_delta_ms'] < 0:
            print(f"  ✓ Hybrid is {abs(improvements['end_to_end_p50_improvement_pct']):.1f}% "
                  f"FASTER (p50 latency)")
        else:
            print(f"  ✗ Hybrid is {improvements['end_to_end_p50_improvement_pct']:.1f}% "
                  f"SLOWER (p50 latency)")

        if deltas['token_delta'] < 0:
            print(f"  ✓ Hybrid uses {abs(improvements['token_reduction_pct']):.1f}% "
                  f"FEWER tokens")
        else:
            print(f"  ✗ Hybrid uses {improvements['token_reduction_pct']:.1f}% "
                  f"MORE tokens")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run RAG benchmark')
    parser.add_argument('--methods', nargs='+', default=['vector', 'hybrid'],
                       choices=['vector', 'hybrid'],
                       help='Retrieval methods to benchmark')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of documents to retrieve')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Hybrid retrieval alpha (BM25 weight)')

    args = parser.parse_args()

    # Run benchmark
    runner = BenchmarkRunner()
    results = runner.run_benchmark(
        methods=args.methods,
        top_k=args.top_k,
        alpha=args.alpha
    )

    print("\n✓ Benchmark complete!")
