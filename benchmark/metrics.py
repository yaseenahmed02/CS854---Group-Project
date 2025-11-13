"""
Metrics
Calculates and aggregates performance metrics for benchmark results.
"""

import numpy as np
from typing import List, Dict, Any


class MetricsCalculator:
    """Calculate performance metrics for retrieval and generation."""

    @staticmethod
    def calculate_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
        """
        Calculate latency percentiles.

        Args:
            latencies: List of latency values in milliseconds

        Returns:
            Dictionary with percentile metrics
        """
        if not latencies:
            return {}

        latencies_array = np.array(latencies)

        return {
            'p50_ms': float(np.percentile(latencies_array, 50)),
            'p95_ms': float(np.percentile(latencies_array, 95)),
            'p99_ms': float(np.percentile(latencies_array, 99)),
            'mean_ms': float(np.mean(latencies_array)),
            'min_ms': float(np.min(latencies_array)),
            'max_ms': float(np.max(latencies_array)),
            'std_ms': float(np.std(latencies_array))
        }

    @staticmethod
    def calculate_token_statistics(token_counts: List[int]) -> Dict[str, float]:
        """
        Calculate token count statistics.

        Args:
            token_counts: List of token counts

        Returns:
            Dictionary with token statistics
        """
        if not token_counts:
            return {}

        tokens_array = np.array(token_counts)

        return {
            'total_tokens': int(np.sum(tokens_array)),
            'mean_tokens': float(np.mean(tokens_array)),
            'min_tokens': int(np.min(tokens_array)),
            'max_tokens': int(np.max(tokens_array)),
            'median_tokens': float(np.median(tokens_array))
        }

    @staticmethod
    def calculate_throughput(total_requests: int,
                           total_time_seconds: float) -> float:
        """
        Calculate throughput in requests per second.

        Args:
            total_requests: Total number of requests
            total_time_seconds: Total time in seconds

        Returns:
            Throughput (requests/second)
        """
        if total_time_seconds <= 0:
            return 0.0
        return total_requests / total_time_seconds

    @staticmethod
    def aggregate_metrics(results: List[Dict[str, Any]],
                         method: str) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple query results.

        Args:
            results: List of query result dictionaries
            method: Retrieval method name

        Returns:
            Aggregated metrics dictionary
        """
        # Extract metrics
        retrieval_times = []
        generation_times = []
        total_times = []
        prompt_tokens = []
        total_tokens = []
        num_retrieved_docs = []

        for result in results:
            metrics = result.get('metrics', {})
            retrieval_times.append(metrics.get('retrieval_time_ms', 0))
            generation_times.append(metrics.get('generation_time_ms', 0))
            total_times.append(metrics.get('total_time_ms', 0))
            prompt_tokens.append(metrics.get('prompt_tokens', 0))
            total_tokens.append(metrics.get('total_tokens', 0))
            num_retrieved_docs.append(result.get('num_retrieved', 0))

        # Calculate aggregates
        aggregated = {
            'method': method,
            'total_queries': len(results),
            'retrieval_latency': MetricsCalculator.calculate_latency_percentiles(retrieval_times),
            'generation_latency': MetricsCalculator.calculate_latency_percentiles(generation_times),
            'end_to_end_latency': MetricsCalculator.calculate_latency_percentiles(total_times),
            'token_statistics': MetricsCalculator.calculate_token_statistics(total_tokens),
            'prompt_token_statistics': MetricsCalculator.calculate_token_statistics(prompt_tokens),
            'avg_docs_retrieved': float(np.mean(num_retrieved_docs)) if num_retrieved_docs else 0,
            'total_retrieval_time_ms': sum(retrieval_times),
            'total_generation_time_ms': sum(generation_times)
        }

        return aggregated

    @staticmethod
    def compare_methods(vector_results: List[Dict[str, Any]],
                       hybrid_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare metrics between vector and hybrid retrieval methods.

        Args:
            vector_results: Results from vector retrieval
            hybrid_results: Results from hybrid retrieval

        Returns:
            Comparison dictionary with deltas and improvements
        """
        vector_metrics = MetricsCalculator.aggregate_metrics(vector_results, 'vector')
        hybrid_metrics = MetricsCalculator.aggregate_metrics(hybrid_results, 'hybrid')

        # Calculate improvements (negative = hybrid is better/faster)
        comparison = {
            'vector_metrics': vector_metrics,
            'hybrid_metrics': hybrid_metrics,
            'deltas': {
                'retrieval_p50_delta_ms': (
                    hybrid_metrics['retrieval_latency']['p50_ms'] -
                    vector_metrics['retrieval_latency']['p50_ms']
                ),
                'retrieval_p95_delta_ms': (
                    hybrid_metrics['retrieval_latency']['p95_ms'] -
                    vector_metrics['retrieval_latency']['p95_ms']
                ),
                'generation_p50_delta_ms': (
                    hybrid_metrics['generation_latency']['p50_ms'] -
                    vector_metrics['generation_latency']['p50_ms']
                ),
                'generation_p95_delta_ms': (
                    hybrid_metrics['generation_latency']['p95_ms'] -
                    vector_metrics['generation_latency']['p95_ms']
                ),
                'end_to_end_p50_delta_ms': (
                    hybrid_metrics['end_to_end_latency']['p50_ms'] -
                    vector_metrics['end_to_end_latency']['p50_ms']
                ),
                'end_to_end_p95_delta_ms': (
                    hybrid_metrics['end_to_end_latency']['p95_ms'] -
                    vector_metrics['end_to_end_latency']['p95_ms']
                ),
                'token_delta': (
                    hybrid_metrics['token_statistics']['mean_tokens'] -
                    vector_metrics['token_statistics']['mean_tokens']
                )
            }
        }

        # Calculate percentage improvements
        def safe_percent_change(new_val, old_val):
            if old_val == 0:
                return 0.0
            return ((new_val - old_val) / old_val) * 100

        comparison['improvements'] = {
            'retrieval_p50_improvement_pct': safe_percent_change(
                hybrid_metrics['retrieval_latency']['p50_ms'],
                vector_metrics['retrieval_latency']['p50_ms']
            ),
            'generation_p50_improvement_pct': safe_percent_change(
                hybrid_metrics['generation_latency']['p50_ms'],
                vector_metrics['generation_latency']['p50_ms']
            ),
            'end_to_end_p50_improvement_pct': safe_percent_change(
                hybrid_metrics['end_to_end_latency']['p50_ms'],
                vector_metrics['end_to_end_latency']['p50_ms']
            ),
            'token_reduction_pct': safe_percent_change(
                hybrid_metrics['token_statistics']['mean_tokens'],
                vector_metrics['token_statistics']['mean_tokens']
            )
        }

        return comparison

    @staticmethod
    def format_summary(metrics: Dict[str, Any]) -> str:
        """
        Format metrics as human-readable summary.

        Args:
            metrics: Metrics dictionary

        Returns:
            Formatted string summary
        """
        lines = [
            f"=== {metrics['method'].upper()} RETRIEVAL METRICS ===",
            f"Total queries: {metrics['total_queries']}",
            "",
            "Retrieval Latency:",
            f"  P50: {metrics['retrieval_latency']['p50_ms']:.2f}ms",
            f"  P95: {metrics['retrieval_latency']['p95_ms']:.2f}ms",
            f"  P99: {metrics['retrieval_latency']['p99_ms']:.2f}ms",
            f"  Mean: {metrics['retrieval_latency']['mean_ms']:.2f}ms",
            "",
            "Generation Latency:",
            f"  P50: {metrics['generation_latency']['p50_ms']:.2f}ms",
            f"  P95: {metrics['generation_latency']['p95_ms']:.2f}ms",
            f"  P99: {metrics['generation_latency']['p99_ms']:.2f}ms",
            f"  Mean: {metrics['generation_latency']['mean_ms']:.2f}ms",
            "",
            "End-to-End Latency:",
            f"  P50: {metrics['end_to_end_latency']['p50_ms']:.2f}ms",
            f"  P95: {metrics['end_to_end_latency']['p95_ms']:.2f}ms",
            f"  P99: {metrics['end_to_end_latency']['p99_ms']:.2f}ms",
            f"  Mean: {metrics['end_to_end_latency']['mean_ms']:.2f}ms",
            "",
            "Token Statistics:",
            f"  Total tokens: {metrics['token_statistics']['total_tokens']}",
            f"  Mean tokens/query: {metrics['token_statistics']['mean_tokens']:.1f}",
            f"  Min tokens: {metrics['token_statistics']['min_tokens']}",
            f"  Max tokens: {metrics['token_statistics']['max_tokens']}",
            "",
            f"Average documents retrieved: {metrics['avg_docs_retrieved']:.1f}"
        ]

        return "\n".join(lines)
