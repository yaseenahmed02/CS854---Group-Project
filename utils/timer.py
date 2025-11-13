"""
Timer
Simple timing utilities for performance measurement.
"""

import time
from typing import Optional
from contextlib import contextmanager


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        """Initialize timer."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed_ms = None

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in milliseconds
        """
        if self.start_time is None:
            raise ValueError("Timer was not started")

        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return self.elapsed_ms

    def get_elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds
        """
        if self.elapsed_ms is None:
            raise ValueError("Timer was not stopped")
        return self.elapsed_ms

    def get_elapsed_seconds(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            Elapsed time in seconds
        """
        return self.get_elapsed_ms() / 1000

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None


@contextmanager
def measure_time(operation_name: str = "Operation", verbose: bool = True):
    """
    Context manager for measuring execution time.

    Args:
        operation_name: Name of the operation being timed
        verbose: Whether to print timing information

    Yields:
        Dictionary with timing results

    Example:
        with measure_time("Database query") as timing:
            result = db.query(...)
        print(f"Query took {timing['elapsed_ms']:.2f}ms")
    """
    timing = {'elapsed_ms': 0, 'elapsed_seconds': 0}
    start = time.perf_counter()

    if verbose:
        print(f"[Timer] {operation_name} started...")

    try:
        yield timing
    finally:
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        elapsed_seconds = elapsed_ms / 1000

        timing['elapsed_ms'] = elapsed_ms
        timing['elapsed_seconds'] = elapsed_seconds

        if verbose:
            print(f"[Timer] {operation_name} completed in {elapsed_ms:.2f}ms "
                  f"({elapsed_seconds:.3f}s)")


def time_function(func):
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that prints timing information

    Example:
        @time_function
        def slow_operation():
            time.sleep(1)
    """
    def wrapper(*args, **kwargs):
        timer = Timer()
        timer.start()

        result = func(*args, **kwargs)

        elapsed = timer.stop()
        print(f"[Timer] {func.__name__} took {elapsed:.2f}ms")

        return result

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class PerformanceTracker:
    """Track multiple timing measurements for analysis."""

    def __init__(self):
        """Initialize performance tracker."""
        self.measurements = []

    def add_measurement(self, operation: str, elapsed_ms: float, metadata: dict = None):
        """
        Add a timing measurement.

        Args:
            operation: Operation name
            elapsed_ms: Elapsed time in milliseconds
            metadata: Optional metadata dictionary
        """
        measurement = {
            'operation': operation,
            'elapsed_ms': elapsed_ms,
            'elapsed_seconds': elapsed_ms / 1000,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self.measurements.append(measurement)

    def get_summary(self) -> dict:
        """
        Get summary statistics of all measurements.

        Returns:
            Dictionary with summary statistics
        """
        if not self.measurements:
            return {}

        times_by_operation = {}
        for m in self.measurements:
            op = m['operation']
            if op not in times_by_operation:
                times_by_operation[op] = []
            times_by_operation[op].append(m['elapsed_ms'])

        summary = {
            'total_measurements': len(self.measurements),
            'by_operation': {}
        }

        for op, times in times_by_operation.items():
            summary['by_operation'][op] = {
                'count': len(times),
                'total_ms': sum(times),
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'p50_ms': self._percentile(times, 50),
                'p95_ms': self._percentile(times, 95),
                'p99_ms': self._percentile(times, 99)
            }

        return summary

    @staticmethod
    def _percentile(data: list, percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            fraction = index - int(index)
            return lower + (upper - lower) * fraction

    def clear(self):
        """Clear all measurements."""
        self.measurements = []

    def export_measurements(self) -> list:
        """
        Export all measurements.

        Returns:
            List of measurement dictionaries
        """
        return self.measurements.copy()
