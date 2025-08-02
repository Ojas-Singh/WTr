"""
Parallel execution utilities for WTr package.
"""

import concurrent.futures
import multiprocessing
from typing import List, Callable, Any, Optional
import functools
import time

class ProgressReporter:
    """Simple progress reporter for parallel tasks."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.completed = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.completed += n
        elapsed = time.time() - self.start_time
        
        if self.total > 0:
            progress = self.completed / self.total
            eta = elapsed / progress - elapsed if progress > 0 else 0
            
            print(f"\r{self.description}: {self.completed}/{self.total} "
                  f"({progress*100:.1f}%) - {elapsed:.1f}s elapsed, "
                  f"{eta:.1f}s remaining", end="", flush=True)
        else:
            print(f"\r{self.description}: {self.completed} completed - "
                  f"{elapsed:.1f}s elapsed", end="", flush=True)
    
    def finish(self):
        """Mark progress as finished."""
        elapsed = time.time() - self.start_time
        print(f"\n{self.description} completed in {elapsed:.1f}s")

def run_parallel(func: Callable, tasks: List[Any], max_workers: Optional[int] = None,
                description: str = "Processing") -> List[Any]:
    """
    Run function on tasks in parallel using ThreadPoolExecutor.
    
    Args:
        func: Function to apply to each task
        tasks: List of arguments to pass to func
        max_workers: Maximum number of worker threads
        description: Description for progress reporting
    
    Returns:
        List of results in same order as tasks
    """
    if max_workers is None:
        max_workers = min(len(tasks), multiprocessing.cpu_count())
    
    progress = ProgressReporter(len(tasks), description)
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(func, task): i for i, task in enumerate(tasks)}
        
        # Collect results as they complete
        result_dict = {}
        for future in concurrent.futures.as_completed(future_to_task):
            task_index = future_to_task[future]
            try:
                result = future.result()
                result_dict[task_index] = result
            except Exception as e:
                print(f"\nWarning: Task {task_index} failed with error: {e}")
                result_dict[task_index] = None
            
            progress.update(1)
        
        # Return results in original order
        results = [result_dict.get(i) for i in range(len(tasks))]
    
    progress.finish()
    return results

def timeout_wrapper(timeout_seconds: float):
    """
    Decorator to add timeout to function calls.
    
    Args:
        timeout_seconds: Maximum execution time
    
    Returns:
        Decorated function that raises TimeoutError if exceeded
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
        return wrapper
    return decorator

class WorkerPool:
    """
    Reusable worker pool for batch processing.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.executor = None
    
    def __enter__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit a task to the pool."""
        if not self.executor:
            raise RuntimeError("WorkerPool not initialized. Use with context manager.")
        return self.executor.submit(func, *args, **kwargs)
    
    def map(self, func: Callable, tasks: List[Any], description: str = "Processing") -> List[Any]:
        """Map function over tasks with progress reporting."""
        if not self.executor:
            raise RuntimeError("WorkerPool not initialized. Use with context manager.")
        
        progress = ProgressReporter(len(tasks), description)
        
        # Submit all tasks
        futures = [self.executor.submit(func, task) for task in tasks]
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"\nWarning: Task failed with error: {e}")
                results.append(None)
            
            progress.update(1)
        
        progress.finish()
        return results

# Example usage functions
def batch_evaluate_surfaces(surface_list: List, evaluation_func: Callable, 
                          max_workers: Optional[int] = None) -> List:
    """
    Evaluate multiple surfaces in parallel.
    
    Args:
        surface_list: List of surface configurations
        evaluation_func: Function to evaluate each surface
        max_workers: Maximum parallel workers
    
    Returns:
        List of evaluation results
    """
    return run_parallel(
        evaluation_func, 
        surface_list, 
        max_workers=max_workers,
        description="Evaluating surfaces"
    )

def batch_mc_sampling(surface_list: List, mc_func: Callable,
                     max_workers: Optional[int] = None) -> List:
    """
    Run MC sampling on multiple surfaces in parallel.
    
    Args:
        surface_list: List of initial surface configurations
        mc_func: MC sampling function
        max_workers: Maximum parallel workers
    
    Returns:
        List of refined surface configurations
    """
    return run_parallel(
        mc_func,
        surface_list,
        max_workers=max_workers, 
        description="MC sampling"
    )
