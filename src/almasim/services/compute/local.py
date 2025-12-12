"""Local computation backend (synchronous execution)."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Optional

from .base import ComputationBackend


def _execute_task(task: Any) -> Any:
    """Execute a single task (module-level function for pickling compatibility)."""
    if callable(task):
        return task()
    elif hasattr(task, "compute"):
        # Handle delayed objects
        return task.compute()
    else:
        return task


class DelayedTask:
    """Delayed task wrapper for local execution (module-level for pickling)."""

    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def compute(self):
        """Execute the task."""
        return self.func(*self.args, **self.kwargs)


class LocalBackend(ComputationBackend):
    """Local computation backend using processes.

    This backend executes computations locally using Python's
    ProcessPoolExecutor for true parallelism across CPU cores.
    Ideal for CPU-bound workloads like ALMA simulations.
    """

    def __init__(self, n_workers: Optional[int] = None):
        """Initialize local backend.

        Parameters
        ----------
        n_workers : int, optional
            Number of worker processes (default: CPU count)
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.executor: Any = None
        self._start_executor()

    def _start_executor(self) -> None:
        """Start the process executor."""
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)

    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data (no-op for local backend, returns data directly)."""
        return data

    def compute(self, tasks: Any, sync: bool = True) -> Any:
        """Compute tasks locally.

        For local backend, tasks are executed immediately.
        """
        if isinstance(tasks, list):
            if sync:
                return [_execute_task(task) for task in tasks]
            else:
                return [self.executor.submit(_execute_task, task) for task in tasks]
        else:
            if sync:
                return _execute_task(tasks)
            else:
                return self.executor.submit(_execute_task, tasks)

    def gather(self, futures: Any) -> List[Any]:
        """Gather results from futures."""
        if isinstance(futures, list):
            return [f.result() if hasattr(f, "result") else f for f in futures]
        else:
            return [futures.result() if hasattr(futures, "result") else futures]

    def delayed(self, func: Callable) -> Callable:
        """Create a delayed version of a function for local execution.

        Returns a decorator that wraps the function to create delayed tasks.
        """

        def delayed_decorator(*args, **kwargs):
            """Create a delayed task from function call."""
            return DelayedTask(func, *args, **kwargs)

        return delayed_decorator

    def close(self) -> None:
        """Close the executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
