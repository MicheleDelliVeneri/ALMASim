"""In-process synchronous computation backend."""

from typing import Any, Callable, List

from .base import ComputationBackend


def _compute_value(task: Any) -> Any:
    """Evaluate a task immediately in the current process."""
    if hasattr(task, "compute"):
        return task.compute(scheduler="synchronous")
    if callable(task):
        return task()
    return task


class SyncBackend(ComputationBackend):
    """Computation backend that runs tasks synchronously in-process.

    This backend is useful for notebooks, examples, and debugging because it
    avoids process pools and networked schedulers entirely.
    """

    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Return data directly without scattering."""
        return data

    def compute(self, tasks: Any, sync: bool = True) -> Any:
        """Compute tasks immediately in the current process."""
        if isinstance(tasks, list):
            return [_compute_value(task) for task in tasks]
        return _compute_value(tasks)

    def gather(self, futures: Any) -> List[Any]:
        """Gather already-computed results."""
        if isinstance(futures, list):
            return futures
        return [futures]

    def delayed(self, func: Callable) -> Callable:
        """Create a delayed wrapper compatible with the backend interface."""

        def delayed_decorator(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return delayed_decorator

    def close(self) -> None:
        """No-op close for synchronous backend."""
        return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
