"""In-memory store for simulation status and logs."""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class SimulationStatus:
    """Simulation status information."""

    simulation_id: str
    status: str = "queued"  # queued, running, completed, failed, cancelled
    progress: float = 0.0
    current_step: str = ""
    message: str = ""
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    output_dir: Optional[str] = None
    cancelled: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class StatusStore:
    """Thread-safe in-memory store for simulation status."""

    def __init__(self):
        self._store: Dict[str, SimulationStatus] = {}
        self._lock = threading.Lock()

    def create(self, simulation_id: str) -> SimulationStatus:
        """Create a new simulation status entry."""
        with self._lock:
            status = SimulationStatus(simulation_id=simulation_id)
            self._store[simulation_id] = status
            return status

    def get(self, simulation_id: str) -> Optional[SimulationStatus]:
        """Get simulation status by ID."""
        with self._lock:
            return self._store.get(simulation_id)

    def update(
        self,
        simulation_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        message: Optional[str] = None,
        log: Optional[str] = None,
        error: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Optional[SimulationStatus]:
        """Update simulation status."""
        with self._lock:
            sim_status = self._store.get(simulation_id)
            if not sim_status:
                return None

            if status is not None:
                sim_status.status = status
            if progress is not None:
                sim_status.progress = progress
            if current_step is not None:
                sim_status.current_step = current_step
            if message is not None:
                sim_status.message = message
            if log is not None:
                sim_status.logs.append(f"[{datetime.now().isoformat()}] {log}")
                # Keep only last 1000 log entries
                if len(sim_status.logs) > 1000:
                    sim_status.logs = sim_status.logs[-1000:]
            if error is not None:
                sim_status.error = error
                sim_status.status = "failed"
            if output_dir is not None:
                sim_status.output_dir = output_dir

            sim_status.updated_at = datetime.now()
            return sim_status

    def cancel(self, simulation_id: str) -> bool:
        """Mark a simulation as cancelled."""
        with self._lock:
            sim_status = self._store.get(simulation_id)
            if not sim_status:
                return False
            sim_status.cancelled = True
            sim_status.status = "cancelled"
            sim_status.message = "Cancellation requested"
            sim_status.updated_at = datetime.now()
            return True

    def is_cancelled(self, simulation_id: str) -> bool:
        """Check whether a simulation has been cancelled."""
        with self._lock:
            sim_status = self._store.get(simulation_id)
            return bool(sim_status.cancelled) if sim_status else False

    def delete(self, simulation_id: str) -> bool:
        """Delete simulation status."""
        with self._lock:
            return self._store.pop(simulation_id, None) is not None

    def list_all(self) -> List[SimulationStatus]:
        """List all simulation statuses."""
        with self._lock:
            return list(self._store.values())


# Global status store instance
status_store = StatusStore()


# ---------------------------------------------------------------------------
# Query store — holds results of background TAP query jobs
# ---------------------------------------------------------------------------


@dataclass
class QueryJobStatus:
    """Status and accumulated rows for a background TAP query job."""

    query_id: str
    status: str = "running"  # running | completed | failed | cancelled
    rows: List[dict] = field(default_factory=list)
    total: int = 0
    error: Optional[str] = None
    cancelled: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    _TTL_SECONDS: int = 3600  # drop jobs older than 1 hour

    def is_expired(self) -> bool:
        return (datetime.now() - self.created_at).total_seconds() > self._TTL_SECONDS


class QueryStore:
    """Thread-safe in-memory store for background TAP query jobs."""

    def __init__(self):
        self._store: Dict[str, QueryJobStatus] = {}
        self._lock = threading.Lock()

    def _evict_expired(self):
        """Remove jobs older than TTL (called under lock)."""
        expired = [qid for qid, job in self._store.items() if job.is_expired()]
        for qid in expired:
            del self._store[qid]

    def create(self, query_id: str) -> QueryJobStatus:
        with self._lock:
            self._evict_expired()
            job = QueryJobStatus(query_id=query_id)
            self._store[query_id] = job
            return job

    def get(self, query_id: str) -> Optional[QueryJobStatus]:
        with self._lock:
            return self._store.get(query_id)

    def append_rows(self, query_id: str, rows: List[dict]) -> None:
        with self._lock:
            job = self._store.get(query_id)
            if job:
                job.rows.extend(rows)
                job.updated_at = datetime.now()

    def complete(self, query_id: str) -> None:
        with self._lock:
            job = self._store.get(query_id)
            if job:
                job.status = "completed"
                job.total = len(job.rows)
                job.updated_at = datetime.now()

    def fail(self, query_id: str, error: str) -> None:
        with self._lock:
            job = self._store.get(query_id)
            if job:
                job.status = "failed"
                job.error = error
                job.updated_at = datetime.now()

    def cancel(self, query_id: str) -> bool:
        """Mark a job as cancelled. Returns True if the job existed."""
        with self._lock:
            job = self._store.get(query_id)
            if not job:
                return False
            job.cancelled = True
            job.status = "cancelled"
            job.updated_at = datetime.now()
            return True

    def is_cancelled(self, query_id: str) -> bool:
        with self._lock:
            job = self._store.get(query_id)
            return job.cancelled if job else False

    def get_page(self, query_id: str, page: int, page_size: int) -> dict:
        with self._lock:
            job = self._store.get(query_id)
            if not job:
                return {
                    "rows": [],
                    "done": True,
                    "total_fetched": 0,
                    "page": page,
                    "status": "failed",
                    "error": "Job not found",
                }
            start = page * page_size
            end = start + page_size
            rows = job.rows[start:end]
            done = job.status in ("completed", "failed", "cancelled") and end >= len(job.rows)
            return {
                "query_id": query_id,
                "page": page,
                "rows": rows,
                "page_size": page_size,
                "total_fetched": len(job.rows),
                "done": done,
                "status": job.status,
                "error": job.error,
            }

    def delete(self, query_id: str) -> bool:
        with self._lock:
            return self._store.pop(query_id, None) is not None


# Global query store instance
query_store = QueryStore()
