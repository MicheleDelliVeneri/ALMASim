"""In-memory store for simulation status and logs."""
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import threading


@dataclass
class SimulationStatus:
    """Simulation status information."""
    simulation_id: str
    status: str = "queued"  # queued, running, completed, failed
    progress: float = 0.0
    current_step: str = ""
    message: str = ""
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
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
            
            sim_status.updated_at = datetime.now()
            return sim_status
    
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

