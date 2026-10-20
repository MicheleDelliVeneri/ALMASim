"""Database service layer: the three repositories on one session (MADIV-28 T6.10).

New code can use the repositories in ``app.db.repositories`` directly.
"""

from sqlalchemy.orm import Session

from .repositories import ObservationRepository, PresetRepository, SimulationRepository


class DatabaseService(ObservationRepository, PresetRepository, SimulationRepository):
    """Observations, query presets and simulation jobs through one session."""

    def __init__(self, db: Session):
        self.db = db
