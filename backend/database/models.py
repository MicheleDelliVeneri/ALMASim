"""SQLAlchemy database models for ALMASim."""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# Association table for many-to-many between observations and science keywords
observation_keywords = Table(
    "observation_keywords",
    Base.metadata,
    Column(
        "observation_id",
        Integer,
        ForeignKey("observations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "keyword_id",
        Integer,
        ForeignKey("science_keywords.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class ScienceKeyword(Base):
    """Science keywords from ALMA TAP."""

    __tablename__ = "science_keywords"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    observations = relationship(
        "Observation", secondary=observation_keywords, back_populates="science_keywords"
    )


class ScientificCategory(Base):
    """Scientific categories from ALMA TAP."""

    __tablename__ = "scientific_categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    observations = relationship("Observation", back_populates="scientific_category")


class Observation(Base):
    """ALMA observation metadata from TAP queries."""

    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Unique identifiers
    member_ous_uid = Column(String(255), unique=True, nullable=False, index=True)
    group_ous_uid = Column(String(255), index=True)
    proposal_id = Column(String(100), index=True)

    # Source information
    target_name = Column(String(255), index=True)
    ra = Column(Float)
    dec = Column(Float)

    # Observation parameters
    band = Column(Integer, index=True)
    pwv = Column(Float)
    schedblock_name = Column(Text)  # Can be very long with multiple schedblocks

    # Resolution and sensitivity
    velocity_resolution = Column(Float)  # km/s
    spatial_resolution = Column(Float)  # arcsec
    s_fov = Column(Float)  # arcsec
    t_resolution = Column(Float)
    cont_sensitivity_bandwidth = Column(Float)  # mJy/beam
    sensitivity_10kms = Column(Float)  # mJy/beam

    # Spectral information
    frequency = Column(Float)  # GHz
    bandwidth = Column(Float)  # GHz
    frequency_support = Column(
        Text
    )  # Complex frequency range string (can be very long)

    # Timing
    obs_release_date = Column(DateTime)
    t_max = Column(Float)  # Integration time in seconds

    # Arrays and configuration
    antenna_arrays = Column(Text)  # Semicolon-separated (can be long)
    band_list = Column(String(100))

    # Proposal and quality
    proposal_abstract = Column(Text)
    qa2_passed = Column(String(10))  # 'T', 'F', or other values from TAP
    obs_type = Column(
        String(100)
    )  # TAP 'type' field (renamed to avoid Python keyword conflict)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    source_file = Column(String(255))  # CSV file that imported this record

    # Foreign keys
    scientific_category_id = Column(Integer, ForeignKey("scientific_categories.id"))

    # Relationships
    scientific_category = relationship(
        "ScientificCategory", back_populates="observations"
    )
    science_keywords = relationship(
        "ScienceKeyword", secondary=observation_keywords, back_populates="observations"
    )
    simulations = relationship("SimulationJob", back_populates="observation")


class QueryResult(Base):
    """Cached TAP query results."""

    __tablename__ = "query_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_name = Column(String(255), nullable=False, index=True)
    query_params = Column(JSON)  # Store the query parameters used
    result_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Store observation IDs that are part of this query result
    observation_ids = Column(ARRAY(Integer))

    # Metadata
    description = Column(Text)
    created_by = Column(String(100))


class SimulationJob(Base):
    """Simulation job tracking."""

    __tablename__ = "simulation_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(
        UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4, index=True
    )

    # Source information
    observation_id = Column(Integer, ForeignKey("observations.id"), nullable=True)
    source_name = Column(String(255), nullable=False)

    # Simulation parameters (from SimulationParams)
    idx = Column(Integer)
    project_name = Column(String(255))

    # Sky model parameters
    source_type = Column(String(50))  # point, gaussian, extended, diffuse, etc.
    n_pix = Column(Float)
    n_channels = Column(Integer)
    rest_frequency = Column(Float)
    redshift = Column(Float)
    lum_infrared = Column(Float)
    snr = Column(Float)
    n_lines = Column(Integer)
    line_names = Column(ARRAY(String))

    # Configuration
    save_mode = Column(String(20))  # npz, fits
    inject_serendipitous = Column(Boolean, default=False)
    ncpu = Column(Integer, default=1)
    remote = Column(Boolean, default=False)

    # Status tracking
    status = Column(
        String(50), nullable=False, default="queued", index=True
    )  # queued, running, completed, failed
    progress = Column(Float, default=0.0)
    current_step = Column(String(255))
    message = Column(Text)
    error = Column(Text)

    # Paths
    output_path = Column(String(512))

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    observation = relationship("Observation", back_populates="simulations")
    logs = relationship(
        "SimulationLog", back_populates="simulation", cascade="all, delete-orphan"
    )


class SimulationLog(Base):
    """Simulation execution logs."""

    __tablename__ = "simulation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("simulation_jobs.simulation_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    level = Column(String(20), default="INFO")  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text, nullable=False)

    # Relationships
    simulation = relationship("SimulationJob", back_populates="logs")


class DownloadJobRecord(Base):
    """Persisted record of a download job."""

    __tablename__ = "download_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4, index=True
    )
    destination = Column(Text, nullable=False)
    member_ous_uids = Column(Text, nullable=False, default="")  # JSON array string
    metadata_json = Column(Text, nullable=False, default="[]")  # JSON array string
    product_filter = Column(String(20), nullable=False, default="all")
    unpack_ms = Column(Boolean, nullable=False, default=False)
    generate_calibrated_visibilities = Column(Boolean, nullable=False, default=False)
    clean_intermediate_files = Column(Boolean, nullable=False, default=False)
    archive_output_root = Column(Text, nullable=True)
    casa_data_root = Column(Text, nullable=True)
    skip_casa_data_update = Column(Boolean, nullable=False, default=False)
    raw_measurement_sets = Column(
        Text, nullable=False, default="[]"
    )  # JSON array string
    calibrated_measurement_sets = Column(
        Text, nullable=False, default="[]"
    )  # JSON array string
    manifest_path = Column(Text, nullable=True)
    total_files = Column(Integer, nullable=False, default=0)
    total_bytes = Column(Float, nullable=False, default=0)
    bytes_downloaded = Column(Float, nullable=False, default=0)
    files_completed = Column(Integer, nullable=False, default=0)
    files_failed = Column(Integer, nullable=False, default=0)
    status = Column(
        String(20), nullable=False, default="pending"
    )  # pending, running, completed, failed, cancelled
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    files = relationship(
        "DownloadFileRecord", back_populates="job", cascade="all, delete-orphan"
    )


class DownloadFileRecord(Base):
    """Persisted record of a single file within a download job."""

    __tablename__ = "download_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("download_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = Column(Text, nullable=False)
    access_url = Column(Text, nullable=False)
    content_length = Column(Float, nullable=False, default=0)
    bytes_downloaded = Column(Float, nullable=False, default=0)
    status = Column(
        String(20), nullable=False, default="pending"
    )  # pending, downloading, completed, failed
    error = Column(Text, nullable=True)
    sha256 = Column(String(64), nullable=True)

    # Relationships
    job = relationship("DownloadJobRecord", back_populates="files")
