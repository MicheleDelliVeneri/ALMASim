"""Metadata business logic service."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from almasim.services.metadata.tap.queries import (
    load_metadata,
    query_metadata_by_science,
    query_science_types,
)
from almasim.services.metadata.tap.service import query_by_science_type as _tap_query

backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from database.models import Observation
from database.service import DatabaseService

logger = logging.getLogger(__name__)


class MetadataService:
    """Service for managing metadata queries with database caching."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize metadata service with optional database session."""
        self.db = db
        self.db_service = DatabaseService(db) if db else None

    def get_science_types(self) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Get science types and categories.

        First tries to fetch from database cache, falls back to TAP query.
        """
        if self.db_service:
            try:
                # Try to get from database
                keywords = self.db_service.get_all_science_keywords()
                categories = self.db_service.get_all_scientific_categories()

                if keywords or categories:
                    logger.info(
                        f"Retrieved science types from database: {len(keywords)} keywords, {len(categories)} categories"
                    )
                    return keywords, categories
            except Exception as e:
                logger.warning(f"Failed to retrieve science types from database: {e}")

        # Fall back to TAP query
        logger.info("Querying science types from ALMA TAP")
        keywords, categories = query_science_types()

        # Cache the keywords and categories in the database
        if self.db_service:
            try:
                from database.csv_importer import CSVImporter

                importer = CSVImporter(self.db)

                # Cache all keywords
                for keyword in keywords:
                    importer.get_or_create_keyword(keyword)

                # Cache all categories
                for category in categories:
                    importer.get_or_create_category(category)

                self.db.commit()
                logger.info(
                    f"Cached {len(keywords)} keywords and {len(categories)} categories in database"
                )
            except Exception as e:
                logger.error(f"Failed to cache science types in database: {e}")
                self.db.rollback()

        return keywords, categories

    def query_by_science(
        self,
        source_name: Optional[str] = None,
        science_keyword: Optional[Sequence[str]] = None,
        scientific_category: Optional[Sequence[str]] = None,
        bands: Optional[Sequence[int]] = None,
        antenna_arrays: Optional[str] = None,
        angular_resolution_range: Optional[Tuple[float, float]] = None,
        observation_date_range: Optional[Tuple[str, str]] = None,
        qa2_status: Optional[Sequence[str]] = None,
        obs_type: Optional[str] = None,
        fov_range: Optional[Tuple[float, float]] = None,
        time_resolution_range: Optional[Tuple[float, float]] = None,
        frequency_range: Optional[Tuple[float, float]] = None,
        save_to: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query metadata by science parameters.

        First checks database cache, then queries ALMA TAP if needed.
        Caches TAP results in database for future queries.
        """
        if self.db_service:
            try:
                # Only serve from cache when observations have science keywords —
                # rows cached before keyword support was added would show empty values.
                cached_with_keywords = self.db_service.count_observations_with_keywords()

                if cached_with_keywords > 0:
                    observations = self.db_service.query_observations(
                        target_name=source_name,
                        science_keywords=science_keyword,
                        scientific_categories=scientific_category,
                        bands=bands,
                        antenna_arrays=antenna_arrays,
                        angular_resolution_range=angular_resolution_range,
                        observation_date_range=observation_date_range,
                        qa2_status=qa2_status,
                        obs_type=obs_type,
                        fov_range=fov_range,
                        time_resolution_range=time_resolution_range,
                        frequency_range=frequency_range,
                        limit=1000,
                    )

                    logger.info(
                        f"Retrieved {len(observations)} observations from database cache"
                    )
                    return self._observations_to_dict(observations)
                else:
                    logger.info(
                        "No keyword-annotated observations in cache, querying ALMA TAP"
                    )
            except Exception as e:
                logger.warning(f"Database query failed: {e}, falling back to TAP")

        # Query from ALMA TAP
        logger.info("Querying ALMA TAP archive")
        result_df = query_metadata_by_science(
            science_keyword=science_keyword,
            scientific_category=scientific_category,
            bands=bands,
            fov_range=fov_range,
            time_resolution_range=time_resolution_range,
            frequency_range=frequency_range,
            source_name=source_name,
            antenna_arrays=antenna_arrays,
            angular_resolution_range=angular_resolution_range,
            observation_date_range=observation_date_range,
            qa2_status=qa2_status,
            obs_type=obs_type,
            save_to=save_to,
        )

        # Cache results in database if we have a db session
        if self.db_service and result_df is not None and not result_df.empty:
            try:
                self._cache_tap_results_in_db(result_df)
                logger.info(f"Cached {len(result_df)} observations in database")
            except Exception as e:
                logger.error(f"Failed to cache TAP results in database: {e}")

        # Convert to dict for API response
        return result_df.to_dict("records") if result_df is not None else []

    def run_background_query(
        self,
        query_id: str,
        source_name: Optional[str] = None,
        science_keyword: Optional[Sequence[str]] = None,
        scientific_category: Optional[Sequence[str]] = None,
        bands: Optional[Sequence[int]] = None,
        antenna_arrays: Optional[str] = None,
        angular_resolution_range: Optional[Tuple[float, float]] = None,
        observation_date_range: Optional[Tuple[str, str]] = None,
        qa2_status: Optional[Sequence[str]] = None,
        obs_type: Optional[str] = None,
        fov_range: Optional[Tuple[float, float]] = None,
        time_resolution_range: Optional[Tuple[float, float]] = None,
        frequency_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Run a full TAP query in the background and store results in query_store."""
        from app.services.status_store import query_store

        try:
            # Call TAP directly — bypasses DB cache so we always get fresh results
            result_df = _tap_query(
                science_keyword=science_keyword,
                scientific_category=scientific_category,
                band=bands,
                fov_range=fov_range,
                time_resolution_range=time_resolution_range,
                frequency_range=frequency_range,
                source_name=source_name,
                antenna_arrays=antenna_arrays,
                angular_resolution_range=angular_resolution_range,
                observation_date_range=observation_date_range,
                qa2_status=qa2_status,
                obs_type=obs_type,
            )
            rows = result_df.to_dict("records") if result_df is not None and not result_df.empty else []
            query_store.append_rows(query_id, rows)
            query_store.complete(query_id)
            logger.info(f"Background query {query_id} completed: {len(rows)} rows")

            if self.db_service and rows:
                try:
                    self._cache_tap_results_in_db(result_df)
                except Exception as e:
                    logger.error(f"Failed to cache background query results: {e}")
        except Exception as e:
            logger.error(f"Background query {query_id} failed: {e}", exc_info=True)
            query_store.fail(query_id, str(e))

    def load_metadata(self, metadata_path: Path) -> List[Dict[str, Any]]:
        """
        Load metadata from CSV file.

        Also caches in database if db session is available.
        """
        result_df = load_metadata(metadata_path)

        # Cache in database if available
        if self.db_service and result_df is not None and not result_df.empty:
            try:
                self._cache_tap_results_in_db(result_df)
                logger.info(
                    f"Cached {len(result_df)} observations from CSV in database"
                )
            except Exception as e:
                logger.error(f"Failed to cache CSV results in database: {e}")

        return result_df.to_dict("records") if result_df is not None else []

    def _observations_to_dict(
        self, observations: List[Observation]
    ) -> List[Dict[str, Any]]:
        """Convert Observation objects to dict format matching TAP output."""
        results = []
        for obs in observations:
            results.append(
                {
                    "ALMA_source_name": obs.target_name,
                    "Band": obs.band,
                    "antenna_arrays": obs.antenna_arrays,
                    "Ang.res.": obs.spatial_resolution,
                    "Obs.date": obs.obs_release_date.isoformat()
                    if obs.obs_release_date
                    else None,
                    "Project_abstract": obs.proposal_abstract,
                    "science_keyword": ", ".join(
                        [kw.keyword for kw in obs.science_keywords]
                    ),
                    "scientific_category": obs.scientific_category.category
                    if obs.scientific_category
                    else None,
                    "QA2_status": obs.qa2_passed,
                    "Type": obs.obs_type,
                    "PWV": obs.pwv,
                    "SB_name": obs.schedblock_name,
                    "Vel.res.": obs.velocity_resolution,
                    "RA": obs.ra,
                    "Dec": obs.dec,
                    "FOV": obs.s_fov,
                    "Int.Time": obs.t_max,
                    "Cont_sens_mJybeam": obs.cont_sensitivity_bandwidth,
                    "Line_sens_10kms_mJybeam": obs.sensitivity_10kms,
                    "Bandwidth": obs.bandwidth,
                    "Freq": obs.frequency,
                    "Freq.sup.": obs.frequency_support,
                    "proposal_id": obs.proposal_id,
                    "member_ous_uid": obs.member_ous_uid,
                    "group_ous_uid": obs.group_ous_uid,
                }
            )
        return results

    def _cache_tap_results_in_db(self, result_df):
        """Cache TAP query results in database with keywords and categories."""
        if not self.db_service or result_df.empty:
            return

        from database.csv_importer import CSVImporter

        try:
            importer = CSVImporter(self.db)
            cached_count = 0

            for _, row in result_df.iterrows():
                # Convert DataFrame row to dict
                row_dict = row.to_dict()

                # Check if already exists
                member_ous_uid = row_dict.get("member_ous_uid")
                if not member_ous_uid:
                    continue

                existing = self.db_service.get_observation_by_member_uid(member_ous_uid)

                # Extract science keywords and category from the TAP result
                science_keyword_str = row_dict.get("science_keyword", "")
                scientific_category_str = row_dict.get("scientific_category", "")

                if existing:
                    # Observation already in DB — upsert keywords/category if missing
                    if not existing.science_keywords and science_keyword_str:
                        keywords = [
                            kw.strip()
                            for kw in str(science_keyword_str).split(",")
                            if kw.strip()
                        ]
                        for keyword in keywords:
                            kw_obj = importer.get_or_create_keyword(keyword)
                            if kw_obj not in existing.science_keywords:
                                existing.science_keywords.append(kw_obj)
                        cached_count += 1

                    if existing.scientific_category is None and scientific_category_str:
                        category = str(scientific_category_str).strip()
                        if category:
                            cat_obj = importer.get_or_create_category(category)
                            existing.scientific_category = cat_obj
                    continue

                # New observation — parse and insert
                observation = importer.parse_csv_row(row_dict, "tap_query")
                if not observation:
                    continue

                # Add science keywords (comma-separated in TAP results)
                if science_keyword_str:
                    keywords = [
                        kw.strip()
                        for kw in str(science_keyword_str).split(",")
                        if kw.strip()
                    ]
                    for keyword in keywords:
                        kw_obj = importer.get_or_create_keyword(keyword)
                        if kw_obj not in observation.science_keywords:
                            observation.science_keywords.append(kw_obj)

                # Add scientific category
                if scientific_category_str:
                    category = str(scientific_category_str).strip()
                    if category:
                        cat_obj = importer.get_or_create_category(category)
                        observation.scientific_category = cat_obj

                self.db.add(observation)
                cached_count += 1

            self.db.commit()
            logger.info(
                f"Successfully cached {cached_count} new observations from TAP query"
            )

        except Exception as e:
            logger.error(f"Error caching TAP results: {e}", exc_info=True)
            self.db.rollback()
            raise
