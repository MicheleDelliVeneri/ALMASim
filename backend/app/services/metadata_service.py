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
from almasim.services.metadata.tap.service import (
    ExclusionFilters,
    InclusionFilters,
    _QA2_STATUS_MAP,
    query_by_science_type as _tap_query,
)
from app.schemas.metadata import ScienceQueryParams


def _derive_array_type_local(antenna_arrays_str: object) -> str:
    """Derive human-readable array type from antenna_arrays string (DA/DV→12m, CM→7m, PM→TP)."""
    if not isinstance(antenna_arrays_str, str) or not antenna_arrays_str:
        return ""
    upper = antenna_arrays_str.upper()
    types = []
    if "DA" in upper or "DV" in upper:
        types.append("12m")
    if "CM" in upper:
        types.append("7m")
    if "PM" in upper:
        types.append("TP")
    return "+".join(types) if types else ""

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
        params: ScienceQueryParams,
        save_to: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Query metadata by science parameters.

        First checks database cache, then queries ALMA TAP if needed.
        Caches TAP results in database for future queries.
        """
        # Map QA2 human-friendly labels to TAP stored values for the DB cache path
        mapped_qa2 = (
            [_QA2_STATUS_MAP.get(s, s) for s in params.qa2_status]
            if params.qa2_status
            else None
        )

        db_result = self._try_cache_query(params, mapped_qa2)
        if db_result is not None:
            return db_result

        return self._run_fresh_tap_query(params, save_to)

    def _try_cache_query(
        self,
        params: ScienceQueryParams,
        mapped_qa2: Optional[List[str]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Attempt to serve results from the DB cache. Returns None on miss or error."""
        if not self.db_service:
            return None
        try:
            if self.db_service.count_observations_with_keywords() == 0:
                logger.info("No keyword-annotated observations in cache, querying ALMA TAP")
                return None
            observations = self.db_service.query_observations(
                target_name=params.source_name,
                science_keywords=params.science_keyword,
                scientific_categories=params.scientific_category,
                bands=params.bands,
                antenna_arrays=params.antenna_arrays,
                array_type=params.array_type,
                array_configuration=params.array_configuration,
                angular_resolution_range=params.angular_resolution_range,
                observation_date_range=params.observation_date_range,
                qa2_status=mapped_qa2,
                obs_type=params.obs_type,
                fov_range=params.fov_range,
                time_resolution_range=params.time_resolution_range,
                frequency_range=params.frequency_range,
                proposal_id_prefix=params.proposal_id_prefix,
                exclude_science_keywords=params.exclude_science_keyword,
                exclude_scientific_categories=params.exclude_scientific_category,
                exclude_source_names=params.exclude_source_name,
                exclude_obs_types=params.exclude_obs_type,
                exclude_solar=params.exclude_solar,
                limit=1000,
            )
            logger.info(f"Retrieved {len(observations)} observations from database cache")
            records = self._observations_to_dict(observations)
            if params.visible_columns:
                vc = set(params.visible_columns)
                records = [{k: v for k, v in row.items() if k in vc} for row in records]
            return records
        except Exception as e:
            logger.warning(f"Database query failed: {e}, falling back to TAP")
            return None

    def _run_fresh_tap_query(
        self,
        params: ScienceQueryParams,
        save_to: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Run a fresh TAP query and optionally cache the results in the DB."""
        logger.info("Querying ALMA TAP archive")
        include = InclusionFilters(
            science_keyword=params.science_keyword,
            scientific_category=params.scientific_category,
            band=params.bands,
            fov_range=params.fov_range,
            time_resolution_range=params.time_resolution_range,
            frequency_range=params.frequency_range,
            source_name=params.source_name,
            antenna_arrays=params.antenna_arrays,
            array_type=params.array_type,
            array_configuration=params.array_configuration,
            angular_resolution_range=params.angular_resolution_range,
            observation_date_range=params.observation_date_range,
            qa2_status=params.qa2_status,  # QA2 mapping is applied inside the TAP layer
            obs_type=params.obs_type,
            proposal_id_prefix=params.proposal_id_prefix,
        )
        exclude = ExclusionFilters(
            science_keyword=params.exclude_science_keyword,
            scientific_category=params.exclude_scientific_category,
            source_name=params.exclude_source_name,
            obs_type=params.exclude_obs_type,
            solar=params.exclude_solar,
        )
        result_df = query_metadata_by_science(
            include=include,
            exclude=exclude,
            visible_columns=params.visible_columns,
            save_to=save_to,
        )
        if self.db_service and result_df is not None and not result_df.empty:
            try:
                self._cache_tap_results_in_db(result_df)
                logger.info(f"Cached {len(result_df)} observations in database")
            except Exception as e:
                logger.error(f"Failed to cache TAP results in database: {e}")
        return result_df.to_dict("records") if result_df is not None else []

    def run_background_query(self, query_id: str, params: ScienceQueryParams) -> None:
        """Run a full TAP query in the background and store results in query_store.

        Bypasses the DB cache so the caller always gets fresh TAP results.
        """
        from app.services.status_store import query_store

        try:
            include = InclusionFilters(
                science_keyword=params.science_keyword,
                scientific_category=params.scientific_category,
                band=params.bands,
                fov_range=params.fov_range,
                time_resolution_range=params.time_resolution_range,
                frequency_range=params.frequency_range,
                source_name=params.source_name,
                antenna_arrays=params.antenna_arrays,
                array_type=params.array_type,
                array_configuration=params.array_configuration,
                angular_resolution_range=params.angular_resolution_range,
                observation_date_range=params.observation_date_range,
                qa2_status=params.qa2_status,
                obs_type=params.obs_type,
                proposal_id_prefix=params.proposal_id_prefix,
            )
            exclude = ExclusionFilters(
                science_keyword=params.exclude_science_keyword,
                scientific_category=params.exclude_scientific_category,
                source_name=params.exclude_source_name,
                obs_type=params.exclude_obs_type,
                solar=params.exclude_solar,
            )
            # Use query_metadata_by_science so columns are normalized/renamed
            # to display names (same as the sync path).
            result_df = query_metadata_by_science(
                include=include,
                exclude=exclude,
                visible_columns=None,
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
        return [self._obs_to_dict(obs) for obs in observations]

    @staticmethod
    def _obs_to_dict(obs: Observation) -> Dict[str, Any]:
        """Serialize a single Observation ORM object to the canonical column dict."""
        return {
            "ALMA_source_name": obs.target_name,
            "Band": obs.band,
            "Array_type": _derive_array_type_local(obs.antenna_arrays),
            "antenna_arrays": obs.antenna_arrays,
            "Ang.res.": obs.spatial_resolution,
            "Obs.date": obs.obs_release_date.isoformat() if obs.obs_release_date else None,
            "Project_abstract": obs.proposal_abstract,
            "science_keyword": ", ".join(kw.keyword for kw in obs.science_keywords),
            "scientific_category": obs.scientific_category.category if obs.scientific_category else None,
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

    def _cache_tap_results_in_db(self, result_df) -> None:
        """Cache TAP query results in database with keywords and categories."""
        if not self.db_service or result_df.empty:
            return
        from database.csv_importer import CSVImporter

        try:
            importer = CSVImporter(self.db)
            seen: set = set()  # deduplicate within the batch (TAP can return duplicate UIDs)
            cached_count = sum(
                self._upsert_row(importer, row.to_dict(), seen)
                for _, row in result_df.iterrows()
            )
            self.db.commit()
            logger.info(f"Successfully cached {cached_count} new observations from TAP query")
        except Exception as e:
            logger.error(f"Error caching TAP results: {e}", exc_info=True)
            self.db.rollback()
            raise

    def _upsert_row(self, importer, row_dict: Dict[str, Any], seen: set) -> int:
        """Insert or update one TAP row in the database. Returns 1 if written, 0 otherwise."""
        member_ous_uid = row_dict.get("member_ous_uid")
        if not member_ous_uid or member_ous_uid in seen:
            return 0
        seen.add(member_ous_uid)
        kw_str = row_dict.get("science_keyword", "")
        cat_str = row_dict.get("scientific_category", "")
        existing = self.db_service.get_observation_by_member_uid(member_ous_uid)
        if existing:
            self._patch_existing(importer, existing, kw_str, cat_str)
            return 0
        return self._insert_new(importer, row_dict, kw_str, cat_str)

    def _patch_existing(self, importer, existing, kw_str: str, cat_str: str) -> None:
        """Add missing keywords/category to an already-cached observation."""
        if not existing.science_keywords and kw_str:
            for kw in (k.strip() for k in str(kw_str).split(",") if k.strip()):
                kw_obj = importer.get_or_create_keyword(kw)
                if kw_obj not in existing.science_keywords:
                    existing.science_keywords.append(kw_obj)
        if existing.scientific_category is None and cat_str:
            cat = str(cat_str).strip()
            if cat:
                existing.scientific_category = importer.get_or_create_category(cat)

    def _insert_new(self, importer, row_dict: Dict[str, Any], kw_str: str, cat_str: str) -> int:
        """Parse and insert a brand-new observation. Returns 1 on success, 0 on skip."""
        observation = importer.parse_csv_row(row_dict, "tap_query")
        if not observation:
            return 0
        if kw_str:
            for kw in (k.strip() for k in str(kw_str).split(",") if k.strip()):
                kw_obj = importer.get_or_create_keyword(kw)
                if kw_obj not in observation.science_keywords:
                    observation.science_keywords.append(kw_obj)
        if cat_str:
            cat = str(cat_str).strip()
            if cat:
                observation.scientific_category = importer.get_or_create_category(cat)
        self.db.add(observation)
        return 1
