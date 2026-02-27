"""CSV data importer for initializing the database."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Observation, QueryResult, ScienceKeyword, ScientificCategory

logger = logging.getLogger(__name__)


class CSVImporter:
    """Import CSV metadata files into the database."""

    def __init__(self, db: Session):
        self.db = db
        self.keyword_cache: Dict[str, ScienceKeyword] = {}
        self.category_cache: Dict[str, ScientificCategory] = {}

    def get_or_create_keyword(self, keyword: str) -> ScienceKeyword:
        """Get or create a science keyword."""
        if keyword in self.keyword_cache:
            return self.keyword_cache[keyword]

        stmt = select(ScienceKeyword).where(ScienceKeyword.keyword == keyword)
        kw_obj = self.db.execute(stmt).scalar_one_or_none()

        if not kw_obj:
            kw_obj = ScienceKeyword(keyword=keyword)
            self.db.add(kw_obj)
            self.db.flush()

        self.keyword_cache[keyword] = kw_obj
        return kw_obj

    def get_or_create_category(self, category: str) -> ScientificCategory:
        """Get or create a scientific category."""
        if category in self.category_cache:
            return self.category_cache[category]

        stmt = select(ScientificCategory).where(ScientificCategory.category == category)
        cat_obj = self.db.execute(stmt).scalar_one_or_none()

        if not cat_obj:
            cat_obj = ScientificCategory(category=category)
            self.db.add(cat_obj)
            self.db.flush()

        self.category_cache[category] = cat_obj
        return cat_obj

    def parse_csv_row(
        self, row: Dict[str, str], source_file: str
    ) -> Optional[Observation]:
        """Parse a CSV row into an Observation object."""
        try:
            # Check if observation already exists
            member_ous_uid = row.get("member_ous_uid")
            if not member_ous_uid:
                logger.warning(f"Skipping row without member_ous_uid: {row}")
                return None

            stmt = select(Observation).where(
                Observation.member_ous_uid == member_ous_uid
            )
            existing = self.db.execute(stmt).scalar_one_or_none()

            if existing:
                logger.debug(f"Observation {member_ous_uid} already exists, skipping")
                return None

            # Parse band - handle both "Band" and integer values
            band_str = row.get("Band", "")
            band = None
            if band_str:
                try:
                    # Remove 'Band' prefix if present and convert to int
                    band = int(band_str.replace("Band", "").strip())
                except (ValueError, AttributeError):
                    logger.warning(f"Could not parse band: {band_str}")

            # Parse dates
            obs_date = None
            obs_date_str = row.get("Obs.date", "")
            if obs_date_str:
                try:
                    obs_date = datetime.fromisoformat(
                        obs_date_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    logger.warning(f"Could not parse date: {obs_date_str}")

            # Create observation
            observation = Observation(
                member_ous_uid=member_ous_uid,
                group_ous_uid=row.get("group_ous_uid"),
                proposal_id=row.get("proposal_id"),
                target_name=row.get("ALMA_source_name", ""),
                ra=float(row["RA"]) if row.get("RA") else None,
                dec=float(row["Dec"]) if row.get("Dec") else None,
                band=band,
                pwv=float(row["PWV"]) if row.get("PWV") else None,
                schedblock_name=row.get("SB_name"),
                velocity_resolution=float(row["Vel.res."])
                if row.get("Vel.res.")
                else None,
                spatial_resolution=float(row["Ang.res."])
                if row.get("Ang.res.")
                else None,
                s_fov=float(row["FOV"]) if row.get("FOV") else None,
                t_resolution=None,  # Not in CSV
                cont_sensitivity_bandwidth=float(row["Cont_sens_mJybeam"])
                if row.get("Cont_sens_mJybeam")
                else None,
                sensitivity_10kms=float(row["Line_sens_10kms_mJybeam"])
                if row.get("Line_sens_10kms_mJybeam")
                else None,
                frequency=float(row["Freq"]) if row.get("Freq") else None,
                bandwidth=float(row["Bandwidth"]) if row.get("Bandwidth") else None,
                frequency_support=row.get("Freq.sup."),
                obs_release_date=obs_date,
                t_max=float(row["Int.Time"]) if row.get("Int.Time") else None,
                antenna_arrays=row.get("antenna_arrays"),
                band_list=str(band) if band else None,
                proposal_abstract=row.get("Project_abstract"),
                qa2_passed=row.get("QA2_status"),
                obs_type=row.get("Type"),
                source_file=source_file,
            )

            return observation

        except Exception as e:
            logger.error(f"Error parsing row: {e}, row: {row}")
            return None

    def import_csv_file(self, csv_path: Path) -> int:
        """
        Import a single CSV file into the database.

        Returns:
            Number of observations imported
        """
        logger.info(f"Importing CSV file: {csv_path}")

        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return 0

        imported_count = 0

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    observation = self.parse_csv_row(row, csv_path.name)
                    if observation:
                        self.db.add(observation)
                        imported_count += 1

                self.db.commit()
                logger.info(
                    f"Successfully imported {imported_count} observations from {csv_path.name}"
                )

        except Exception as e:
            logger.error(f"Error importing CSV file {csv_path}: {e}")
            self.db.rollback()
            raise

        return imported_count

    def import_json_file(self, json_path: Path) -> int:
        """
        Import a JSON file into the database.

        Expected format: {"count": N, "data": [{...}, {...}]}

        Returns:
            Number of observations imported
        """
        logger.info(f"Importing JSON file: {json_path}")

        if not json_path.exists():
            logger.warning(f"JSON file not found: {json_path}")
            return 0

        imported_count = 0

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Handle both wrapped format {"count": N, "data": [...]} and plain array
                records = data.get("data", data) if isinstance(data, dict) else data

                for row in records:
                    observation = self.parse_csv_row(row, json_path.name)
                    if observation:
                        self.db.add(observation)
                        imported_count += 1

                self.db.commit()
                logger.info(
                    f"Successfully imported {imported_count} observations from {json_path.name}"
                )

        except Exception as e:
            logger.error(f"Error importing JSON file {json_path}: {e}")
            self.db.rollback()
            raise

        return imported_count

    def import_data_directory(self, data_dir: Path) -> Dict[str, int]:
        """
        Import all CSV and JSON files from the data directory.

        Returns:
            Dictionary mapping filename to number of imported records
        """
        results = {}

        # Import CSV files
        for csv_file in data_dir.glob("*.csv"):
            try:
                count = self.import_csv_file(csv_file)
                results[csv_file.name] = count
            except Exception as e:
                logger.error(f"Failed to import {csv_file}: {e}")
                results[csv_file.name] = 0

        # Import JSON files
        for json_file in data_dir.glob("*.json"):
            try:
                count = self.import_json_file(json_file)
                results[json_file.name] = count
            except Exception as e:
                logger.error(f"Failed to import {json_file}: {e}")
                results[json_file.name] = 0

        return results


def initialize_database_from_csv(db: Session, data_dir: Path) -> Dict[str, int]:
    """
    Initialize database with data from CSV/JSON files.

    Args:
        db: Database session
        data_dir: Directory containing CSV/JSON files

    Returns:
        Dictionary mapping filename to number of imported records
    """
    importer = CSVImporter(db)
    return importer.import_data_directory(data_dir)
