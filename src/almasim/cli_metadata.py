"""Metadata query commands for the ALMASim CLI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

import typer

from .cli_shared import (
    default_output_path,
    split_csv_values,
    validate_date_range,
    validate_range,
)

metadata_app = typer.Typer(
    help="Metadata query commands.",
    no_args_is_help=True,
)


@lru_cache(maxsize=1)
def _tap_contract() -> dict[str, Any]:
    from .services.metadata.tap import (
        ALL_COLUMNS,
        ExclusionFilters,
        InclusionFilters,
        query_metadata_by_science,
        query_products,
    )

    return {
        "ALL_COLUMNS": ALL_COLUMNS,
        "ExclusionFilters": ExclusionFilters,
        "InclusionFilters": InclusionFilters,
        "query_metadata_by_science": query_metadata_by_science,
        "query_products": query_products,
    }


def save_dataframe_csv(*args, **kwargs):
    from .services.metadata.storage import save_dataframe_csv as _save_dataframe_csv

    return _save_dataframe_csv(*args, **kwargs)


def query_metadata_by_science(*args, **kwargs):
    return _tap_contract()["query_metadata_by_science"](*args, **kwargs)


def query_products(*args, **kwargs):
    return _tap_contract()["query_products"](*args, **kwargs)


def _build_query_summary(
    include: Any,
    exclude: Any,
    visible_columns: Optional[List[str]],
    limit: Optional[int],
) -> str:
    summary: dict[str, Any] = {
        "source_name": include.source_name,
        "science_keyword": include.science_keyword,
        "scientific_category": include.scientific_category,
        "band": include.band,
        "antenna_arrays": include.antenna_arrays,
        "array_type": include.array_type,
        "array_configuration": include.array_configuration,
        "angular_resolution_range": include.angular_resolution_range,
        "observation_date_range": include.observation_date_range,
        "qa2_status": include.qa2_status,
        "obs_type": include.obs_type,
        "fov_range": include.fov_range,
        "time_resolution_range": include.time_resolution_range,
        "frequency_range": include.frequency_range,
        "proposal_id_prefix": include.proposal_id_prefix,
        "proposal_id": include.proposal_id,
        "public_only": include.public_only,
        "science_only": include.science_only,
        "exclude_mosaic": include.exclude_mosaic,
        "exclude_science_keyword": exclude.science_keyword,
        "exclude_scientific_category": exclude.scientific_category,
        "exclude_source_name": exclude.source_name,
        "exclude_obs_type": exclude.obs_type,
        "exclude_solar": exclude.solar,
        "visible_columns": visible_columns,
        "limit": limit,
    }
    active = {key: value for key, value in summary.items() if value not in (None, [], False)}
    return ", ".join(f"{key}={value}" for key, value in active.items())


@metadata_app.command("query")
def metadata_query(
    source_name: Optional[str] = typer.Option(
        None,
        "--source-name",
        help="Source-name substring filter.",
    ),
    science_keyword: Optional[List[str]] = typer.Option(
        None,
        "--science-keyword",
        help="Science keyword(s) to include. Repeat or pass comma-separated values.",
    ),
    scientific_category: Optional[List[str]] = typer.Option(
        None,
        "--scientific-category",
        help="Scientific category(ies) to include. Repeat or pass comma-separated values.",
    ),
    band: Optional[List[int]] = typer.Option(
        None,
        "--band",
        help="ALMA band filter. Repeat for multiple values.",
    ),
    antenna_arrays: Optional[str] = typer.Option(
        None,
        "--antenna-arrays",
        help="Raw antenna_arrays substring filter.",
    ),
    array_type: Optional[List[str]] = typer.Option(
        None,
        "--array-type",
        help="Array type(s): 12m, 7m, TP. Repeat or pass comma-separated values.",
    ),
    array_configuration: Optional[List[str]] = typer.Option(
        None,
        "--array-configuration",
        help="Array configuration(s) like C-1, C-2. Repeat or pass comma-separated values.",
    ),
    angular_resolution_min: Optional[float] = typer.Option(
        None,
        "--angular-resolution-min",
        help="Minimum angular resolution (arcsec).",
    ),
    angular_resolution_max: Optional[float] = typer.Option(
        None,
        "--angular-resolution-max",
        help="Maximum angular resolution (arcsec).",
    ),
    observation_date_start: Optional[str] = typer.Option(
        None,
        "--observation-date-start",
        help="Observation date start (ISO date).",
    ),
    observation_date_end: Optional[str] = typer.Option(
        None,
        "--observation-date-end",
        help="Observation date end (ISO date).",
    ),
    qa2_status: Optional[List[str]] = typer.Option(
        None,
        "--qa2-status",
        help="QA2 status values: Pass, Fail, SemiPass, T, F, X.",
    ),
    obs_type: Optional[List[str]] = typer.Option(
        None,
        "--obs-type",
        help="Observation type(s) to include. Repeat or pass comma-separated values.",
    ),
    fov_min: Optional[float] = typer.Option(
        None,
        "--fov-min",
        help="Minimum FOV.",
    ),
    fov_max: Optional[float] = typer.Option(
        None,
        "--fov-max",
        help="Maximum FOV.",
    ),
    time_resolution_min: Optional[float] = typer.Option(
        None,
        "--time-resolution-min",
        help="Minimum time resolution.",
    ),
    time_resolution_max: Optional[float] = typer.Option(
        None,
        "--time-resolution-max",
        help="Maximum time resolution.",
    ),
    frequency_min: Optional[float] = typer.Option(
        None,
        "--frequency-min",
        help="Minimum frequency.",
    ),
    frequency_max: Optional[float] = typer.Option(
        None,
        "--frequency-max",
        help="Maximum frequency.",
    ),
    exclude_science_keyword: Optional[List[str]] = typer.Option(
        None,
        "--exclude-science-keyword",
        help="Science keyword(s) to exclude. Repeat or pass comma-separated values.",
    ),
    exclude_scientific_category: Optional[List[str]] = typer.Option(
        None,
        "--exclude-scientific-category",
        help="Scientific category(ies) to exclude. Repeat or pass comma-separated values.",
    ),
    exclude_source_name: Optional[List[str]] = typer.Option(
        None,
        "--exclude-source-name",
        help="Source-name substring(s) to exclude. Repeat or pass comma-separated values.",
    ),
    exclude_obs_type: Optional[List[str]] = typer.Option(
        None,
        "--exclude-obs-type",
        help="Observation type substring(s) to exclude. Repeat or pass comma-separated values.",
    ),
    exclude_solar: bool = typer.Option(
        False,
        "--exclude-solar",
        help="Exclude solar observations.",
    ),
    cycles: Optional[List[int]] = typer.Option(
        None,
        "--cycle",
        help="ALMA cycle number(s). Cycle N maps to proposal_id prefix {2012+N}.",
    ),
    project_codes: Optional[List[str]] = typer.Option(
        None,
        "--project-code",
        help=(
            "Exact ALMA project code(s). Repeat for multiple or pass comma-separated values. "
            "Example: --project-code 2019.1.00001.S --project-code 2022.1.00333.S"
        ),
    ),
    public_only: bool = typer.Option(
        True,
        "--public-only/--include-proprietary",
        help="Restrict to public data (default: public only).",
    ),
    science_only: bool = typer.Option(
        True,
        "--science-only/--include-non-science",
        help="Restrict to science observations (default: science only).",
    ),
    exclude_mosaic: bool = typer.Option(
        True,
        "--exclude-mosaic/--include-mosaic",
        help="Exclude mosaic observations (default: excluded).",
    ),
    visible_columns: Optional[List[str]] = typer.Option(
        None,
        "--visible-column",
        help="Column(s) to include in output (ordered). Repeat this option per column.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        min=1,
        help="Maximum number of metadata rows to keep. Defaults to unlimited.",
    ),
    save_csv: Optional[Path] = typer.Option(
        None,
        "--save-csv",
        help="Destination CSV for normalized metadata rows.",
    ),
    save_products_csv: Optional[Path] = typer.Option(
        None,
        "--save-products-csv",
        help="Optional destination CSV for resolved product rows.",
    ),
    product_member_limit: Optional[int] = typer.Option(
        None,
        "--product-member-limit",
        min=1,
        help=(
            "How many queried member OUS UIDs to resolve into product rows. Defaults to unlimited."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt and start the query immediately.",
    ),
) -> None:
    """Query ALMA metadata from TAP and save normalized CSV results."""
    tap = _tap_contract()
    all_columns = tap["ALL_COLUMNS"]
    inclusion_filters = tap["InclusionFilters"]
    exclusion_filters = tap["ExclusionFilters"]

    science_keywords = split_csv_values(science_keyword)
    scientific_categories = split_csv_values(scientific_category)
    array_types = split_csv_values(array_type)
    array_configurations = split_csv_values(array_configuration)
    qa2_status_values = split_csv_values(qa2_status)
    obs_types = split_csv_values(obs_type)
    exclude_science_keywords = split_csv_values(exclude_science_keyword)
    exclude_scientific_categories = split_csv_values(exclude_scientific_category)
    exclude_source_names = split_csv_values(exclude_source_name)
    exclude_obs_types = split_csv_values(exclude_obs_type)
    output_columns = split_csv_values(visible_columns)

    invalid_array_types = [
        value for value in (array_types or []) if value not in {"12m", "7m", "TP"}
    ]
    if invalid_array_types:
        typer.echo(
            "Invalid --array-type values: "
            + ", ".join(invalid_array_types)
            + ". Allowed values: 12m, 7m, TP.",
            err=True,
        )
        raise typer.Exit(code=2)

    invalid_columns = [column for column in (output_columns or []) if column not in all_columns]
    if invalid_columns:
        typer.echo(
            "Invalid --visible-column values: " + ", ".join(invalid_columns) + ".",
            err=True,
        )
        typer.echo("Allowed columns: " + ", ".join(all_columns), err=True)
        raise typer.Exit(code=2)

    angular_resolution_range = validate_range(
        min_value=angular_resolution_min,
        max_value=angular_resolution_max,
        label="angular resolution range",
    )
    observation_date_range = validate_date_range(
        start_date=observation_date_start,
        end_date=observation_date_end,
    )
    fov_range = validate_range(min_value=fov_min, max_value=fov_max, label="FOV range")
    time_resolution_range = validate_range(
        min_value=time_resolution_min,
        max_value=time_resolution_max,
        label="time resolution range",
    )
    frequency_range = validate_range(
        min_value=frequency_min,
        max_value=frequency_max,
        label="frequency range",
    )

    proposal_id_prefix = [f"{2012 + cycle}." for cycle in cycles] if cycles else None
    project_code_list = split_csv_values(project_codes)

    include = inclusion_filters(
        science_keyword=science_keywords,
        scientific_category=scientific_categories,
        band=band,
        source_name=source_name,
        antenna_arrays=antenna_arrays,
        array_type=array_types,
        array_configuration=array_configurations,
        angular_resolution_range=angular_resolution_range,
        observation_date_range=observation_date_range,
        qa2_status=qa2_status_values,
        obs_type=obs_types,
        fov_range=fov_range,
        time_resolution_range=time_resolution_range,
        frequency_range=frequency_range,
        proposal_id_prefix=proposal_id_prefix,
        proposal_id=project_code_list,
        public_only=public_only,
        science_only=science_only,
        exclude_mosaic=exclude_mosaic,
    )
    exclude = exclusion_filters(
        science_keyword=exclude_science_keywords,
        scientific_category=exclude_scientific_categories,
        source_name=exclude_source_names,
        obs_type=exclude_obs_types,
        solar=exclude_solar,
    )

    if not yes:
        message = (
            "About to start ALMA TAP metadata query with "
            + _build_query_summary(include, exclude, output_columns, limit)
            + ". Continue?"
        )
        if not typer.confirm(message, default=True):
            typer.echo("Query cancelled.")
            raise typer.Exit(code=0)

    typer.echo("Starting ALMA TAP metadata query...")

    metadata = query_metadata_by_science(
        include=include,
        exclude=exclude,
        visible_columns=output_columns,
    )
    if metadata.empty:
        typer.echo("No metadata rows matched the requested filters", err=True)
        raise typer.Exit(code=1)

    if limit is not None:
        metadata = metadata.head(limit).reset_index(drop=True)

    output_csv = (
        (save_csv or default_output_path("metadata_query_results.csv")).expanduser().resolve()
    )
    save_dataframe_csv(metadata, output_csv)

    typer.echo(f"Saved metadata CSV: {output_csv}")
    typer.echo(f"Metadata rows: {len(metadata)}")

    display_columns = [
        col
        for col in ("ALMA_source_name", "Band", "Freq", "member_ous_uid")
        if col in metadata.columns
    ]
    if display_columns:
        typer.echo(metadata[display_columns].head(min(5, len(metadata))).to_string(index=False))

    if save_products_csv is not None:
        if "member_ous_uid" not in metadata.columns:
            typer.echo("Queried metadata does not include member_ous_uid values", err=True)
            raise typer.Exit(code=1)

        member_series = metadata["member_ous_uid"].dropna().astype(str)
        if product_member_limit is not None:
            member_series = member_series.head(product_member_limit)
        member_uids = member_series.tolist()
        if not member_uids:
            typer.echo("Queried metadata does not include member_ous_uid values", err=True)
            raise typer.Exit(code=1)

        products = query_products(member_uids)
        products_csv = save_products_csv.expanduser().resolve()
        products_csv.parent.mkdir(parents=True, exist_ok=True)
        products.to_csv(products_csv, index=False)
        typer.echo(f"Saved products CSV: {products_csv}")
        typer.echo(f"Product rows: {len(products)}")
