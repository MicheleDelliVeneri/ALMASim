"""CLI example for querying ALMA metadata and saving query results.

This script demonstrates:
1. querying metadata directly from the ALMA TAP service
2. saving the normalized metadata table as CSV
3. optionally resolving product-level access URLs for selected observations
"""

from __future__ import annotations

import argparse
from pathlib import Path

from almasim.services.metadata.tap import (
    InclusionFilters,
    query_metadata_by_science,
    query_products,
)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--science-keyword",
        action="append",
        default=[],
        help="Science keyword filter. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--scientific-category",
        action="append",
        default=[],
        help="Scientific category filter. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--band",
        action="append",
        type=int,
        default=[],
        help="ALMA band filter. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Optional source-name substring filter for the TAP query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of metadata rows to keep.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=repo_root / "examples" / "output" / "metadata_query_results.csv",
        help="Destination CSV for the normalized metadata table.",
    )
    parser.add_argument(
        "--save-products-csv",
        type=Path,
        default=None,
        help="Optional destination CSV for product-level access URLs.",
    )
    parser.add_argument(
        "--product-member-limit",
        type=int,
        default=1,
        help="How many queried member OUS UIDs to resolve into product rows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    science_keyword = args.science_keyword or ["Galaxies"]
    band = args.band or [6]

    include = InclusionFilters(
        science_keyword=science_keyword,
        scientific_category=args.scientific_category or None,
        band=band,
        source_name=args.source_name,
    )
    metadata = query_metadata_by_science(include=include)
    if metadata.empty:
        raise SystemExit("No metadata rows matched the requested filters")

    if args.limit is not None:
        metadata = metadata.head(args.limit).reset_index(drop=True)

    save_csv = args.save_csv.expanduser().resolve()
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(save_csv, index=False)

    print(f"Saved metadata CSV: {save_csv}")
    print(f"Metadata rows: {len(metadata)}")
    display_columns = [
        col
        for col in ("ALMA_source_name", "Band", "Freq", "member_ous_uid")
        if col in metadata.columns
    ]
    if display_columns:
        print(
            metadata[display_columns].head(min(5, len(metadata))).to_string(index=False)
        )

    if args.save_products_csv is not None:
        member_uids = (
            metadata["member_ous_uid"]
            .dropna()
            .astype(str)
            .head(args.product_member_limit)
            .tolist()
        )
        if not member_uids:
            raise SystemExit("Queried metadata does not include member_ous_uid values")

        products = query_products(member_uids)
        save_products_csv = args.save_products_csv.expanduser().resolve()
        save_products_csv.parent.mkdir(parents=True, exist_ok=True)
        products.to_csv(save_products_csv, index=False)

        print(f"Saved products CSV: {save_products_csv}")
        print(f"Product rows: {len(products)}")


if __name__ == "__main__":
    main()
