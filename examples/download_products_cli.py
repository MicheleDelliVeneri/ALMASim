"""CLI example for resolving and downloading ALMA data products."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from almasim.services.download import (
    PRODUCT_TYPES,
    download_products,
    filter_products,
    format_bytes,
    load_products_csv,
    resolve_products,
    save_products_csv,
)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Metadata CSV containing member_ous_uid rows to resolve.",
    )
    parser.add_argument(
        "--products-csv",
        type=Path,
        default=None,
        help="Previously resolved DataLink products CSV to download directly.",
    )
    parser.add_argument(
        "--member-ous-uid",
        action="append",
        default=[],
        help="Direct member_ous_uid input. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--member-limit",
        type=int,
        default=1,
        help="Maximum number of member_ous_uid rows to resolve from a metadata CSV.",
    )
    parser.add_argument(
        "--product-filter",
        type=str,
        default="all",
        choices=sorted(PRODUCT_TYPES),
        help="Subset of resolved products to download.",
    )
    parser.add_argument(
        "--save-products-csv",
        type=Path,
        default=repo_root / "examples" / "output" / "resolved_products.csv",
        help="Where to save resolved DataLink products before download.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=repo_root / "examples" / "output" / "downloads",
        help="Directory where downloaded files are written.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum number of concurrent downloads.",
    )
    parser.add_argument(
        "--extract-tar",
        action="store_true",
        help="Extract downloaded tar/tgz archives after download.",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Resolve and save products CSV, but do not download files.",
    )
    return parser


def _member_uids_from_metadata(path: Path, limit: int) -> list[str]:
    metadata = pd.read_csv(path.expanduser().resolve())
    if "member_ous_uid" not in metadata.columns:
        raise SystemExit(f"Metadata CSV does not contain member_ous_uid: {path}")
    return metadata["member_ous_uid"].dropna().astype(str).head(limit).tolist()


def _load_or_resolve_products(args: argparse.Namespace):
    if args.products_csv is not None:
        products = load_products_csv(args.products_csv)
        print(f"Loaded products CSV: {args.products_csv.expanduser().resolve()}")
        return products

    member_uids = list(args.member_ous_uid)
    if args.metadata_csv is not None:
        member_uids.extend(_member_uids_from_metadata(args.metadata_csv, args.member_limit))

    member_uids = [uid for uid in member_uids if uid]
    if not member_uids:
        raise SystemExit(
            "Provide --products-csv, --metadata-csv, or at least one --member-ous-uid"
        )

    products = resolve_products(member_uids)
    if not products:
        raise SystemExit("No products were resolved for the requested member_ous_uid values")

    saved_path = save_products_csv(products, args.save_products_csv)
    print(f"Saved resolved products CSV: {saved_path}")
    return products


def main() -> None:
    args = build_parser().parse_args()
    products = _load_or_resolve_products(args)
    filtered = filter_products(products, args.product_filter)
    if not filtered:
        raise SystemExit(f"No products matched filter: {args.product_filter}")

    total_bytes = sum(product.content_length for product in filtered)
    print(f"Resolved products: {len(products)}")
    print(f"Selected for download: {len(filtered)} ({format_bytes(total_bytes)})")

    if args.resolve_only:
        return

    summary = download_products(
        filtered,
        args.destination,
        max_parallel=args.max_parallel,
        extract_tar=args.extract_tar,
        logger_fn=print,
    )
    print(f"Destination: {summary.destination}")
    print(f"Completed: {summary.files_completed}")
    print(f"Failed: {summary.files_failed}")


if __name__ == "__main__":
    main()
