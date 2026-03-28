"""Unit tests for the library-first download service."""

from pathlib import Path

from almasim.services.download import (
    DataProduct,
    download_products,
    filter_products,
    load_products_csv,
    save_products_csv,
)


def test_filter_products():
    """Filtering should preserve only matching product types."""
    products = [
        DataProduct(
            access_url="https://example.org/a.fits",
            uid="uid://A001/X1",
            filename="a.fits",
            content_length=10,
            content_type="application/fits",
            product_type="fits",
        ),
        DataProduct(
            access_url="https://example.org/b.tar",
            uid="uid://A001/X1",
            filename="b.tar",
            content_length=20,
            content_type="application/x-tar",
            product_type="raw",
        ),
    ]

    filtered = filter_products(products, "fits")

    assert len(filtered) == 1
    assert filtered[0].filename == "a.fits"


def test_products_csv_roundtrip(tmp_path):
    """Resolved products should round-trip through CSV."""
    products = [
        DataProduct(
            access_url="https://example.org/a.fits",
            uid="uid://A001/X1",
            filename="a.fits",
            content_length=10,
            content_type="application/fits",
            product_type="fits",
            semantics="alma#imagecube",
        )
    ]
    csv_path = tmp_path / "products.csv"

    save_products_csv(products, csv_path)
    loaded = load_products_csv(csv_path)

    assert len(loaded) == 1
    assert loaded[0].access_url == products[0].access_url
    assert loaded[0].uid == products[0].uid
    assert loaded[0].filename == products[0].filename
    assert loaded[0].product_type == products[0].product_type
    assert loaded[0].semantics == products[0].semantics


def test_download_products_skips_existing_file(tmp_path):
    """Existing files should be treated as completed without network access."""
    destination = tmp_path / "downloads"
    destination.mkdir()
    file_path = destination / "existing.txt"
    file_path.write_bytes(b"already here")

    products = [
        DataProduct(
            access_url="https://example.org/existing.txt",
            uid="uid://A001/X1",
            filename="existing.txt",
            content_length=file_path.stat().st_size,
            content_type="text/plain",
            product_type="auxiliary",
        )
    ]

    summary = download_products(products, destination)

    assert summary.total_files == 1
    assert summary.files_completed == 1
    assert summary.files_failed == 0
    assert summary.files[0].status == "completed"
    assert summary.files[0].bytes_downloaded == file_path.stat().st_size
