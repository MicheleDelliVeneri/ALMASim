"""Download records and the CSV catalogue (MADIV-28 T8.2: a direct test)."""

import pytest

from almasim.services.download.catalog_io import load_products_csv, save_products_csv
from almasim.services.download.models import DataProduct, format_bytes

pytestmark = pytest.mark.unit


def test_format_bytes_uses_decimal_units():
    assert format_bytes(0) == "0.0 B"
    assert format_bytes(1500) == "1.5 KB"
    assert format_bytes(2_000_000_000) == "2.0 GB"
    assert format_bytes(5 * 10**15) == "5.0 PB"


def test_products_round_trip_through_the_csv(tmp_path):
    products = [
        DataProduct(
            access_url="https://example.org/a.tar",
            uid="uid://A001/X1/X2",
            filename="a.tar",
            content_length=10,
            content_type="application/x-tar",
            product_type="raw",
        )
    ]
    path = save_products_csv(products, tmp_path / "products.csv")
    assert load_products_csv(path) == products
