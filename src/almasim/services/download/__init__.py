"""Library-first download helpers for ALMA data products.

Split into submodules by MADIV-28 T6.9: ``models``, ``datalink`` (resolve and
filter), ``catalog_io`` (products CSV), ``transfer`` (single files, retries,
mirrors) and ``orchestrate`` (``download_products``).
"""

from .catalog_io import load_products_csv, products_to_dataframe, save_products_csv
from .datalink import filter_products, resolve_products
from .models import PRODUCT_TYPES, DataProduct, DownloadSummary, FileDownloadStatus, format_bytes
from .orchestrate import _cleanup_download_inputs as _cleanup_download_inputs
from .orchestrate import download_products
from .transfer import MAX_PARALLEL_PER_MIRROR as MAX_PARALLEL_PER_MIRROR
from .transfer import MAX_PARALLEL_TOTAL as MAX_PARALLEL_TOTAL

__all__ = [
    "DataProduct",
    "FileDownloadStatus",
    "DownloadSummary",
    "PRODUCT_TYPES",
    "format_bytes",
    "resolve_products",
    "filter_products",
    "products_to_dataframe",
    "save_products_csv",
    "load_products_csv",
    "download_products",
]
