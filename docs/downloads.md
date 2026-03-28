# Downloads

This page describes ALMASim’s data-download workflow.

## Location

The download service lives in [`src/almasim/services/download.py`](../src/almasim/services/download.py).

## Capabilities

ALMASim can:

- resolve product rows from metadata
- save resolved products to CSV
- reload saved product CSVs
- filter download targets
- download products with a Python-first service layer

## Download Model

The intended workflow is:

1. query metadata
2. resolve products for selected observations
3. inspect or save the product list
4. download selected products

## Frontend and Backend

The frontend exposes download workflows, while the backend acts as an adapter over the shared service layer.

## Example Scripts

- [`examples/download_products_cli.py`](../examples/download_products_cli.py)
- [`examples/download_products_notebook.ipynb`](../examples/download_products_notebook.ipynb)
