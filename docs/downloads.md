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

## Skipping execution blocks ALMA never calibrated

Every execution block ships a `<eb>.qa0_report.pdf` in its member OUS `qa/`
directory, inside the *auxiliary* tar. Its **QA0 Status** decides whether ALMA
delivers calibration at all: `Pass` blocks always carry calibration products,
`SemiPass` blocks never do. ALMA still ships the raw ASDM of a SemiPass block, so
downloading raw products blindly pulls execution blocks (about 12 % in Cycle 11)
that unpack fine and then can never be calibrated.

`almasim products download --skip-qa0-semipass` reads those reports and drops
the SemiPass raw ASDMs before the transfer. The reports have to be on disk
first, so download in two passes:

```bash
# 1. auxiliary products first: small, and they carry the QA0 reports
almasim products download --products-csv resolved.csv \
  --product-filter auxiliary --destination downloads --extract-tar --yes

# 2. raw ASDMs, minus the SemiPass execution blocks
almasim products download --products-csv resolved.csv \
  --product-filter raw --destination downloads --extract-tar \
  --skip-qa0-semipass --yes
```

The reports are looked up under `--qa0-report-root` (default: `--destination`)
in the standard `<project>/science_goal.*/group.*/member.*/qa/` layout. A raw
product whose report is missing or unreadable is **kept**, never dropped, and
counted in the summary line so a wrong root is visible immediately. Do not use
`ExecBlock Status` for this decision; it does not discriminate.

The service-level entry point is `almasim.services.archive.filter_products_by_qa0`.

## Frontend and Backend

The frontend exposes download workflows, while the backend acts as an adapter over the shared service layer.

## Example Scripts

- [`examples/download_products_cli.py`](../examples/download_products_cli.py)
- [`examples/download_products_notebook.ipynb`](../examples/download_products_notebook.ipynb)
