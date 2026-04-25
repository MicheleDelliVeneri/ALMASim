# Staged API Examples

This directory contains runnable examples for the staged simulation API:

- `query_metadata_cli.py`
  Query ALMA metadata from TAP, save it as CSV, and optionally save product-level access URLs for selected observations.
- `query_metadata_notebook.ipynb`
  Notebook walkthrough for querying metadata, previewing results, and exporting CSV files for later simulation or download workflows.
- `download_products_cli.py`
  Resolve DataLink products from `member_ous_uid` rows or a saved products CSV, then download them with a library-first Python API.
- `archive_ms_cli.py`
  Imports raw ALMA ASDMs into MeasurementSets, then applies delivered calibration products and writes calibrated science MSs.
- `download_products_notebook.ipynb`
  Notebook walkthrough for resolving products from queried metadata and downloading them to disk.
- `staged_pipeline_cli.py`
  Queries metadata or loads a CSV, then runs clean-cube generation, in-memory observation simulation, and ML shard export from the command line.
- `staged_pipeline_notebook.ipynb`
  Notebook walkthrough of the same pipeline for interactive exploration and DDRM data preparation, starting from a live metadata query.
- `imaging_cli.py`
  Builds a synthetic clean/dirty/beam cube triplet, saves the products expected by the imaging page, runs iterative deconvolution, and checks that the restored cube is closer to the clean cube than the dirty cube.

The examples use the in-process `sync` compute backend so they do not require Dask, process pools, or a running scheduler.

Metadata query CLI usage:

```bash
python examples/query_metadata_cli.py \
  --science-keyword Galaxies \
  --band 6 \
  --save-csv examples/output/metadata_query_results.csv \
  --save-products-csv examples/output/metadata_products.csv
```

Download CLI usage:

```bash
python examples/download_products_cli.py \
  --metadata-csv examples/output/metadata_query_results.csv \
  --member-limit 1 \
  --product-filter all \
  --save-products-csv examples/output/resolved_products.csv \
  --destination examples/output/downloads \
  --extract-tar
```

Download, unpack raw MSs, and generate calibrated visibilities:

```bash
python examples/download_products_cli.py \
  --products-csv examples/output/resolved_products.csv \
  --destination examples/output/downloads \
  --extract-tar \
  --unpack-ms \
  --generate-calibrated-visibilities \
  --archive-output-root examples/output/archive_ms
```

Archive MS CLI usage:

```bash
python examples/archive_ms_cli.py \
  --input-root /path/to/2023.1.01196.S \
  --output-root examples/output/archive_ms
```

Staged pipeline CLI usage:

```bash
python examples/staged_pipeline_cli.py \
  --science-keyword Galaxies \
  --band 6 \
  --row-idx 0 \
  --project-name ddrm_demo \
  --save-metadata-csv examples/output/staged_pipeline_metadata.csv \
  --ml-shard-path examples/output/ddrm_training_sample.h5
```

Imaging CLI usage:

```bash
python examples/imaging_cli.py \
  --output-dir examples/output/imaging_demo \
  --cycles 180 \
  --gain 0.12
```
