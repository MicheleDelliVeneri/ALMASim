#!/usr/bin/env python3
"""Compare pipeline products with reference products (MADIV-28 regression gate).

Every refactor must leave the science products unchanged, so this script
compares, bit for bit unless a tolerance is given:

* MeasurementSets and other casacore tables: row counts, column names, a
  SHA-256 of every column (read in chunks), column keywords and all subtables;
  ``<ms>.flagversions`` as well when present.
* FITS files: the set of files, every HDU's header (output paths normalised)
  and its data, including binary tables such as the per-channel ``BEAMS``.
* JSON records (``quality.json``, ``.done`` markers) after dropping timing
  fields, CSV and text files, and the ``#`` annotation lines of task logs.

Paths that legitimately differ between a run and its reference (the run
directory, the calibrated MS location) are normalised with ``--map TOKEN NEW
REF[,REF...]``: every occurrence of NEW or of any REF in a compared string is
replaced by ``<TOKEN>`` before comparing.

A casacore table digest can be saved once with ``digest`` and given in place
of the reference table, so a run only reads its own products.

Usage::

    compare_products.py digest TABLE OUT.json
    compare_products.py table NEW REF_TABLE_OR_DIGEST [--report R]
    compare_products.py tree NEW_DIR REF_DIR [--map ...] [--report R]
    compare_products.py golden --run RUN --uid UID --task SPW-31_FIELD-2 \\
        --ref-raw REF_RAW_MS_OR_DIGEST --ref-cal REF_CAL_MS_OR_DIGEST \\
        --ref-task REF_TASK_DIR [--map ...] [--report R]

Exit status is 0 when everything matches and 1 otherwise; the report lists
one line per check and ends with ``PASS`` or ``FAIL``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CHUNK_BYTES = 64 * 1024 * 1024
# Fields that record when or how long something ran, not what it produced.
VOLATILE_JSON_KEYS = {"completed_at", "failed_at", "started_at", "seconds", "elapsed_s"}
# Header cards that record when a file was written.
VOLATILE_FITS_KEYS = {"DATE"}
# casacore subtables that record processing history (timestamps, hosts, paths).
VOLATILE_SUBTABLES = {"HISTORY"}
# Files whose content is timestamped by the tools that write them.
SKIPPED_SUFFIXES = {".log", ".last", ".pre"}
DURATION = re.compile(r"\b\d+(\.\d+)?\s?s\b")


@dataclass
class Report:
    lines: list[tuple[bool, str, str]] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.lines.append((ok, name, detail))
        return ok

    @property
    def passed(self) -> bool:
        return bool(self.lines) and all(ok for ok, _, _ in self.lines)

    def render(self) -> str:
        out = [
            f"- {'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else "")
            for ok, name, detail in self.lines
        ]
        failed = sum(not ok for ok, _, _ in self.lines)
        out.append("")
        out.append(
            f"{'PASS' if self.passed else 'FAIL'}: {len(self.lines)} checks, {failed} failed"
        )
        return "\n".join(out) + "\n"


class PathMap:
    """Replace run-specific and reference-specific path prefixes by tokens."""

    def __init__(self, maps: list[tuple[str, str, str]] | None = None) -> None:
        pairs: list[tuple[str, str]] = []
        for token, new, refs in maps or []:
            for path in [new, *refs.split(",")]:
                if path:
                    pairs.append((path.rstrip("/"), f"<{token}>"))
        # Longest prefixes first so nested paths map to the most specific token.
        self.pairs = sorted(pairs, key=lambda p: len(p[0]), reverse=True)

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            for path, token in self.pairs:
                value = value.replace(path, token)
            return value
        if isinstance(value, dict):
            return {k: self(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self(v) for v in value]
        return value


# --------------------------------------------------------------------------
# casacore tables


def _hash_array(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray) and value.dtype.kind not in "OUS":
        array = np.ascontiguousarray(value)
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        digest.update(array.tobytes())
    else:
        digest.update(json.dumps(value, default=_json_default, sort_keys=True).encode())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _column_digest(table: Any, column: str) -> str:
    digest = hashlib.sha256()
    nrows = table.nrows()
    if nrows == 0:
        return digest.hexdigest()
    try:
        first = table.getcell(column, 0)
        row_bytes = max(1, getattr(first, "nbytes", 64))
    except RuntimeError:
        row_bytes = 64
    step = max(1, CHUNK_BYTES // row_bytes)
    for start in range(0, nrows, step):
        count = min(step, nrows - start)
        try:
            _hash_array(digest, table.getcol(column, startrow=start, nrow=count))
        except RuntimeError:
            # Variable-shape or partly undefined cells: hash row by row.
            cells = table.getvarcol(column, startrow=start, nrow=count)
            for key in sorted(cells, key=lambda k: int(k.lstrip("r"))):
                _hash_array(digest, cells[key])
    return digest.hexdigest()


def table_digest(path: Path, names: PathMap | None = None) -> dict[str, Any]:
    """Return row count, per-column SHA-256 and keywords of a table and its subtables."""
    from casacore.tables import table as open_table

    names = names or PathMap()
    tab = open_table(str(path), readonly=True, ack=False, lockoptions="usernoread")
    try:
        subtables = sorted(tab.getsubtables())
        subtable_names = {Path(s).name for s in subtables}

        def clean(value: Any) -> Any:
            # Keywords referring to subtables hold absolute paths.
            if isinstance(value, str) and value.startswith("Table: "):
                return "Table: " + Path(value[len("Table: ") :]).name
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            return names(value)

        keywords = {k: clean(v) for k, v in tab.getkeywords().items() if k not in subtable_names}
        columns = {}
        for column in sorted(tab.colnames()):
            columns[column] = {
                "sha256": _column_digest(tab, column),
                "keywords": clean(tab.getcolkeywords(column)),
            }
        result: dict[str, Any] = {
            "nrows": tab.nrows(),
            "keywords": json.loads(json.dumps(keywords, default=_json_default)),
            "columns": json.loads(json.dumps(columns, default=_json_default)),
            "subtables": {},
        }
    finally:
        tab.close()
    for sub in subtables:
        name = Path(sub).name
        if name in VOLATILE_SUBTABLES:
            continue
        result["subtables"][name] = table_digest(Path(sub), names)
    flagversions = Path(str(path) + ".flagversions")
    if flagversions.is_dir():
        listing = flagversions / "FLAG_VERSION_LIST"
        result["flagversions"] = {
            "list": listing.read_text().splitlines() if listing.exists() else [],
            "tables": {
                p.name: table_digest(p, names)
                for p in sorted(flagversions.iterdir())
                if p.is_dir() and p.name.startswith("flags.")
            },
        }
    return result


def load_digest(path: Path, names: PathMap | None = None) -> dict[str, Any]:
    if path.suffix == ".json" and path.is_file():
        return json.loads(path.read_text())
    return table_digest(path, names)


def compare_digests(report: Report, label: str, new: dict[str, Any], ref: dict[str, Any]) -> None:
    report.check(f"{label} rows", new["nrows"] == ref["nrows"], f"{new['nrows']} vs {ref['nrows']}")
    new_cols, ref_cols = set(new["columns"]), set(ref["columns"])
    report.check(
        f"{label} columns",
        new_cols == ref_cols,
        _set_detail(new_cols, ref_cols),
    )
    differing = [c for c in sorted(new_cols & ref_cols) if new["columns"][c] != ref["columns"][c]]
    report.check(f"{label} column data", not differing, ", ".join(differing))
    report.check(f"{label} keywords", new["keywords"] == ref["keywords"])
    new_subs, ref_subs = set(new["subtables"]), set(ref["subtables"])
    report.check(f"{label} subtables", new_subs == ref_subs, _set_detail(new_subs, ref_subs))
    for name in sorted(new_subs & ref_subs):
        compare_digests(report, f"{label}/{name}", new["subtables"][name], ref["subtables"][name])
    if "flagversions" in new or "flagversions" in ref:
        new_fv = new.get("flagversions", {"list": [], "tables": {}})
        ref_fv = ref.get("flagversions", {"list": [], "tables": {}})
        report.check(
            f"{label} flag versions",
            new_fv["list"] == ref_fv["list"],
            f"{new_fv['list']} vs {ref_fv['list']}",
        )
        for name in sorted(set(new_fv["tables"]) & set(ref_fv["tables"])):
            compare_digests(
                report,
                f"{label}.flagversions/{name}",
                new_fv["tables"][name],
                ref_fv["tables"][name],
            )


def _set_detail(new: set[str], ref: set[str]) -> str:
    parts = []
    if new - ref:
        parts.append("extra " + ", ".join(sorted(new - ref)))
    if ref - new:
        parts.append("missing " + ", ".join(sorted(ref - new)))
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Files: FITS, JSON, CSV/text, logs


def _arrays_equal(new: np.ndarray, ref: np.ndarray, tolerance: float) -> tuple[bool, str]:
    if new.shape != ref.shape or new.dtype != ref.dtype:
        return False, f"shape/dtype {new.shape} {new.dtype} vs {ref.shape} {ref.dtype}"
    if new.dtype.kind not in "fc":
        equal = np.array_equal(new, ref)
        return equal, "" if equal else "values differ"
    if np.array_equal(new, ref, equal_nan=True):
        return True, ""
    if not np.array_equal(np.isnan(new), np.isnan(ref)):
        return False, "NaN positions differ"
    diff = np.nanmax(np.abs(new - ref))
    scale = float(np.nanstd(ref)) or 1.0
    detail = f"max |diff| {diff:.3g} = {diff / scale:.3g} x std(ref)"
    return bool(tolerance and diff <= tolerance * scale), detail


def compare_fits(
    report: Report, label: str, new: Path, ref: Path, names: PathMap, tolerance: float
) -> None:
    from astropy.io import fits

    with fits.open(new, memmap=False) as new_hdus, fits.open(ref, memmap=False) as ref_hdus:
        if not report.check(
            f"{label} HDUs", len(new_hdus) == len(ref_hdus), f"{len(new_hdus)} vs {len(ref_hdus)}"
        ):
            return
        for index, (a, b) in enumerate(zip(new_hdus, ref_hdus, strict=True)):
            hdu = f"{label}[{index}]"
            header_a = [
                (c.keyword, names(c.value))
                for c in a.header.cards
                if c.keyword not in VOLATILE_FITS_KEYS
            ]
            header_b = [
                (c.keyword, names(c.value))
                for c in b.header.cards
                if c.keyword not in VOLATILE_FITS_KEYS
            ]
            changed = sorted({k for k, v in set(header_a) ^ set(header_b)})
            report.check(f"{hdu} header", header_a == header_b, ", ".join(changed))
            if isinstance(a, fits.BinTableHDU) or isinstance(b, fits.BinTableHDU):
                cols_a = a.columns.names if a.data is not None else []
                cols_b = b.columns.names if b.data is not None else []
                ok = cols_a == cols_b and all(
                    _arrays_equal(np.asarray(a.data[c]), np.asarray(b.data[c]), tolerance)[0]
                    for c in cols_a
                )
                report.check(f"{hdu} table", ok)
            elif a.data is None or b.data is None:
                report.check(f"{hdu} data", a.data is None and b.data is None)
            else:
                ok, detail = _arrays_equal(np.asarray(a.data), np.asarray(b.data), tolerance)
                report.check(f"{hdu} data", ok, detail)


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items() if k not in VOLATILE_JSON_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def compare_json(report: Report, label: str, new: Path, ref: Path, names: PathMap) -> None:
    a = names(_strip_volatile(json.loads(new.read_text())))
    b = names(_strip_volatile(json.loads(ref.read_text())))
    report.check(label, a == b, "" if a == b else _first_json_difference(a, b))


def _first_json_difference(a: Any, b: Any, path: str = "") -> str:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                return _first_json_difference(a.get(key), b.get(key), f"{path}.{key}")
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if x != y:
                return _first_json_difference(x, y, f"{path}[{i}]")
    return f"{path or '<root>'}: {str(a)[:120]!r} vs {str(b)[:120]!r}"


def compare_text(report: Report, label: str, new: Path, ref: Path, names: PathMap) -> None:
    a, b = names(new.read_text()), names(ref.read_text())
    report.check(label, a == b)


def log_annotations(path: Path, names: PathMap) -> list[str]:
    """The ``#`` lines a task log records (commands, row counts), without timings."""
    lines = [
        line for line in path.read_text(errors="replace").splitlines() if line.startswith("# ")
    ]
    return [DURATION.sub("<t>", names(line)) for line in lines]


def _is_table(path: Path) -> bool:
    return path.is_dir() and (path / "table.dat").exists()


def compare_tree(
    report: Report,
    new_root: Path,
    ref_root: Path,
    names: PathMap,
    tolerance: float = 0.0,
    label: str = "",
) -> None:
    """Compare every product file below two directories."""

    def products(root: Path) -> dict[str, Path]:
        found: dict[str, Path] = {}
        stack = [root]
        while stack:
            current = stack.pop()
            for entry in sorted(current.iterdir()):
                rel = entry.relative_to(root).as_posix()
                if _is_table(entry):
                    found[rel] = entry
                elif entry.is_dir():
                    stack.append(entry)
                elif entry.suffix not in SKIPPED_SUFFIXES:
                    found[rel] = entry
        return found

    new_files, ref_files = products(new_root), products(ref_root)
    prefix = f"{label}/" if label else ""
    report.check(
        f"{prefix}file set",
        set(new_files) == set(ref_files),
        _set_detail(set(new_files), set(ref_files)),
    )
    for rel in sorted(set(new_files) & set(ref_files)):
        a, b = new_files[rel], ref_files[rel]
        name = prefix + rel
        if _is_table(a):
            compare_digests(report, name, table_digest(a, names), table_digest(b, names))
        elif a.suffix == ".fits":
            compare_fits(report, name, a, b, names, tolerance)
        elif a.suffix in {".json", ".done", ".failed"}:
            compare_json(report, name, a, b, names)
        else:
            compare_text(report, name, a, b, names)


# --------------------------------------------------------------------------
# Command line


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--map",
        nargs=3,
        action="append",
        default=[],
        metavar=("TOKEN", "NEW", "REF[,REF...]"),
        help="Normalise these path prefixes to <TOKEN> before comparing.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Accept float data differing by at most this many standard "
        "deviations of the reference (default 0: bitwise equal).",
    )
    parser.add_argument("--report", type=Path, help="Also write the report to this file.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("digest", help="Save a casacore table digest as JSON.")
    p.add_argument("table", type=Path)
    p.add_argument("output", type=Path)
    _add_common(p)

    p = sub.add_parser("table", help="Compare a casacore table with a table or digest.")
    p.add_argument("new", type=Path)
    p.add_argument("ref", type=Path)
    _add_common(p)

    p = sub.add_parser("tree", help="Compare every product below two directories.")
    p.add_argument("new", type=Path)
    p.add_argument("ref", type=Path)
    _add_common(p)

    p = sub.add_parser("golden", help="Compare one golden run with its references.")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--uid", required=True)
    p.add_argument("--task", required=True, help="e.g. SPW-31_FIELD-2")
    p.add_argument("--ref-raw", type=Path, help="Reference raw MS or its digest.")
    p.add_argument("--ref-cal", type=Path, help="Reference calibrated MS or its digest.")
    p.add_argument("--ref-task", type=Path, required=True, help="Reference task directory.")
    _add_common(p)

    args = parser.parse_args(argv)
    names = PathMap([tuple(m) for m in args.map])
    report = Report()

    if args.command == "digest":
        args.output.write_text(json.dumps(table_digest(args.table, names), indent=1) + "\n")
        print(f"wrote {args.output}")
        return 0
    if args.command == "table":
        compare_digests(
            report, args.new.name, table_digest(args.new, names), load_digest(args.ref, names)
        )
    elif args.command == "tree":
        compare_tree(report, args.new, args.ref, names, args.tolerance)
    else:
        run, uid = args.run, args.uid
        if args.ref_raw:
            compare_digests(
                report,
                "raw MS",
                table_digest(run / "raw_ms" / "working" / f"{uid}.ms", names),
                load_digest(args.ref_raw, names),
            )
        if args.ref_cal:
            compare_digests(
                report,
                "calibrated MS",
                table_digest(run / "cal" / f"{uid}.ms.split.cal", names),
                load_digest(args.ref_cal, names),
            )
        task_dir = run / "cubes" / f"{uid}.ms.split" / args.task
        compare_tree(report, task_dir, args.ref_task, names, args.tolerance, label=args.task)
        for suffix in (".done",):
            new_marker = task_dir.parent / f"{args.task}{suffix}"
            ref_marker = args.ref_task.parent / f"{args.task}{suffix}"
            if report.check(f"{args.task}{suffix} exists", new_marker.exists()):
                compare_json(report, f"{args.task}{suffix}", new_marker, ref_marker, names)
        new_log = task_dir.parent / f"{args.task}.log"
        ref_log = args.ref_task.parent / f"{args.task}.log"
        if report.check(f"{args.task}.log exists", new_log.exists()):
            a, b = log_annotations(new_log, names), log_annotations(ref_log, names)
            report.check(
                f"{args.task}.log annotations",
                a == b,
                "" if a == b else _first_json_difference(a, b),
            )

    text = report.render()
    print(text, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
