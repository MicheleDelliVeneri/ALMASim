#!/usr/bin/env python3
"""Build the trimmed golden-window fixture for the CI regression tier (MADIV-28 T0.13).

The full golden window (uid___A002_X113d9b7_Xb670, 13 GB raw MS) is too large for
a GitHub runner. This keeps one spectral window, a subset of antennas and a few
scans, and packages them as a small ALMA delivery that runs through the same
chain (extract -> unpack -> calibrate -> image):

* Rows are selected by row number and deep-copied with python-casacore, so the
  trimmed MS keeps the original row order, every subtable and every id
  (spectral windows, fields, antennas). The delivered calibration tables and
  their spectral-window maps therefore still apply.
* The delivered ``Pipeline_Final`` flag version holds one FLAG cell per raw-MS
  row, so the same row numbers trim it into a flag version for the trimmed MS.
* ``calapply.txt`` is restricted to the kept spectral window.
* The trimmed MS is exported back to an ASDM with CASA ``exportasdm`` and
  re-imported with the pipeline's own unpack. If the re-imported MS has the
  same rows in the same order, the fixture ships the ASDM (CI then exercises
  importasdm); otherwise it ships the trimmed raw MS.

Run it in Slurm (it reads the 13 GB reference MS)::

    sbatch ... build_ci_fixture.py --work DIR [--antennas 20] [--scans 3,4,6,7,9,10]

The result is DIR/fixture (the directory tree) and DIR/<name>.tar.gz with
SHA256SUMS and a README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import numpy as np

UID = "uid___A002_X113d9b7_Xb670"
PROJECT = "2023.1.00133.S"
MEMBER_PATH = (
    f"{PROJECT}/science_goal.uid___A001_X3621_X39bf/group.uid___A001_X3621_X39c0/"
    "member.uid___A001_X3621_X39c1"
)
CYCLE11 = Path("/SKAO_data/cycle11")
RAW_MS = CYCLE11 / "ms-raw/raw_ms/working" / f"{UID}.ms"
AUX_CALIBRATION = CYCLE11 / "ms-auxiliary" / MEMBER_PATH / "calibration"
CASA_DATA = CYCLE11 / ".casa-data"


def log(message: str) -> None:
    print(message, flush=True)


def select_rows(spw: int, scans: list[int], n_antennas: int, flag_table: Path):
    """Row numbers of the trimmed MS and the antennas kept."""
    from casacore.tables import table, taql

    raw = table(str(RAW_MS), ack=False, lockoptions="usernoread")
    dd = table(str(RAW_MS / "DATA_DESCRIPTION"), ack=False)
    ddids = [i for i, s in enumerate(dd.getcol("SPECTRAL_WINDOW_ID")) if s == spw]
    dd.close()
    scan_list = ",".join(str(s) for s in scans)
    ddid_list = ",".join(str(d) for d in ddids)
    selected = taql(
        f"select rowid() as row, ANTENNA1, ANTENNA2 from $raw "
        f"where DATA_DESC_ID in [{ddid_list}] and SCAN_NUMBER in [{scan_list}]"
    )
    rows = selected.getcol("row")
    ant1, ant2 = selected.getcol("ANTENNA1"), selected.getcol("ANTENNA2")
    log(f"spw {spw} (DATA_DESC_ID {ddids}), scans {scans}: {len(rows)} rows")

    # Flagged fraction per antenna under the delivered Pipeline_Final flags.
    flags = table(str(flag_table), ack=False)
    n_ant = table(str(RAW_MS / "ANTENNA"), ack=False).nrows()
    flagged = np.zeros(n_ant)
    total = np.zeros(n_ant)
    for start in range(0, len(rows), 200_000):
        chunk = rows[start : start + 200_000]
        f = flags.selectrows(chunk.tolist()).getcol("FLAG")
        frac = f.reshape(len(chunk), -1).mean(axis=1)
        for a in (ant1[start : start + 200_000], ant2[start : start + 200_000]):
            np.add.at(flagged, a, frac)
            np.add.at(total, a, 1)
    flags.close()
    fraction = np.divide(flagged, total, out=np.ones(n_ant), where=total > 0)

    # Spread the kept antennas over the array: sort usable antennas by distance
    # from the array centre and take evenly spaced ones.
    positions = table(str(RAW_MS / "ANTENNA"), ack=False).getcol("POSITION")
    radius = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
    usable = [a for a in np.argsort(radius) if total[a] > 0 and fraction[a] < 0.5]
    picks = np.linspace(0, len(usable) - 1, num=min(n_antennas, len(usable))).round()
    keep = sorted(int(usable[int(i)]) for i in np.unique(picks))
    log(f"antennas kept ({len(keep)}): {keep}")

    mask = np.isin(ant1, keep) & np.isin(ant2, keep)
    raw.close()
    return np.sort(rows[mask]), keep


def deep_copy_rows(source: Path, rows: np.ndarray, destination: Path) -> None:
    from casacore.tables import table

    tab = table(str(source), ack=False, lockoptions="usernoread")
    tab.selectrows(rows.tolist()).copy(str(destination), deep=True)
    tab.close()


def trim_pointing(ms: Path, antennas: list[int], margin_s: float = 60.0) -> None:
    """Keep POINTING rows of the kept antennas around the kept data (506 MB in full)."""
    from casacore.tables import table

    main = table(str(ms), ack=False)
    time = main.getcol("TIME")
    main.close()
    pointing = ms / "POINTING"
    tab = table(str(pointing), ack=False)
    keep = np.flatnonzero(
        np.isin(tab.getcol("ANTENNA_ID"), antennas)
        & (tab.getcol("TIME") >= time.min() - margin_s)
        & (tab.getcol("TIME") <= time.max() + margin_s)
    )
    log(f"POINTING: keeping {len(keep)} of {tab.nrows()} rows")
    trimmed = ms / "POINTING.trimmed"
    tab.selectrows(keep.tolist()).copy(str(trimmed), deep=True)
    tab.close()
    shutil.rmtree(pointing)
    trimmed.rename(pointing)


def build(args: argparse.Namespace) -> int:
    work = args.work.resolve()
    fixture = work / "fixture"
    if fixture.exists():
        shutil.rmtree(fixture)
    calibration = fixture / "auxiliary" / MEMBER_PATH / "calibration"
    calibration.mkdir(parents=True)
    scratch = work / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    # Delivered flag versions for the raw MS.
    with tarfile.open(AUX_CALIBRATION / f"{UID}.ms.flagversions.tgz") as tar:
        tar.extractall(scratch, filter="data")
    delivered = scratch / f"{UID}.ms.flagversions"

    rows, antennas = select_rows(
        args.spw, args.scans, args.antennas, delivered / "flags.Pipeline_Final"
    )
    trimmed = scratch / "trimmed" / f"{UID}.ms"
    if trimmed.parent.exists():
        shutil.rmtree(trimmed.parent)
    trimmed.parent.mkdir(parents=True)
    log(f"deep copy of {len(rows)} rows -> {trimmed}")
    deep_copy_rows(RAW_MS, rows, trimmed)
    trim_pointing(trimmed, antennas)

    # Pipeline_Final for the trimmed rows, delivered the way ALMA delivers it.
    versions = scratch / "flagversions" / f"{UID}.ms.flagversions"
    if versions.parent.exists():
        shutil.rmtree(versions.parent)
    versions.mkdir(parents=True)
    deep_copy_rows(delivered / "flags.Pipeline_Final", rows, versions / "flags.Pipeline_Final")
    shutil.copy2(delivered / "FLAG_VERSION_LIST", versions / "FLAG_VERSION_LIST")
    listed = (versions / "FLAG_VERSION_LIST").read_text()
    (versions / "FLAG_VERSION_LIST").write_text(
        "\n".join(line for line in listed.splitlines() if line.startswith("Pipeline_Final")) + "\n"
    )
    with tarfile.open(calibration / f"{UID}.ms.flagversions.tgz", "w:gz") as tar:
        tar.add(versions, arcname=versions.name)

    # calapply restricted to the kept spectral window; caltables as delivered.
    calapply = (AUX_CALIBRATION / f"{UID}.ms.calapply.txt").read_text()
    spw_pattern = re.compile(r"spw='[0-9,]+'")
    assert spw_pattern.search(calapply), "no spw selection in calapply.txt"
    (calibration / f"{UID}.ms.calapply.txt").write_text(
        spw_pattern.sub(f"spw='{args.spw}'", calapply)
    )
    for caltables in sorted(AUX_CALIBRATION.glob("*caltables.tgz")):
        shutil.copy2(caltables, calibration / caltables.name)

    raw_kind = try_asdm_round_trip(trimmed, scratch, args) if args.asdm else "ms"
    raw_dir = fixture / "raw"
    raw_dir.mkdir()
    if raw_kind == "asdm":
        asdm = scratch / "asdm" / f"{UID}.asdm.sdm"
        archive = raw_dir / f"{PROJECT}_{UID}.asdm.sdm.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(asdm, arcname=f"{MEMBER_PATH}/raw/{asdm.name}")
    else:
        archive = raw_dir / f"{UID}.ms.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(trimmed, arcname=trimmed.name)

    write_readme(fixture, raw_kind, rows, antennas, args)
    sums = []
    for path in sorted(p for p in fixture.rglob("*") if p.is_file()):
        sums.append(f"{sha256(path)}  {path.relative_to(fixture)}")
    (fixture / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    bundle = work / f"{args.name}.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=6) as tar:
        tar.add(fixture, arcname=args.name)
    (work / f"{bundle.name}.sha256").write_text(f"{sha256(bundle)}  {bundle.name}\n")
    summary = {
        "bundle": str(bundle),
        "bundle_bytes": bundle.stat().st_size,
        "raw_kind": raw_kind,
        "rows": int(len(rows)),
        "antennas": antennas,
        "scans": args.scans,
        "spw": args.spw,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(json.dumps(summary, indent=1))
    return 0


def try_asdm_round_trip(trimmed: Path, scratch: Path, args: argparse.Namespace) -> str:
    """Export the trimmed MS to an ASDM and re-import it; 'asdm' if rows survive in order."""
    from casacore.tables import table

    asdm_dir = scratch / "asdm"
    if asdm_dir.exists():
        shutil.rmtree(asdm_dir)
    asdm_dir.mkdir()
    asdm = asdm_dir / f"{UID}.asdm.sdm"
    # casatasks must not share a process with python-casacore. The child sets CASA
    # up the way the pipeline does (CASASITECONFIG, its own matplotlib cache);
    # without that the casatools import can hang.
    code = textwrap.dedent(f"""
        from almasim.services.archive.unpack_ms import configure_casa_environment
        configure_casa_environment({str(scratch)!r}, {str(CASA_DATA)!r},
                                   log_file={str(scratch / "casa-exportasdm.log")!r})
        from casatasks import exportasdm
        exportasdm(vis={str(trimmed)!r}, asdm={str(asdm)!r}, datacolumn='DATA')
    """)
    try:
        result = subprocess.run([sys.executable, "-c", code], cwd=scratch, timeout=3600)
    except subprocess.TimeoutExpired:
        log("exportasdm did not finish within an hour; shipping the trimmed MS")
        return "ms"
    if result.returncode != 0 or not asdm.is_dir():
        log(f"exportasdm failed (exit {result.returncode}); shipping the trimmed MS")
        return "ms"

    # Re-import exactly as CI will: through almasim's unpack.
    staged = scratch / "reimport-input" / MEMBER_PATH / "raw"
    if staged.parent.parent.exists():
        shutil.rmtree(scratch / "reimport-input")
    staged.mkdir(parents=True)
    os.symlink(asdm, staged / asdm.name)
    out = scratch / "reimport"
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        sys.executable, "-m", "almasim.cli", "visibilities", "unpack",
        "--input-root", str(scratch / "reimport-input"), "--output-root", str(out),
        "--asdm-uid", UID, "--postprocess-backend", "sync",
        "--casa-data-root", str(CASA_DATA), "--skip-casa-data-update",
    ]  # fmt: skip
    result = subprocess.run(cmd, cwd=scratch)
    reimported = out / "working" / f"{UID}.ms"
    if result.returncode != 0 or not reimported.is_dir():
        log(f"re-import failed (exit {result.returncode}); shipping the trimmed MS")
        return "ms"

    a = table(str(trimmed), ack=False)
    b = table(str(reimported), ack=False)
    same = a.nrows() == b.nrows() and all(
        np.array_equal(a.getcol(c), b.getcol(c))
        for c in ("TIME", "ANTENNA1", "ANTENNA2", "DATA_DESC_ID", "FIELD_ID", "SCAN_NUMBER")
    )
    log(f"re-imported MS: {b.nrows()} rows vs {a.nrows()}; same row keys in order: {same}")
    if same:
        diff = max(
            float(np.abs(a.getcol("DATA", s, 100_000) - b.getcol("DATA", s, 100_000)).max())
            for s in range(0, a.nrows(), 100_000)
        )
        log(f"max |DATA difference| after the round trip: {diff:.3g}")
    a.close()
    b.close()
    return "asdm" if same else "ms"


def write_readme(fixture: Path, raw_kind: str, rows, antennas, args) -> None:
    raw_line = (
        f"`raw/{PROJECT}_{UID}.asdm.sdm.tar`: the trimmed data as an ALMA archive tar "
        "(exported with CASA exportasdm)."
        if raw_kind == "asdm"
        else f"`raw/{UID}.ms.tar`: the trimmed raw MeasurementSet."
    )
    (fixture / "README.md").write_text(
        textwrap.dedent(f"""\
        # ALMASim CI golden fixture ({args.name})

        A trimmed copy of ALMA execution block `{UID}` (project {PROJECT}, member
        uid://A001/X3621/X39c1), used by the ALMASim CI regression tier (MADIV-28).
        It runs through extract -> unpack -> calibrate -> image -> clean-quality on
        a GitHub runner.

        * {raw_line}
        * `auxiliary/{MEMBER_PATH}/calibration/`: the delivered calibration tables,
          `calapply.txt` restricted to spectral window {args.spw}, and the delivered
          `Pipeline_Final` flag version trimmed to the same rows.

        Trimming: spectral window {args.spw}; scans {args.scans}; {len(antennas)} antennas
        (ids {antennas}); {len(rows)} rows, selected by row number from the raw MS
        and deep-copied, so row order, subtables and all ids are unchanged.
        Built by `scripts/regression/build_ci_fixture.py`.

        This is not science-grade data: the trimming lowers the sensitivity, and CI
        runs self-calibration with a lowered S/N threshold.

        ## Acknowledgement

        This fixture makes use of the following ALMA data: ADS/JAO.ALMA#{PROJECT}.
        ALMA is a partnership of ESO (representing its member states), NSF (USA) and
        NINS (Japan), together with NRC (Canada), NSTC and ASIAA (Taiwan), and KASI
        (Republic of Korea), in cooperation with the Republic of Chile. The Joint ALMA
        Observatory is operated by ESO, AUI/NRAO and NAOJ.
        """)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--name", default="almasim-golden-ci-v1")
    parser.add_argument("--spw", type=int, default=31)
    parser.add_argument("--scans", type=lambda s: [int(x) for x in s.split(",")],
                        default=[3, 4, 6, 7, 9, 10])  # fmt: skip
    parser.add_argument("--antennas", type=int, default=20)
    parser.add_argument("--no-asdm", dest="asdm", action="store_false",
                        help="Ship the trimmed MS without trying the ASDM round trip.")  # fmt: skip
    return build(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
