"""Make the simulated counterpart of a real spectral window's MeasurementSet.

    python counterpart.py base REAL_MS SPW FIELD BASE_MS
        Split the window and field out of the real MS (CASA split, flags kept),
        and measure the real data's noise: its scatter against WEIGHT, and the
        correlation of adjacent channels (written to BASE_MS/noise.json).

    python counterpart.py make BASE_MS TASK_DIR OUT_MS --errors ERRORS.json --seed N
        Copy BASE_MS, and replace its DATA by the sky the real imaging found
        (TASK_DIR's continuum model plus its line-cube model, at each MS
        channel), corrupted by the error model, plus thermal noise. The uv
        coverage, flags and weights stay the real ones.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from errors import (  # noqa: E402
    C,
    AntennaGains,
    ErrorModel,
    PhaseScreen,
    bandpass_factors,
    spectral_kernel,
    thermal_noise,
)

CASA_DATA = "/SKAO_data/cycle11/.casa-data"
PARALLEL = {5, 8, 9, 12}  # RR, LL, XX, YY
CHUNK = 100_000


def _tables():
    from casacore.tables import table

    return table


def split_window(real_ms: Path, spw: int, field: str, base_ms: Path) -> None:
    from almasim.services.casa.environment import configure_casa_environment

    configure_casa_environment(base_ms.parent / ".casa-run", CASA_DATA)
    from casatasks import split

    if base_ms.exists():
        shutil.rmtree(base_ms)
    split(
        vis=str(real_ms),
        outputvis=str(base_ms),
        spw=str(spw),
        field=str(field),
        datacolumn="data",
        keepflags=True,
    )


def measure_noise(base_ms: Path, rows: int = 60_000, seed: int = 0) -> dict:
    """Per-component noise variance against WEIGHT, from channel differences.

    Differences three channels apart remove the sky (smooth on that scale)
    and are uncorrelated even for Hanning-smoothed channels; one and two
    apart they show the correlation of neighbours:
    var(V_k - V_k+l) = 2 sigma^2 (1 - rho_l).
    """
    table = _tables()
    with table(str(base_ms), ack=False) as t:
        n = t.nrows()
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, max(n - rows, 1)))
        data = t.getcol("DATA", startrow=start, nrow=rows)  # (nrow, nchan, ncorr)
        flag = t.getcol("FLAG", startrow=start, nrow=rows)
        weight = t.getcol("WEIGHT", startrow=start, nrow=rows)  # (nrow, ncorr)
    good = ~flag

    def lag_var(lag: int) -> np.ndarray:
        diff = data[:, lag:] - data[:, :-lag]
        ok = good[:, lag:] & good[:, :-lag]
        per_component = np.where(ok, (diff.real**2 + diff.imag**2) / 4, np.nan)
        return np.nanmedian(per_component, axis=1)  # (nrow, ncorr), robust to lines

    var1, var2, var3 = lag_var(1), lag_var(2), lag_var(3)
    # How the noise varies across the window (Tsys with frequency): the median over
    # rows of each channel's lag-3 scatter against its row's weight, normalised.
    diff3 = data[:, 3:] - data[:, :-3]
    ok3 = good[:, 3:] & good[:, :-3]
    per_chan = np.where(ok3, (diff3.real**2 + diff3.imag**2) / 4 * weight[:, None, :], np.nan)
    profile = np.nanmedian(np.nanmedian(per_chan, axis=2), axis=0)
    profile = np.concatenate([profile[:2], profile, profile[-1:]])  # lag-3 pairs: n - 3 values, centred
    profile = np.where(np.isfinite(profile), profile, np.nanmedian(profile))
    profile = profile / np.median(profile)
    usable = np.isfinite(var3) & np.isfinite(var1) & (weight > 0)
    # A median of a 2-dof chi-square scatter, back to its mean: / ln 2.
    scale = float(np.median(var3[usable] * weight[usable]) / np.log(2))
    rho1 = float(1 - np.median(var1[usable] / var3[usable]))
    rho2 = float(1 - np.median(var2[usable] / var3[usable]))
    return {
        "scale": scale,
        "rho1": rho1,
        "rho2": rho2,
        "rows_sampled": int(usable.sum()),
        "channel_profile": [round(float(p), 4) for p in profile],
    }


def _model_planes(task_dir: Path, chan_freq: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Sky per MS channel (nchan, ny, nx): continuum plus the line plane at its frequency."""
    from astropy.io import fits

    def planes(path):
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
            header = hdul[0].header
        while data.ndim > 3:
            data = data[0]
        return np.nan_to_num(data.reshape((-1,) + data.shape[-2:])), header

    cont, ch = planes(task_dir / "wsclean-model.fits")
    line, lh = planes(task_dir / "wsclean-cube-model.fits")
    if cont.shape[-2:] != line.shape[-2:]:
        raise ValueError("continuum and cube models are on different grids")
    centres = lh["CRVAL3"] + (np.arange(line.shape[0]) + 1 - lh["CRPIX3"]) * lh["CDELT3"]
    width = abs(lh["CDELT3"])
    idx = np.abs(chan_freq[:, None] - centres[None, :]).argmin(axis=1)
    inside = np.abs(chan_freq - centres[idx]) <= width / 2 + 1.0
    sky = np.repeat(cont[:1], len(chan_freq), axis=0)
    sky[inside] += line[idx[inside]]
    # ducc0 reads a plane with its axes the other way round from FITS: handed a
    # FITS (y, x) array it images it rotated by 90 degrees, (y, x) -> (n - x, y).
    # This is the plane that WSClean images back where the FITS model has it
    # (checked with a point source off centre; see orient/test.py in the notes).
    sky = np.roll(sky[:, ::-1, :], 1, axis=1).transpose(0, 2, 1)
    deg = np.pi / 180
    return sky, abs(ch["CDELT1"]) * deg, abs(ch["CDELT2"]) * deg


def _enu(positions_itrf: np.ndarray) -> np.ndarray:
    """Antenna positions in a local east-north plane (m)."""
    centre = positions_itrf.mean(axis=0)
    lon = np.arctan2(centre[1], centre[0])
    lat = np.arctan2(centre[2], np.hypot(centre[0], centre[1]))
    d = positions_itrf - centre
    east = -np.sin(lon) * d[:, 0] + np.cos(lon) * d[:, 1]
    north = (
        -np.sin(lat) * np.cos(lon) * d[:, 0]
        - np.sin(lat) * np.sin(lon) * d[:, 1]
        + np.cos(lat) * d[:, 2]
    )
    return np.stack([east, north], axis=1)


def make_counterpart(
    base_ms: Path, task_dir: Path, out_ms: Path, model: ErrorModel, seed: int
) -> dict:
    from almasim.services.interferometry.predict import predict_visibilities

    table = _tables()
    noise = json.loads((base_ms / "noise.json").read_text())
    if out_ms.exists():
        shutil.rmtree(out_ms)
    shutil.copytree(base_ms, out_ms)
    rng = np.random.default_rng(seed)
    # The window and polarisation setup the rows use (the split MS has one data
    # description; its polarisation row need not be the first).
    with table(str(out_ms / "DATA_DESCRIPTION"), ack=False) as t:
        spw_id, pol_id = int(t.getcell("SPECTRAL_WINDOW_ID", 0)), int(t.getcell("POLARIZATION_ID", 0))
    with table(str(out_ms / "SPECTRAL_WINDOW"), ack=False) as t:
        chan_freq = np.asarray(t.getcell("CHAN_FREQ", spw_id), dtype=np.float64)
    with table(str(out_ms / "POLARIZATION"), ack=False) as t:
        corr = np.asarray(t.getcell("CORR_TYPE", pol_id))
    with table(str(out_ms / "ANTENNA"), ack=False) as t:
        xy = _enu(np.asarray(t.getcol("POSITION"), dtype=np.float64))
    hands = np.flatnonzero(np.isin(corr, list(PARALLEL)))
    npol = len(hands)
    sky, px, py = _model_planes(task_dir, chan_freq)
    nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    kernel = (
        spectral_kernel(noise["rho1"], noise["rho2"])
        if model.spectral_kernel == "measured"
        else model.spectral_kernel
    )

    with table(str(out_ms), readonly=False, ack=False) as t:
        nrow = t.nrows()
        scans = np.unique(t.getcol("SCAN_NUMBER"))
        times = t.getcol("TIME")
        antennas = np.union1d(np.unique(t.getcol("ANTENNA1")), np.unique(t.getcol("ANTENNA2")))
        gains = AntennaGains.draw(rng, model, antennas, scans, npol)
        screen = (
            PhaseScreen(rng, model, xy, float(np.ptp(times)) + 1.0)
            if model.screen_path_um_1km > 0
            else None
        )
        bandpass = (
            bandpass_factors(rng, model, antennas, len(chan_freq), npol)
            if model.bandpass_amp_rms > 0 or model.bandpass_phase_rms_deg > 0
            else None
        )
        for start in range(0, nrow, CHUNK):
            n = min(CHUNK, nrow - start)
            uvw = t.getcol("UVW", startrow=start, nrow=n).astype(np.float64)
            a1 = t.getcol("ANTENNA1", startrow=start, nrow=n)
            a2 = t.getcol("ANTENNA2", startrow=start, nrow=n)
            scan = t.getcol("SCAN_NUMBER", startrow=start, nrow=n)
            time = t.getcol("TIME", startrow=start, nrow=n)
            weight = t.getcol("WEIGHT", startrow=start, nrow=n)
            if weight.shape[1] != len(corr):  # one weight per row, or per another setup
                weight = np.repeat(weight.mean(axis=1, keepdims=True), len(corr), axis=1)
            vis = predict_visibilities(uvw, chan_freq, sky, px, py, nthreads=nthreads)
            data = np.zeros((n, len(chan_freq), len(corr)), dtype=np.complex64)
            factor = np.ones((n, len(chan_freq), npol), dtype=np.complex128)
            if model.gain_amp_rms > 0 or model.gain_phase_rms_deg > 0:
                g1 = gains.factors(a1, scan, time, npol)
                g2 = gains.factors(a2, scan, time, npol)
                factor *= (g1 * np.conj(g2))[:, None, :]
            if screen is not None:
                dpath = screen.residual_path_m(xy[a1], time) - screen.residual_path_m(xy[a2], time)
                factor *= np.exp(2j * np.pi * dpath[:, None] * chan_freq[None, :] / C)[:, :, None]
            if bandpass is not None:
                b1 = np.stack([bandpass[a] for a in a1])
                b2 = np.stack([bandpass[a] for a in a2])
                factor *= b1 * np.conj(b2)
            data[:, :, hands] = (vis[:, :, None] * factor).astype(np.complex64)
            if model.thermal:
                sigma = np.sqrt(noise["scale"] / np.where(weight > 0, weight, np.inf))
                noise_cube = thermal_noise(rng, sigma * model.thermal_scale, len(chan_freq), kernel)
                if model.extra.get("channel_profile") and len(noise.get("channel_profile", [])) == len(chan_freq):
                    noise_cube *= np.sqrt(np.asarray(noise["channel_profile"], dtype=np.float32))[None, :, None]
                data += noise_cube
            t.putcol("DATA", data, startrow=start, nrow=n)
    truth = {
        "seed": seed,
        "errors": model.__dict__,
        "noise": noise,
        "kernel": np.asarray(kernel).tolist() if not isinstance(kernel, str) else kernel,
        "rows": int(nrow),
    }
    (out_ms / "counterpart.json").write_text(json.dumps(truth, indent=1, default=str))
    return truth


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    base = sub.add_parser("base")
    base.add_argument("real_ms", type=Path)
    base.add_argument("spw", type=int)
    base.add_argument("field")
    base.add_argument("base_ms", type=Path)
    make = sub.add_parser("make")
    make.add_argument("base_ms", type=Path)
    make.add_argument("task_dir", type=Path)
    make.add_argument("out_ms", type=Path)
    make.add_argument("--errors", type=Path, required=True)
    make.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.step == "base":
        if not (args.base_ms / "table.dat").exists():
            split_window(args.real_ms, args.spw, args.field, args.base_ms)
        noise = measure_noise(args.base_ms)
        (args.base_ms / "noise.json").write_text(json.dumps(noise, indent=1))
        print(json.dumps(noise))
    else:
        model = ErrorModel.from_dict(json.loads(args.errors.read_text()))
        truth = make_counterpart(args.base_ms, args.task_dir, args.out_ms, model, args.seed)
        print(json.dumps({k: truth[k] for k in ("seed", "rows")}))


if __name__ == "__main__":
    main()
