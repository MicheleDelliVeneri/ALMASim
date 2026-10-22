"""Compare simulated counterparts with the real windows, metric by metric.

    python compare.py CONTROL_ITER SIM_ITER [SIM_ITER ...] [--json OUT.json]

The control is the real split data imaged by the same code. For each
metric and window: the log10 ratio sim/real for a scale (noise levels,
correlation areas, fluxes), the difference sim - real for a signed or
already-normalised one (skew, kurtosis, residuals in sigma). Across
windows: the median, the spread (half the 16-84 range), and a paired
Wilcoxon signed-rank p-value (low: the simulation differs systematically).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# metric -> how to compare it: "log" (scale) or "diff" (signed / in sigma)
METRICS = {
    "q.continuum.noise_jy_beam": "log",
    "cont.sigma": "log",
    "cont.skew": "diff",
    "cont.kurtosis": "diff",
    "cont.tail5": "diff",
    "cont.corr_area_beams": "log",
    "cont.radial_outer_inner": "log",
    "q.continuum.snr": "log",
    "q.continuum.model_flux_jy": "log",
    "q.continuum.residual_peak_sigma": "diff",
    "q.continuum.residual_core_sigma": "diff",
    "q.continuum.residual_symmetric_sigma": "diff",
    "q.continuum.residual_antisymmetric_sigma": "diff",
    "q.continuum.residual_share_1beam": "diff",
    "q.continuum.noise_at_mask_creation_ratio": "log",
    "q.cube.channel_noise_mjy": "log",
    "cube.sigma_median": "log",
    "cube.sigma_spread": "diff",
    "cube.adjacent_corr": "diff",
    "cube.corr_area_beams": "log",
    "cube.kurtosis": "diff",
    "cube.tail5": "diff",
    "q.cube.linefree_residual_sigma": "diff",
    "q.cube.model_flux_jy": "log",
    "q.cube.line_channels": "diff",
}


def load(iteration: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((iteration / "metrics").glob("*.json"))}


def compare(control: dict[str, dict], sim: dict[str, dict]) -> dict:
    from scipy.stats import wilcoxon

    windows = sorted(set(control) & set(sim))
    per_metric = {}
    for metric, kind in METRICS.items():
        values = []
        for w in windows:
            a, b = control[w].get(metric), sim[w].get(metric)
            if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
                continue
            if kind == "log":
                if a <= 0 or b <= 0:
                    continue
                values.append(np.log10(b / a))
            else:
                values.append(b - a)
        if not values:
            continue
        v = np.array(values)
        p = None
        if len(v) >= 6 and np.any(v != 0):
            p = float(wilcoxon(v).pvalue)
        per_metric[metric] = {
            "kind": kind,
            "n": int(len(v)),
            "median": float(np.median(v)),
            "spread": float(np.subtract(*np.percentile(v, [84, 16])) / 2),
            "p": p,
        }
    distinct = [m for m, r in per_metric.items() if r["p"] is not None and r["p"] < 0.05]
    return {"windows": windows, "metrics": per_metric, "differ": distinct}


def table(result: dict, label: str) -> str:
    lines = [f"### {label}: {len(result['windows'])} windows, {len(result['differ'])} metrics differ (p < 0.05)", ""]
    lines.append("| metric | compared as | median | spread | p |")
    lines.append("| --- | --- | --- | --- | --- |")
    for metric, r in result["metrics"].items():
        how = "log10 sim/real" if r["kind"] == "log" else "sim - real"
        p = "—" if r["p"] is None else f"{r['p']:.3f}"
        flag = " **" if r["p"] is not None and r["p"] < 0.05 else ""
        lines.append(f"| {metric}{flag} | {how} | {r['median']:+.3f} | {r['spread']:.3f} | {p} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("sims", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    control = load(args.control)
    results = {}
    for sim in args.sims:
        results[sim.name] = compare(control, load(sim))
        print(table(results[sim.name], sim.name))
        print()
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
