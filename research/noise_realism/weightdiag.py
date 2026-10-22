"""How the real noise scales with WEIGHT, and what the image sees.

    python weightdiag.py I OUT.json

For window I's base MS, rows sampled evenly over the whole observation:
lag-3 channel differences give each row's noise variance (per component);
times WEIGHT that is the scale the counterpart uses (1 if WEIGHT were exact).
Reports that scale as the counterpart measures it (median, over rows, of
row medians / ln 2), as a plain mean, and weighted by WEIGHT (what an image,
which weights each visibility by WEIGHT, sees), and by WEIGHT quintile.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterpart import _tables  # noqa: E402
from window import name, subset  # noqa: E402

BASE = Path("/SKAO_data/cycle11/MADIV-28/noise/base")

i, out = int(sys.argv[1]), Path(sys.argv[2])
row = subset().iloc[i]
ms = BASE / f"{name(row)}.ms"
table = _tables()
with table(str(ms), ack=False) as t:
    step = max(t.nrows() // 200_000, 1)
    data = t.getcol("DATA", rowincr=step)
    flag = t.getcol("FLAG", rowincr=step)
    weight = t.getcol("WEIGHT", rowincr=step)
    a1, a2 = t.getcol("ANTENNA1", rowincr=step), t.getcol("ANTENNA2", rowincr=step)
if weight.shape[1] != data.shape[2]:
    weight = np.repeat(weight.mean(axis=1, keepdims=True), data.shape[2], axis=1)
keep = a1 != a2
data, flag, weight = data[keep], flag[keep], weight[keep]
diff = data[:, 3:] - data[:, :-3]
ok = ~flag[:, 3:] & ~flag[:, :-3]
comp = np.where(ok, (diff.real**2 + diff.imag**2) / 4, np.nan)  # per component, mean sigma^2
row_mean = np.nanmean(comp, axis=1)  # (nrow, ncorr)
row_med = np.nanmedian(comp, axis=1)
w = weight
good = np.isfinite(row_mean) & (w > 0)
s_mean = row_mean[good] * w[good]
result = {
    "window": name(row),
    "source": row.source,
    "ghz": float(row.ghz),
    "rows": int(good.sum()),
    "scale_counterpart": float(np.median(row_med[good] * w[good]) / np.log(2)),
    "scale_mean": float(np.mean(s_mean)),
    "scale_weighted": float(np.sum(row_mean[good] * w[good] ** 2) / np.sum(w[good])),
    "noise_json": json.loads((ms / "noise.json").read_text())["scale"],
}
q = np.quantile(w[good], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
result["by_weight_quintile"] = [
    float(np.mean(s_mean[(w[good] >= lo) & (w[good] <= hi)])) for lo, hi in zip(q[:-1], q[1:])
]
result["weight_quintile_edges"] = [float(x) for x in q]
out.write_text(json.dumps(result, indent=1))
print(json.dumps(result))
