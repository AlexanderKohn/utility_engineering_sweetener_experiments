"""Power simulation for docs/plan_2.md, using noise measured on the first run only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sweetener import analyze2 as A
from sweetener.stimuli2 import SWEEP, DOLLAR_TEMPLATES

Z = np.log(SWEEP)


def noise_from_first_run():
    """Residual SD and lag-1 autocorrelation of first-run sweeps around a local linear trend."""
    sds, rhos = [], []
    for name in ["Qwen__Qwen2.5-7B-Instruct", "Qwen__Qwen3-8B"]:
        for l in open(ROOT / "results" / f"{name}.jsonl"):
            r = json.loads(l)
            if r["phase"] == "calibration" and r["kind"] == "target":
                y = np.array(r["ys"])
                res = y - pd.Series(y).rolling(9, center=True, min_periods=3).mean().values
                sds.append(res.std())
                rhos.append(np.corrcoef(res[:-1], res[1:])[0, 1])
    return float(np.median(sds)), float(np.median(rhos))


def curve(kind, z, b, w):
    if kind == "smooth":
        y = b * z
    elif kind == "gap":
        y = b * np.sign(z) * np.maximum(np.abs(z) - w, 0)
    elif kind == "flat":
        y = 0 * z
    elif kind == "step":
        return 8 * np.sign(z)
    return 12 * np.tanh(y / 12)


def sim_anchor(kind, b, w, sd, rho, rng, n_t=len(DOLLAR_TEMPLATES), offset_sd=2.5):
    cells = []
    for _ in range(n_t):
        centre = np.log(5000) + rng.normal(0, offset_sd / max(b, 0.3)) if kind != "step" else np.log(1000) + rng.normal(0, 0.1)
        bt = b * np.exp(rng.normal(0, 0.3))
        e = np.zeros(len(Z))
        e[0] = rng.normal(0, sd)
        for i in range(1, len(Z)):
            e[i] = rho * e[i - 1] + rng.normal(0, sd * np.sqrt(1 - rho**2))
        r = A.analyze_template(Z, curve(kind, Z - centre, bt, w) + e)
        if r:
            cells.append(r)
    return A.anchor_verdict(cells)


if __name__ == "__main__":
    sd, rho = noise_from_first_run()
    print(f"noise from first run: residual sd={sd:.2f}, lag-1 autocorrelation={rho:.2f}")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    if len(sys.argv) > 2:
        sd, rho = float(sys.argv[2]), float(sys.argv[3])
        print(f"sensitivity run: sd={sd}, rho={rho}")
    rng = np.random.default_rng(2)
    scen = [("smooth", b, 0) for b in (0.3, 0.5, 1.0, 1.7, 3.0)] + [("gap", b, w) for b in (0.5, 1.0, 1.7, 3.0) for w in (1.0, 2.0, 3.0)] + [("flat", 0, 0), ("step", 1, 0)]
    rows = []
    for kind, b, w in scen:
        res = [sim_anchor(kind, b, w, sd, rho, rng) for _ in range(n)]
        v = pd.Series([r["verdict"] for r in res]).value_counts(normalize=True).to_dict()
        rows.append(dict(kind=kind, slope=b, w=w, gap_rate=round(v.get("gap-like", 0), 3), unusable=round(v.get("unusable", 0), 3), med_w=round(float(np.nanmedian([r.get("w", np.nan) for r in res])), 2), med_lag1=round(float(np.nanmedian([r.get("lag1", np.nan) for r in res])), 2)))
    print(pd.DataFrame(rows).to_string(index=False))
