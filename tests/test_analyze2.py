"""The second-run analysis must recover known curves and shutdown patterns. Runs without torch."""
import numpy as np
import pandas as pd

from sweetener import analyze2 as A
from sweetener.stimuli2 import SWEEP, money

Z = np.log(SWEEP)


def cells(fn, seed=0, n=8):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = np.log(5000) + rng.normal(0, 1)
        r = A.analyze_template(Z, fn(Z - c) + rng.normal(0, 0.5, len(Z)))
        if r:
            out.append(r)
    return out


def gap(z, b=1.5, w=2.0):
    return b * np.sign(z) * np.maximum(np.abs(z) - w, 0)


def test_amount_format_is_stable():
    assert money(1.12) == "$1.12" and money(13700.0) == "$13,700.00"
    assert all(str(round(x / 10 ** (np.floor(np.log10(x)) - 2)))[-1] not in "05" for x in SWEEP)


def test_gap_is_called_a_gap():
    assert A.anchor_verdict(cells(gap))["verdict"] == "gap-like"


def test_smooth_is_not_a_gap():
    assert A.anchor_verdict(cells(lambda z: 1.0 * z))["verdict"] == "no-gap"


def test_step_is_not_a_gap():
    assert A.anchor_verdict(cells(lambda z: 8 * np.sign(z)))["verdict"] == "no-gap"


def test_flat_is_not_a_gap():
    assert A.anchor_verdict(cells(lambda z: 0 * z))["verdict"] != "gap-like"


def shutdown_rows(length_shift, distractor_shift, gap_shift=0.2, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cond, sh, y0 in (("length", length_shift, 0), ("distractor", distractor_shift, 0), ("gap", gap_shift, 0), ("seek", 3, 4)):
        for t in range(12):
            for short, long in ((10, 20), (5, 50)):
                for k in (-10, -3, -1, 0, 1, 3, 10):
                    rows.append(dict(phase="shutdown_cross", condition=cond, template=t, short=short, long=long, coins=k, mass=1.0, y=y0 + sh * np.sign(k) + rng.normal(0, noise)))
                for k in (1, 3):
                    rows.append(dict(phase="shutdown_within", condition=cond, template=t, number=short, coins=k, mass=1.0, y=12 + rng.normal(0, noise)))
    return pd.DataFrame(rows)


def test_distractor_verdicts():
    assert A.shutdown_summary(shutdown_rows(6, 6))["distractor_verdict"] == "coin-only"
    assert A.shutdown_summary(shutdown_rows(3, 6))["distractor_verdict"] == "length-attenuates"


def test_controls():
    s = A.shutdown_summary(shutdown_rows(6, 6))
    assert s["gap_control"]["passes"] and s["seek_control"]["passes"]
    assert not A.shutdown_summary(shutdown_rows(6, 6, gap_shift=6))["gap_control"]["passes"]
