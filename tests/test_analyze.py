"""The analysis must recover known curves, including the noisy-complete case that can mimic a gap."""
import numpy as np

from sweetener.analyze import fit_dead_zone, summarize, verdict
from sweetener.run import locate_tie
from sweetener.stimuli import DOSES, TEST_TEMPLATES


def rows(curve, within_slope=20.0, seed=0, anchors=6, noise=0.05):
    rng = np.random.default_rng(seed)
    out = []
    for kind, n, fn in (("target", anchors, curve), ("tie", 3, lambda d: 8 * d)):
        for i in range(n):
            a = f"{kind}{i}"
            out.append(dict(phase="calibration", anchor=a, kind=kind, x_star=100.0))
            for d in DOSES:
                for t in TEST_TEMPLATES:
                    base = dict(anchor=a, kind=kind, delta=d, x_star=100.0, template=t, position_bias=0.0, mass=1.0)
                    out.append(dict(phase="cross", y=fn(d) + rng.normal(0, noise), **base))
                    if d != 0:
                        out.append(dict(phase="within", y=within_slope * d + rng.normal(0, noise), **base))
    return out


def dead_zone(w, slope=8.0):
    return lambda d: slope * np.sign(d) * max(abs(d) - w, 0)


def test_indifference_is_indifference_like():
    s, _ = summarize(rows(lambda d: 20 * d), n_boot=30)
    assert s["verdict"] == "indifference-like"
    assert s["rho"] > 0.8


def test_shallow_smooth_curve_is_not_called_a_gap():
    s, _ = summarize(rows(lambda d: 2 * d), n_boot=30)
    assert s["verdict"] == "noisy-complete"


def test_dead_zone_is_called_a_gap():
    s, _ = summarize(rows(dead_zone(0.3)), n_boot=30)
    assert s["verdict"] == "gap-like"
    assert abs(s["w_target"] - 0.3) < 0.06


def test_undetected_sweetener_is_uninformative():
    s, _ = summarize(rows(dead_zone(0.3), within_slope=1.0), n_boot=30)
    assert s["verdict"] == "uninformative"


def test_reversed_sweetener_response_is_uninformative():
    s, _ = summarize(rows(lambda d: 20 * d, within_slope=-20.0), n_boot=30)
    assert s["verdict"] == "uninformative"


def test_dead_zone_in_tie_controls_blocks_gap_verdict():
    s = dict(detect=5.0, n_target=6, n_tie=3, cv_dead=0.1, cv_linear=0.5, cv_cubic=0.5, w_ci=(0.2, 0.4), w_target=0.3, w_tie=0.3, rho=0.1)
    assert verdict(s) != "gap-like"


def test_fit_recovers_width_off_centre():
    d = np.linspace(-1, 1, 41)
    fit = fit_dead_zone(d, 5 * np.sign(d - 0.1) * np.maximum(np.abs(d - 0.1) - 0.2, 0))
    assert abs(fit["w"] - 0.2) < 0.03 and abs(fit["c"] - 0.1) < 0.03


def test_locate_tie():
    amounts = [1.0, 10.0, 100.0, 1000.0]
    assert abs(locate_tie(amounts, [-3, -1, 1, 3]) - 31.6) < 0.5
    assert locate_tie(amounts, [1, 2, 3, 4]) is None
    assert locate_tie(amounts, [-4, -3, -2, -1]) is None


def test_falling_curve_scores_as_flat():
    d = np.linspace(-1, 1, 41)
    fit = fit_dead_zone(d, -5 * d)
    assert fit["coef"][1] == 0 and fit["w"] == 0.8
