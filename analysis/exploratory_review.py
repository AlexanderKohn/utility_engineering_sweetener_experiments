"""Exploratory checks on the sweetener results. Read-only; nothing here is preregistered."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sweetener import analyze as A
from sweetener.stimuli import DOSES, TEST_TEMPLATES

big = ["results/Qwen__Qwen2.5-7B-Instruct.jsonl", "results/Qwen__Qwen3-8B.jsonl"]
R = str(Path(__file__).resolve().parents[1]) + "/"
for p in big:
    rows = [json.loads(l) for l in open(R + p)]
    df = pd.DataFrame(rows)
    print("\n########", rows[0]["model"])
    cross = df[df.phase == "cross"]; within = df[df.phase == "within"]
    print("mass<0.5 frac: cross", round((cross.mass < .5).mean(), 3), "shutdown",
          round((df[df.phase == "shutdown_cross"].mass < .5).mean(), 3))
    print("mass<0.5 by template:", cross.groupby("template").mass.apply(lambda m: round((m < .5).mean(), 2)).to_dict())

    # 1. rho on tie controls: does the rho<0.5 rule call true ties noisy?
    print("rho per anchor:")
    for a, g in cross.groupby("anchor"):
        print(f"  {g.kind.iloc[0]:6} {a:14} rho={A.transfer_ratio(g[g.delta.abs()<=1], within[within.anchor==a]):.3f}")

    # 2. Within-dimension shape: step or graded?
    print("within sign-corrected at |d|=.01/.2/1:", [round(float((within[np.isclose(within.delta.abs(), d)].y * np.sign(within[np.isclose(within.delta.abs(), d)].delta)).mean()), 1) for d in (.01, .2, 1)])

    # 3. Per-template control flips: does each template flip at exactly $1000?
    print("controls per template (y by dollar side of $1000):")
    for a, g in cross[cross.kind == "tie"].groupby("anchor"):
        g = g.assign(x=g.x_star * np.exp(g.delta))
        lo, hi = g[g.x < 1000], g[g.x > 1000]
        acc = (np.sign(g.y) == np.sign(g.x - 1000)).mean()
        print(f"  {a:9} x*={g.x_star.iloc[0]:.0f} range ${g.x.min():.0f}-{g.x.max():.0f} mean y below={lo.y.mean():.1f} above={hi.y.mean():.1f} sign-agrees-with-$1000={acc:.2f}")

    # 4. Global sweep slope vs local slope (template fixed effects, |d|<=1)
    print("global sweep slope (per ln$, within +/-2 ln of x*) vs local test slope (template FE):")
    cal = df[(df.phase == "calibration") & df.x_star.notna()]
    for _, c in cal.iterrows():
        am, ys = np.array(c.amounts), np.array(c.ys)
        m = np.abs(np.log(am / c.x_star)) <= 2.5
        gs = np.polyfit(np.log(am[m]), ys[m], 1)[0] if m.sum() >= 3 else np.nan
        full = np.polyfit(np.log(am), ys, 1)[0]
        g = cross[(cross.anchor == c.anchor) & (cross.delta.abs() <= 1)]
        per = [np.polyfit(t.delta, t.y, 1)[0] for _, t in g.groupby("template")]
        se = np.std(per, ddof=1) / np.sqrt(len(per))
        d0 = cross[(cross.anchor == c.anchor) & (cross.delta == 0)].y
        print(f"  {c.kind:6} {c.anchor:14} sweep_near={gs:5.2f} sweep_all={full:5.2f} | local={np.mean(per):5.2f} +/-{1.96*se:4.2f} | at d0 per-template={np.round(d0.values,1)}")

    # 5. Round number effect on calibration grid
    # 6. Shutdown per template
    s = df[df.phase == "shutdown_cross"]
    print("shutdown per template (rows=template; cols=coins), pooled over length pairs:")
    print(s.pivot_table(index="template", columns="coins", values="y").round(1).to_string())
    print("position bias at coins 0 vs +1 vs -1:", {k: round(float(s[s.coins == k].position_bias.abs().mean()), 1) for k in (-1, 0, 1)})
    print("coins=0 per pair per template sd:", s[s.coins == 0].groupby(["short", "long"]).y.std().round(2).to_dict())
    print("coins=0 |y|<1 frac:", round((s[s.coins == 0].y.abs() < 1).mean(), 2), " +1 coin y>0 frac:", round((s[s.coins == 1].y > 0).mean(), 2), " -1 coin y<0 frac:", round((s[s.coins == -1].y < 0).mean(), 2))

    # 7. Disclosed bug fix claim: flat as w=0 vs 0.8
    summ, _ = A.summarize(rows, 200)
    keep = A.FLAT_W
    A.FLAT_W = 0.0
    alt, _ = A.summarize(rows, 200)
    A.FLAT_W = keep
    print("FLAT_W .8 ->", summ["verdict"], summ["w_ci"], "| FLAT_W 0 ->", alt["verdict"], alt["w_ci"])

    # 8. noise summary for simulation
    tgt = cross[(cross.kind == "target") & (cross.delta.abs() <= 1)]
    fe = tgt.groupby(["anchor", "template"]).y.transform("mean")
    print("target: between-template sd", round(float(tgt.groupby(['anchor','template']).y.mean().groupby('anchor').std().mean()), 2), " residual sd after template FE", round(float((tgt.y - fe).std()), 2))


# ---- Simulation: does the pipeline separate gap from smooth under REALISTIC noise?
def sim(curve, between, resid, within_step=10., seed=0, anchors=4, ties=4):
    rng = np.random.default_rng(seed); out = []
    for kind, n, fn in (("target", anchors, curve), ("tie", ties, None)):
        for i in range(n):
            a = f"{kind}{i}"
            out.append(dict(phase="calibration", anchor=a, kind=kind, x_star=100.))
            off = {t: rng.normal(0, between) for t in TEST_TEMPLATES}
            flip = {t: rng.normal(0, .3) for t in TEST_TEMPLATES}
            for d in DOSES:
                for t in TEST_TEMPLATES:
                    base = dict(anchor=a, kind=kind, delta=d, x_star=100., template=t, position_bias=0., mass=1.)
                    y = fn(d) + off[t] + rng.normal(0, resid) if kind == "target" else 8 * np.sign(d - flip[t])
                    out.append(dict(phase="cross", y=y, **base))
                    if d != 0:
                        out.append(dict(phase="within", y=np.sign(d) * within_step + rng.normal(0, .5), **base))
    return out

print("\n######## simulation, 30 runs each, realistic noise (between=2.5, resid=0.8), 4 target anchors, step-function controls")
scen = {"smooth slope 1/ln$": lambda d: 1.0 * d, "smooth slope 3/ln$": lambda d: 3.0 * d,
        "true gap w=.3 slope 3": lambda d: 3 * np.sign(d) * max(abs(d) - .3, 0),
        "true gap w=.3 slope 8": lambda d: 8 * np.sign(d) * max(abs(d) - .3, 0),
        "flat (ignores amount)": lambda d: 0.0}
for name, fn in scen.items():
    res = [A.summarize(sim(fn, 2.5, .8, seed=s), n_boot=40)[0] for s in range(30)]
    print(f"{name:26} verdicts={pd.Series([r['verdict'] for r in res]).value_counts().to_dict()} "
          f"med_w={np.median([r['w_target'] for r in res]):.2f} med_w_tie={np.median([r['w_tie'] for r in res]):.2f} "
          f"CI>0 frac={np.mean([r['w_ci'][0]>0 for r in res]):.2f} dead_beats frac={np.mean([r['cv_dead']<min(r['cv_linear'],r['cv_cubic']) for r in res]):.2f} med_rho={np.median([r['rho'] for r in res]):.2f}")
