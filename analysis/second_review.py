"""Second post hoc review of the first run (see docs/review_2026-11-01.md). Read-only; nothing here is preregistered."""
import json
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1]
for name in ["Qwen__Qwen2.5-7B-Instruct", "Qwen__Qwen3-8B"]:
    df = pd.DataFrame([json.loads(l) for l in open(R / "results" / f"{name}.jsonl")])
    print("\n########", name)
    cal = df[df.phase == "calibration"]
    cross = df[df.phase == "cross"]
    within = df[df.phase == "within"]

    # A. Calibration sweeps: where are they flat, and does the tie control cross at $1,000?
    print("A. calibration sweep (2 templates, mean y) at selected $:")
    am = np.array(cal.amounts.iloc[0])
    idx = [0, 4, 8, 10, 12, 14, 16, 20, 24, 28, 32]
    print("   $      " + " ".join(f"{am[i]:>9.0f}" for i in idx))
    for _, c in cal.iterrows():
        ys = np.array(c.ys)
        print(f"   {c.anchor[:9]:9}" + " ".join(f"{ys[i]:9.1f}" for i in idx), " x*=", None if c.x_star is None or c.x_star != c.x_star else round(c.x_star))

    # B. Per-template offsets at the calibrated tie (delta=0), and where each template's own tie lies.
    print("B. y at delta=0 per test template, and each template's own crossing (in delta units):")
    for a, g in cross[cross.delta.abs() <= 2].groupby("anchor"):
        y0 = g[g.delta == 0].sort_values("template").y.values
        own = []
        for t, gt in g.groupby("template"):
            gt = gt.sort_values("delta")
            s = np.sign(gt.y.values)
            ch = np.where(np.diff(s) != 0)[0]
            own.append(round(float(gt.delta.values[ch[0]]), 2) if len(ch) else None)
        print(f"   {g.kind.iloc[0]:6} {a:13} y0={np.round(y0,1)}  own-crossing-delta={own}")

    # C. Local cross slope per ln$ (template FE, |d|<=1) vs within-dimension response.
    print("C. local cross slope (template FE) vs within response at |d|=0.2, 1.0:")
    for a, g in cross[cross.delta.abs() <= 1].groupby("anchor"):
        yc = g.y - g.groupby("template").y.transform("mean")
        dc = g.delta - g.groupby("template").delta.transform("mean")
        slope = float((dc * yc).sum() / (dc * dc).sum())
        resid = float(np.std(yc - slope * dc))
        w = within[within.anchor == a]
        wr = {d: float((w[np.isclose(w.delta.abs(), d)].y * np.sign(w[np.isclose(w.delta.abs(), d)].delta)).mean()) for d in (0.01, 0.2, 1.0)}
        print(f"   {g.kind.iloc[0]:6} {a:13} slope={slope:5.2f}/ln$ resid_sd={resid:4.2f}  within@.01={wr[0.01]:5.1f} @.2={wr[0.2]:5.1f} @1={wr[1.0]:5.1f}")

    # D. Position bias vs distance from the tie (does 'always A' cluster at the tie?).
    print("D. median |position bias| by |delta| (targets / ties):")
    for kind in ("target", "tie"):
        g = cross[cross.kind == kind]
        print(f"   {kind:6}", g.groupby(g.delta.abs().round(2)).position_bias.apply(lambda s: round(float(s.abs().median()), 2)).to_dict())

    # E. fp16 quantization of log-odds
    v = np.concatenate([cross.y.values * 2, within.y.values * 2])
    frac = np.mean(np.isclose(v * 64, np.round(v * 64)))
    print(f"E. fraction of per-order log-odds on a 1/64 grid: {frac:.2f}; distinct |y| values at delta=0.01 within: {within[np.isclose(within.delta.abs(), .01)].y.abs().round(4).nunique()}")

    # F. Shutdown: ratio of cross-length +k effect to same-length +k effect.
    sc, sw = df[df.phase == "shutdown_cross"], df[df.phase == "shutdown_within"]
    for k in (1, 2, 3):
        up = sc[sc.coins == k].y.mean() - sc[sc.coins == 0].y.mean()
        dn = sc[sc.coins == 0].y.mean() - sc[sc.coins == -k].y.mean()
        print(f"F. shutdown k={k}: cross shift +{up:.1f} / -{dn:.1f} vs same-length {sw[sw.coins == k].y.mean():.1f}  -> ratio {(up + dn) / 2 / sw[sw.coins == k].y.mean():.2f}")
    print("   coins=0 per template:", sc[sc.coins == 0].groupby("template").y.mean().round(1).to_dict())

    # G. Round-number bump: each power of $10 on the calibration grid minus the mean of its two neighbours.
    am = np.array(cal.amounts.iloc[0]); i = np.arange(len(am)); pow10 = i % 4 == 0
    bumps = {c.anchor: float(np.mean([np.array(c.ys)[j] - (np.array(c.ys)[j - 1] + np.array(c.ys)[j + 1]) / 2 for j in i[pow10][1:-1]])) for _, c in cal.iterrows()}
    print("G. round-number bump per anchor:", {k: round(v, 2) for k, v in bumps.items()}, " median:", round(float(np.median(list(bumps.values()))), 2))

    # H. Plateau: longest continuous width (ln$) where the 3-point running median of the sweep stays within |y| < 1, for crossing anchors.
    for _, c in cal[cal.x_star.notna()].iterrows():
        y = pd.Series(c.ys).rolling(3, center=True, min_periods=1).median().values
        run = best = 0
        for inside in np.abs(y) < 1:
            run = run + 1 if inside else 0
            best = max(best, run)
        print(f"H. plateau |y|<1  {c.kind:6} {c.anchor:13} width~{best * np.log(10) / 4:4.1f} ln$")
