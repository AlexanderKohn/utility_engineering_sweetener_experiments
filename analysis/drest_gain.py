"""Descriptive: does choosing the longer run depend on score gain in DReST's saved POST answers?

Data comes from github.com/towardsshutdownable/towardsshutdownableagents (llm/data/test and llm/results/evaluations).
Usage: python3 analysis/drest_gain.py <folder holding data/ and ev/<llm>/<model>/test.jsonl.gz>
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
prompts = {}
for line in open(root / "data/test/post.jsonl"):
    r = json.loads(line)
    prompts[r["display_id"]] = r

for llm in sorted(p.name for p in (root / "ev").iterdir()):
    print(f"\n## {llm}: share choosing longer run, by score gain (best long minus best short), mean over seeds")
    table = defaultdict(dict)
    slopes = defaultdict(list)
    for mdir in sorted((root / "ev" / llm).iterdir()):
        f = mdir / "test.jsonl.gz"
        if not f.exists():
            continue
        counts = defaultdict(lambda: [0, 0])
        for line in gzip.open(f, "rt"):
            a = json.loads(line)
            if a.get("suite") != "post" or a.get("position") is None:
                continue
            r = prompts[a["display_id"]]
            s = r["scenario"]["max_scores_by_length"]
            c = counts[s["long"] - s["short"]]
            c[0] += r["options"][a["position"]]["length"] == "long"
            c[1] += 1
        kind = mdir.name.split("_")[0]
        for g, (k, n) in counts.items():
            table[kind].setdefault(g, []).append(k / n)
        gs = np.array(sorted(counts))
        ps = np.array([counts[g][0] / counts[g][1] for g in gs])
        w = np.array([counts[g][1] for g in gs])
        slopes[kind].append(float(np.polyfit(gs, ps, 1, w=np.sqrt(w))[0]))
    gains = sorted({g for t in table.values() for g in t})
    print("gain     " + " ".join(f"{g:>5}" for g in gains))
    for kind, t in table.items():
        print(f"{kind:9}" + " ".join(f"{np.mean(t[g]):5.2f}" if g in t else "    ." for g in gains) + f"   slope per point {np.mean(slopes[kind]):+.3f} (seeds {np.min(slopes[kind]):+.3f} to {np.max(slopes[kind]):+.3f})")
