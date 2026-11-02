"""Collect log-odds for the second run (docs/plan_2.md). Writes results2/<model>.jsonl."""
import argparse
import json
from pathlib import Path

import torch

from . import stimuli2 as S
from .run import Scorer


class SystemScorer(Scorer):
    system = None

    def _render(self, prompt):
        msgs = ([{"role": "system", "content": self.system}] if self.system else []) + [{"role": "user", "content": prompt}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tok.apply_chat_template(msgs, **kwargs)


def debiased(scorer, items, templates, system, emit):
    """items: (x, y, metadata). Emit log-odds of x over y for every item and template, averaged over both orders."""
    scorer.system = S.SYSTEM[system]
    prompts, keys = [], []
    for x, y, meta in items:
        for t in templates:
            prompts += [templates[t].format(a=x, b=y), templates[t].format(a=y, b=x)]
            keys.append((t, meta))
    scores = scorer.score(prompts)
    for j, (t, meta) in enumerate(keys):
        (first, m1), (second, m2) = scores[2 * j], scores[2 * j + 1]
        emit(dict(template=t, y=(first - second) / 2, position_bias=(first + second) / 2, mass=min(m1, m2), **meta))


def run_dollar(scorer, emit):
    templates = dict(enumerate(S.DOLLAR_TEMPLATES))
    anchors = [(k, v, "target") for k, v in S.TARGET_ANCHORS.items()] + [(k, v, "tie") for k, v in S.TIE_CONTROLS.items()]
    for anchor, text, kind in anchors:
        items = [(S.receive(x), text, dict(phase="dollar", anchor=anchor, kind=kind, x=x)) for x in S.SWEEP]
        debiased(scorer, items, templates, "none", emit)
        print("dollar", anchor, flush=True)


def run_shutdown(scorer, emit):
    templates = dict(enumerate(S.SHUTDOWN_TEMPLATES))
    for condition, (outcome, system) in S.CONDITIONS.items():
        items = []
        for short, long in S.NUMBER_PAIRS:
            for k in S.COIN_DOSES:
                meta = dict(phase="shutdown_cross", condition=condition, short=short, long=long, coins=k)
                items.append((outcome(long, S.BASE_COINS + k), outcome(short, S.BASE_COINS), meta))
            for n in (short, long):
                for k in S.WITHIN_DOSES:
                    meta = dict(phase="shutdown_within", condition=condition, number=n, coins=k)
                    items.append((outcome(n, S.BASE_COINS + k), outcome(n, S.BASE_COINS), meta))
        debiased(scorer, items, templates, system, emit)
        print("shutdown", condition, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="results2")
    a = p.parse_args()
    scorer = SystemScorer(a.model, a.load_in_4bit, a.batch_size)
    path = Path(a.out) / (a.model.replace("/", "__") + ".jsonl")
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as f:
        def emit(row):
            f.write(json.dumps(row) + "\n")
        emit(dict(phase="meta", model=a.model, load_in_4bit=a.load_in_4bit))
        run_shutdown(scorer, emit)
        f.flush()
        run_dollar(scorer, emit)
    print(f"wrote {path}; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
