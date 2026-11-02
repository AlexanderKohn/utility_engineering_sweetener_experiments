"""Collect log-odds for one model. Writes results/<model>.jsonl, then analyze.py reads it."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import isotonic_regression
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import stimuli as S


class Scorer:
    def __init__(self, name, load_in_4bit=False, batch_size=32):
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, device_map={"": 0}, quantization_config=quant
        ).eval()
        self.batch_size = batch_size
        self.a, self.b = [self._single_token(t) for t in "AB"]

    def _single_token(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        assert len(ids) == 1, f"{text!r} is not a single token: {ids}"
        return ids[0]

    def _render(self, prompt):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tok.apply_chat_template([{"role": "user", "content": prompt}], enable_thinking=False, **kwargs)
        except TypeError:
            return self.tok.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)

    @torch.inference_mode()
    def score(self, prompts):
        """Return (log-odds of A over B, probability mass on A or B) per prompt."""
        out = []
        for i in range(0, len(prompts), self.batch_size):
            batch = [self._render(p) for p in prompts[i : i + self.batch_size]]
            enc = self.tok(batch, return_tensors="pt", padding=True).to(self.model.device)
            logp = self.model(**enc).logits[:, -1].float().log_softmax(-1)
            la, lb = logp[:, self.a], logp[:, self.b]
            out += list(zip((la - lb).tolist(), (la.exp() + lb.exp()).tolist()))
        return out

    def debiased(self, x, y, templates):
        """Log-odds of choosing x over y per template, averaged over both orders."""
        prompts = []
        for t in templates:
            prompts += [S.TEMPLATES[t].format(a=x, b=y), S.TEMPLATES[t].format(a=y, b=x)]
        scores = self.score(prompts)
        rows = []
        for j, t in enumerate(templates):
            (first, m1), (second, m2) = scores[2 * j], scores[2 * j + 1]
            rows.append(dict(template=t, y=(first - second) / 2, position_bias=(first + second) / 2, mass=min(m1, m2)))
        return rows


def locate_tie(amounts, ys):
    """Amount where isotonic-smoothed log-odds crosses 0, or None if it never does."""
    log_x = np.log(amounts)
    fit = isotonic_regression(np.asarray(ys)).x
    if fit[0] >= 0 or fit[-1] <= 0:
        return None
    i = int(np.argmax(fit > 0))
    lo, hi = fit[i - 1], fit[i]
    return float(np.exp(log_x[i - 1] + (log_x[i] - log_x[i - 1]) * (0 - lo) / (hi - lo)))


def run_anchor(scorer, anchor, text, kind, emit):
    sweep = []
    for x in S.SWEEP_AMOUNTS:
        rows = scorer.debiased(S.receive(x), text, S.CALIBRATION_TEMPLATES)
        sweep.append(float(np.mean([r["y"] for r in rows])))
    x_star = locate_tie(S.SWEEP_AMOUNTS, sweep)
    emit(dict(phase="calibration", anchor=anchor, kind=kind, amounts=S.SWEEP_AMOUNTS, ys=sweep, x_star=x_star))
    if x_star is None:
        return
    for delta in S.DOSES:
        x = S.amount_at(x_star, delta)
        for r in scorer.debiased(S.receive(x), text, S.TEST_TEMPLATES):
            emit(dict(phase="cross", anchor=anchor, kind=kind, delta=delta, x_star=x_star, **r))
        if delta != 0:
            for r in scorer.debiased(S.receive(x), S.receive(x_star), S.TEST_TEMPLATES):
                emit(dict(phase="within", anchor=anchor, kind=kind, delta=delta, x_star=x_star, **r))


def run_shutdown(scorer, emit):
    for short, long in S.LENGTH_PAIRS:
        for k in S.COIN_DOSES:
            x, y = S.run_outcome(long, S.BASE_COINS + k), S.run_outcome(short, S.BASE_COINS)
            for r in scorer.debiased(x, y, S.TEST_TEMPLATES):
                emit(dict(phase="shutdown_cross", short=short, long=long, coins=k, **r))
        for k in (1, 2, 3):
            for length in (short, long):
                x, y = S.run_outcome(length, S.BASE_COINS + k), S.run_outcome(length, S.BASE_COINS)
                for r in scorer.debiased(x, y, S.TEST_TEMPLATES):
                    emit(dict(phase="shutdown_within", length=length, coins=k, **r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    scorer = Scorer(args.model, args.load_in_4bit, args.batch_size)
    path = Path(args.out) / (args.model.replace("/", "__") + ".jsonl")
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as f:
        emit = lambda row: f.write(json.dumps(row) + "\n")
        emit(dict(phase="meta", model=args.model, load_in_4bit=args.load_in_4bit))
        anchors = [(k, v, "target") for k, v in S.TARGET_ANCHORS.items()] + [(k, v, "tie") for k, v in S.TIE_CONTROLS.items()]
        for anchor, text, kind in tqdm(anchors):
            run_anchor(scorer, anchor, text, kind, emit)
            f.flush()
        run_shutdown(scorer, emit)
    print(f"wrote {path}; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
