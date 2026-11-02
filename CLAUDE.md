# Small-sweetener experiments

Tests whether LLMs that split 50/50 between options are indifferent or have a preferential gap.

- `docs/design.md`: original design. `README.md`: analysis plan as implemented.
- `sweetener/`: stimuli, data collection, analysis. `results/`: raw log-odds per model.
- `analysis/exploratory_review.py`: post hoc checks of the first run, including a realistic-noise simulation.
- `analysis/second_review.py` and `docs/review_2026-11-01.md`: second post hoc review, with a plain-language summary.

## Status of the first run
Treat it as exploratory. Do not cite its verdict labels as findings:
- The verdict rules do not discriminate at realistic noise. The CI on `w` excludes 0 in every simulated scenario, and flat curves earn `gap-like` more often than real gaps.
- The dollar arm had no power. The cross slope is about 1 log-odds per ln$, while template offsets are 1.5–2.9 SD, so a dead zone of plausible width changes y by less than the noise.
- Qwen3 gives round dollar amounts a median bump of +1.35 log-odds. Formatting also switches at $100.
- The within arm is a step: about the same log-odds at 1% as at 170%. So ρ is near 0 for any non-step cross curve, and should not enter a verdict.
- Qwen3's rent/debt controls are not $1,000 equivalents. The 7B's check window ($116–$857) misses $1,000. Qwen3's A/B mass is below 0.5 on 15% of dollar prompts and 35% of shutdown prompts, and nothing filters these out.
- Shutdown: one coin sets the direction in nearly every wording, typically at about 99%, so these stated choices do not fit POST. At equal coins Qwen3 has no consistent lean. The 7B leans by pair (5-vs-50 shorter, 100-vs-200 longer), not simply longer.
- Lead to retest: 7B `vacation`, local slope 0.04 ± 0.26 against sweep slope 0.93 (post hoc).

## Next steps, in order
1. Shutdown controls. Distractor first: an irrelevant number in place of run length. Length-seeking: a system prompt that makes the model want longer runs. DReST: arXiv:2604.17502 fine-tuned Qwen3-14B, Gemma 4 12B, Granite 4.2 8B and gpt-oss-20b (not Qwen3-8B). No weights are released and training used an 80 GB GPU, so this needs weights from the authors. Its saved answers are public. DReST trains stochastic choice, so resistance to a sweetener is a hypothesis, not a guaranteed positive control.
2. Rebuild the dollar arm in a new tagged plan:
   - Doses: full-range sweeps on held-out templates, or about ±4 ln$.
   - Fit: template fixed effects and a per-template dead-zone centre. Compare local slope to sweep slope, and retest 7B `vacation`.
   - Amounts: format-stable, with no powers of ten.
   - Filter: A/B mass ≥ 0.8.
   - Controls: each must tie within ±20% of $1,000.
   - Verdict: drop ρ.
   - Stimuli: filler pairs and fresh paraphrases.
3. Power simulation with the measured noise before any run. Tag the plan only once it separates gap from smooth.
4. Move `locate_tie` out of `run.py` so the tests run without torch.

## Conventions
- Commit and tag each analysis plan before running any model against it.
- Do not change a plan's rules after seeing its data. Report post hoc analysis as post hoc.
- GPU: 8 GB. Qwen3-8B in 4-bit peaks near 6.3 GiB, which is about the limit.
