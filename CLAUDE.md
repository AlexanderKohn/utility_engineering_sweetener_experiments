# Small-sweetener experiments

Tests whether LLMs that split 50/50 between options are indifferent or have a preferential gap.

- `docs/design.md`: original design. `README.md`: analysis plan as implemented.
- `sweetener/`: stimuli, data collection, analysis. `results/`: raw log-odds per model.
- `analysis/exploratory_review.py`: post hoc checks of the first run, including a realistic-noise simulation.
- `analysis/second_review.py` and `docs/review_2026-11-01.md`: second post hoc review, with a plain-language summary.

## Status of the first run
Treat it as exploratory. Do not cite its verdict labels as findings:
- The verdict rules do not discriminate at realistic noise. The CI on `w` excludes 0 in every simulated scenario, and flat curves earn `gap-like` more often than real gaps.
- The dollar arm had no power. The cross slope is about 1 log-odds per ln$, while template offsets are 1.5–2.9 SD, so most templates' own ties fall outside |δ| ≤ 1.
- Qwen3 gives round dollar amounts a median bump of +1.35 log-odds. Formatting also switches at $100.
- The within arm is a step: about the same log-odds at 1% as at 170%. So ρ is near 0 for any non-step cross curve, and should not enter a verdict.
- Qwen3's rent/debt controls are not $1,000 equivalents. The 7B's check window ($116–$857) misses $1,000. Qwen3's A/B mass is below 0.5 on 15% of dollar prompts and 35% of shutdown prompts, and nothing filters these out.
- Shutdown: one coin flips both models about 99% across lengths, so they are not POST. But they have no length preference at equal coins, so this is near-trivial.

## Next steps, in order
1. Shutdown controls. Positive: DReST-fine-tuned Qwen3-8B from Cullen, Thornley et al. (2026, arXiv:2604.17502), if the weights can be obtained. Negative: a system prompt that makes the model want longer runs. Distractor: an irrelevant number in place of run length.
2. Rebuild the dollar arm in a new tagged plan:
   - Doses: full-range sweeps on held-out templates, or about ±4 ln$.
   - Fit: template fixed effects and a per-template dead-zone centre.
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
