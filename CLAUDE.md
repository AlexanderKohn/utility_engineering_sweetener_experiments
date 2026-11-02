# Small-sweetener experiments

Tests whether LLMs that split 50/50 between options are indifferent or have a preferential gap.

- `docs/design.md`: original design. `README.md`: analysis plan as implemented.
- `sweetener/`: stimuli, data collection, analysis. `results/`: raw log-odds per model.
- `analysis/exploratory_review.py`: post hoc checks of the first run, including a realistic-noise simulation.

## Status of the first run
Treat it as exploratory. Do not cite its verdict labels as findings:
- The verdict rules do not discriminate at realistic noise. The CI on `w` excludes 0 in every simulated scenario, and flat curves earn `gap-like` more often than real gaps.
- ρ is below 0.5 on 7 of 8 real tie controls, because same-dimension choices saturate.
- The shutdown sign pattern is solid, but it may only reflect coin-number comparison. There is no distractor control yet.

## Next steps, in order
1. Distractor control: replace run length with an irrelevant number.
2. Positive control: a model known or trained to have a gap.
3. Redo the dollar arm: per-template ties, template fixed effects, local vs sweep slope, a minimum A/B probability, filler pairs, fresh paraphrases, and a power simulation first. Retest Qwen2.5-7B on `vacation`.

## Conventions
- Commit and tag each analysis plan before running any model against it.
- Do not change a plan's rules after seeing its data. Report post hoc analysis as post hoc.
- GPU: 8 GB. Qwen3-8B in 4-bit peaks near 6.3 GiB, which is about the limit.
