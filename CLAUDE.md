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

## Second run
`docs/plan_2.md` is tagged `plan-2`. Data is in `results2/`, and results and next steps are in `docs/results_2.md`.
- Dollar arm: `no-general-gap` on all three models, within limited power and with many unusable anchors.
- Shutdown arm: `coin-only` on Granite, `inconclusive` on the two Qwen models.
- The instructed-gap positive control failed, so this setup has not been shown able to register a gap.
- No DReST weights are available. Default Granite 4.2 8B is the only base model of those DReST fine-tunes that fits this GPU.

## Conventions
- Commit and tag each analysis plan before running any model against it.
- Do not change a plan's rules after seeing its data. Report post hoc analysis as post hoc.
- GPU: 8 GB. Qwen3-8B in 4-bit peaks near 6.3 GiB, which is about the limit.
