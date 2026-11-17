# CV801 Writer × Reviewer statistical analysis plan

## Scope and analysis unit

This plan covers the controlled `3 × 3` grid of T2 Writers (`2B`, `4B`, and
`9B`) and reviewer tiers (`R-low`, `R-mid`, and `R-high`) on the same 1,000
Golden Test images. The independent unit is one Golden Test image. Object-level
claims and links remain clustered within image and are never counted as
independent samples.

The analysis is paired because every configuration is evaluated on the same
frozen image set. A cell is eligible only when it contains exactly 1,000 unique
sample IDs and all nine cells have the same ordered ID set.

## Frozen endpoints

The four primary endpoints are:

1. independent visual total (`0–6`, higher is better);
2. object-ID link F1 (`0–1`, higher is better);
3. schema-valid indicator (`0/1`, higher is better); and
4. unsupported-claim share (`0–1`, lower is better).

Secondary endpoints include the three independent visual sub-scores, exact
object count, count MAE, caption F1, categorical field agreement, action F1,
confidence MAE, latency, peak memory, and GPU-hours. BLEU-4 is retained as a
corpus summary but is not used as an image-level paired endpoint.

## Prespecified comparisons

The primary interaction question is whether stronger reviewers close the
quality gap between a small Writer and the 9B Writer. For each endpoint, report:

- `[(2B, high) − (9B, high)] − [(2B, low) − (9B, low)]`; and
- `[(4B, high) − (9B, high)] − [(4B, low) − (9B, low)]`.

For higher-is-better endpoints, a positive value means that `R-high` closes the
small-Writer gap relative to `R-low`. For unsupported-claim share and other
lower-is-better endpoints, multiply values by `−1` before applying this
interpretation or state the raw direction explicitly.

The following are exploratory effect-size families:

- Writer scale: `4B − 2B` and `9B − 4B` within each reviewer tier;
- reviewer scale: `R-mid − R-low` and `R-high − R-mid` within each Writer.

Do not infer an interaction merely because one within-cell comparison crosses
zero and another does not. Report the direct difference-in-differences above.

## Uncertainty and multiplicity

Use 10,000 paired image-cluster bootstrap resamples with seed `20261104` and
percentile 95% confidence intervals. Micro object-link F1 and unsupported-claim
share are recomputed from per-image numerator/denominator components within
every resample; they are not approximated by averaging per-image ratios. Report
the observed paired difference and interval for every prespecified contrast. The two gap-closure interactions
per endpoint are primary. Adjacent Writer/reviewer contrasts are exploratory
and must be interpreted jointly as effect-size families, not as isolated
significance tests.

The main report should emphasize effect size and uncertainty rather than binary
`significant/non-significant` wording. If formal p-values are later requested,
define the test statistic and apply Holm correction within each endpoint and
contrast family; do not retrofit uncorrected p-values after inspecting results.

## Missingness, repairs, and exclusions

- No Golden Test image is excluded because a model produced invalid JSON.
  Invalid or schema-invalid output receives a zero validity indicator and the
  metric-specific missing/invalid handling defined by the frozen evaluator.
- Targeted syntax/schema repair is allowed only when the original model content
  is preserved or the sample is rerun under the same frozen decoding protocol.
  Every repair lineage and replaced sample ID must remain auditable.
- A configuration with fewer than 1,000 unique IDs, duplicate IDs, or a
  different ID set is not analyzed; it is marked incomplete.
- Failed or superseded runs remain in provenance logs but never mix with the
  final 1,000-row configuration.

## Cost–performance analysis

For each logical system, report allocated GPU-hours, execution span excluding
queue wait, seconds per image, and peak allocated GPU memory when available.
GPU-hours are calculated from Slurm elapsed allocation time and requested GPU
count. Shared stages are charged consistently across configurations and the
charging rule is stated in the table caption.

Construct a measured Pareto frontier using quality endpoints and GPU-hours.
Parameter count is configuration metadata, not a substitute for measured
compute cost.

## Reporting placement

- Main result table: nine cell means for the four primary endpoints.
- Interaction table: the two small-Writer gap-closure contrasts with 95% CIs.
- Efficiency table: GPU-hours, seconds/image, and visual total.
- Supplementary tables: all adjacent contrasts, secondary endpoints, repair
  provenance, and per-job accounting.
- Result prose must remain adjacent to its table and must not overwrite the
  already reported 9B experiments.

The executable analysis is `CV801/scripts/analyze_writer_reviewer_grid.py`.
