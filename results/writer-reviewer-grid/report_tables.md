# Writer x Reviewer report tables

Generated only after the complete 1,000-image paired audit. Values are derived from 10,000 paired image-cluster bootstrap resamples with seed `20261104`.

## Independent visual total

| T2 Writer | R-low | R-mid | R-high |
|---|---:|---:|---:|
| 2B | 3.568 | 3.597 | 3.548 |
| 4B | 3.196 | 3.176 | 3.223 |
| 9B | 4.240 | 4.265 | 4.212 |

| Gap closure | Estimate | 95% CI |
|---|---:|---:|
| 2B vs. 9B, R-high minus R-low | +0.008 | [-0.100, +0.113] |
| 4B vs. 9B, R-high minus R-low | +0.055 | [-0.086, +0.193] |

## Object-ID link F1

| T2 Writer | R-low | R-mid | R-high |
|---|---:|---:|---:|
| 2B | 0.2% | 0.6% | 1.4% |
| 4B | -- | -- | -- |
| 9B | 72.5% | 73.5% | 71.5% |

| Gap closure | Estimate | 95% CI |
|---|---:|---:|
| 2B vs. 9B, R-high minus R-low | +2.2 pp | [+0.6 pp, +3.7 pp] |
| 4B vs. 9B, R-high minus R-low | -- | -- |

## Schema-valid rate

| T2 Writer | R-low | R-mid | R-high |
|---|---:|---:|---:|
| 2B | 6.6% | 10.2% | 14.2% |
| 4B | 0.0% | 0.0% | 0.0% |
| 9B | 92.8% | 94.3% | 91.0% |

| Gap closure | Estimate | 95% CI |
|---|---:|---:|
| 2B vs. 9B, R-high minus R-low | +9.4 pp | [+6.3 pp, +12.6 pp] |
| 4B vs. 9B, R-high minus R-low | +1.8 pp | [-0.1 pp, +3.7 pp] |

## Unsupported-claim share

| T2 Writer | R-low | R-mid | R-high |
|---|---:|---:|---:|
| 2B | 46.3% | 45.5% | 45.1% |
| 4B | 48.8% | 47.8% | 47.1% |
| 9B | 34.9% | 34.0% | 34.6% |

| Gap closure | Estimate | 95% CI |
|---|---:|---:|
| 2B vs. 9B, R-high minus R-low | -0.8 pp | [-3.3 pp, +1.5 pp] |
| 4B vs. 9B, R-high minus R-low | -1.3 pp | [-3.5 pp, +0.9 pp] |

The tables intentionally contain no automated causal or superiority claim. Interpretation must combine all four endpoints and the stated direction of each metric.
