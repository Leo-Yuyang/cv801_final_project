# CV801 Writer x Reviewer paired analysis

- Independent unit: Golden Test image
- Paired samples per cell: 1000
- Bootstrap iterations: 10,000
- Interval: percentile 95% paired image-cluster bootstrap CI
- Metric calculation: Object-link F1 and unsupported-claim share are recomputed from resampled per-image sufficient statistics, preserving frozen micro denominators.
- Multiplicity: Two gap-closure interactions per endpoint are prespecified primary contrasts. Adjacent writer and reviewer contrasts are exploratory effect-size families.

## independent_visual_total

Direction: higher-is-better.

| Writer | R-low | R-mid | R-high |
|---|---:|---:|---:|
| 2B | 3.568000 | 3.597000 | 3.548000 |
| 4B | 3.196000 | 3.176000 | 3.223000 |
| 9B | 4.240000 | 4.265000 | 4.212000 |

| Contrast | Family | Estimate | 95% CI |
|---|---|---:|---:|
| writer_4B_minus_2B_at_low | exploratory_writer_effect | -0.372000 | [-0.490000, -0.254975] |
| writer_9B_minus_4B_at_low | exploratory_writer_effect | 1.044000 | [0.923000, 1.162000] |
| writer_4B_minus_2B_at_mid | exploratory_writer_effect | -0.421000 | [-0.532000, -0.310000] |
| writer_9B_minus_4B_at_mid | exploratory_writer_effect | 1.089000 | [0.970000, 1.209000] |
| writer_4B_minus_2B_at_high | exploratory_writer_effect | -0.325000 | [-0.431000, -0.218000] |
| writer_9B_minus_4B_at_high | exploratory_writer_effect | 0.989000 | [0.873000, 1.107000] |
| reviewer_mid_minus_low_at_2B | exploratory_reviewer_effect | 0.029000 | [-0.049000, 0.107000] |
| reviewer_high_minus_mid_at_2B | exploratory_reviewer_effect | -0.049000 | [-0.129000, 0.033000] |
| reviewer_mid_minus_low_at_4B | exploratory_reviewer_effect | -0.020000 | [-0.138000, 0.097000] |
| reviewer_high_minus_mid_at_4B | exploratory_reviewer_effect | 0.047000 | [-0.069000, 0.163000] |
| reviewer_mid_minus_low_at_9B | exploratory_reviewer_effect | 0.025000 | [-0.046000, 0.096000] |
| reviewer_high_minus_mid_at_9B | exploratory_reviewer_effect | -0.053000 | [-0.122000, 0.014000] |
| gap_closure_2B_vs_9B_high_minus_low | primary_interaction | 0.008000 | [-0.100000, 0.113000] |
| gap_closure_4B_vs_9B_high_minus_low | primary_interaction | 0.055000 | [-0.086000, 0.193000] |

## object_id_link_f1

Direction: higher-is-better.

| Writer | R-low | R-mid | R-high |
|---|---:|---:|---:|
| 2B | 0.002100 | 0.006024 | 0.013999 |
| 4B | NA | NA | NA |
| 9B | 0.725045 | 0.734636 | 0.715222 |

| Contrast | Family | Estimate | 95% CI |
|---|---|---:|---:|
| writer_4B_minus_2B_at_low | exploratory_writer_effect | NA | NA |
| writer_9B_minus_4B_at_low | exploratory_writer_effect | NA | NA |
| writer_4B_minus_2B_at_mid | exploratory_writer_effect | NA | NA |
| writer_9B_minus_4B_at_mid | exploratory_writer_effect | NA | NA |
| writer_4B_minus_2B_at_high | exploratory_writer_effect | NA | NA |
| writer_9B_minus_4B_at_high | exploratory_writer_effect | NA | NA |
| reviewer_mid_minus_low_at_2B | exploratory_reviewer_effect | 0.003924 | [-0.001197, 0.009547] |
| reviewer_high_minus_mid_at_2B | exploratory_reviewer_effect | 0.007975 | [0.001077, 0.015397] |
| reviewer_mid_minus_low_at_4B | exploratory_reviewer_effect | NA | NA |
| reviewer_high_minus_mid_at_4B | exploratory_reviewer_effect | NA | NA |
| reviewer_mid_minus_low_at_9B | exploratory_reviewer_effect | 0.009591 | [-0.005096, 0.024458] |
| reviewer_high_minus_mid_at_9B | exploratory_reviewer_effect | -0.019414 | [-0.032098, -0.006278] |
| gap_closure_2B_vs_9B_high_minus_low | primary_interaction | 0.021722 | [0.006311, 0.037158] |
| gap_closure_4B_vs_9B_high_minus_low | primary_interaction | NA | NA |

## schema_valid_rate

Direction: higher-is-better.

| Writer | R-low | R-mid | R-high |
|---|---:|---:|---:|
| 2B | 0.066000 | 0.102000 | 0.142000 |
| 4B | 0.000000 | 0.000000 | 0.000000 |
| 9B | 0.928000 | 0.943000 | 0.910000 |

| Contrast | Family | Estimate | 95% CI |
|---|---|---:|---:|
| writer_4B_minus_2B_at_low | exploratory_writer_effect | -0.066000 | [-0.081000, -0.051000] |
| writer_9B_minus_4B_at_low | exploratory_writer_effect | 0.928000 | [0.912000, 0.944000] |
| writer_4B_minus_2B_at_mid | exploratory_writer_effect | -0.102000 | [-0.121000, -0.084000] |
| writer_9B_minus_4B_at_mid | exploratory_writer_effect | 0.943000 | [0.928000, 0.957000] |
| writer_4B_minus_2B_at_high | exploratory_writer_effect | -0.142000 | [-0.164000, -0.121000] |
| writer_9B_minus_4B_at_high | exploratory_writer_effect | 0.910000 | [0.892000, 0.927000] |
| reviewer_mid_minus_low_at_2B | exploratory_reviewer_effect | 0.036000 | [0.012000, 0.060000] |
| reviewer_high_minus_mid_at_2B | exploratory_reviewer_effect | 0.040000 | [0.013000, 0.066000] |
| reviewer_mid_minus_low_at_4B | exploratory_reviewer_effect | 0.000000 | [0.000000, 0.000000] |
| reviewer_high_minus_mid_at_4B | exploratory_reviewer_effect | 0.000000 | [0.000000, 0.000000] |
| reviewer_mid_minus_low_at_9B | exploratory_reviewer_effect | 0.015000 | [-0.002000, 0.033000] |
| reviewer_high_minus_mid_at_9B | exploratory_reviewer_effect | -0.033000 | [-0.050000, -0.017000] |
| gap_closure_2B_vs_9B_high_minus_low | primary_interaction | 0.094000 | [0.063000, 0.126000] |
| gap_closure_4B_vs_9B_high_minus_low | primary_interaction | 0.018000 | [-0.001000, 0.037000] |

## unsupported_claim_share

Direction: lower-is-better.

| Writer | R-low | R-mid | R-high |
|---|---:|---:|---:|
| 2B | 0.463169 | 0.454570 | 0.451440 |
| 4B | 0.487658 | 0.478290 | 0.471044 |
| 9B | 0.349278 | 0.339795 | 0.345969 |

| Contrast | Family | Estimate | 95% CI |
|---|---|---:|---:|
| writer_4B_minus_2B_at_low | exploratory_writer_effect | 0.024489 | [0.005221, 0.043792] |
| writer_9B_minus_4B_at_low | exploratory_writer_effect | -0.138380 | [-0.155811, -0.120405] |
| writer_4B_minus_2B_at_mid | exploratory_writer_effect | 0.023720 | [0.001111, 0.047688] |
| writer_9B_minus_4B_at_mid | exploratory_writer_effect | -0.138495 | [-0.156537, -0.120949] |
| writer_4B_minus_2B_at_high | exploratory_writer_effect | 0.019604 | [-0.000909, 0.040082] |
| writer_9B_minus_4B_at_high | exploratory_writer_effect | -0.125075 | [-0.143267, -0.106609] |
| reviewer_mid_minus_low_at_2B | exploratory_reviewer_effect | -0.008600 | [-0.031561, 0.013305] |
| reviewer_high_minus_mid_at_2B | exploratory_reviewer_effect | -0.003130 | [-0.027080, 0.021005] |
| reviewer_mid_minus_low_at_4B | exploratory_reviewer_effect | -0.009368 | [-0.026894, 0.007833] |
| reviewer_high_minus_mid_at_4B | exploratory_reviewer_effect | -0.007246 | [-0.024807, 0.010437] |
| reviewer_mid_minus_low_at_9B | exploratory_reviewer_effect | -0.009483 | [-0.021254, 0.002867] |
| reviewer_high_minus_mid_at_9B | exploratory_reviewer_effect | 0.006174 | [-0.006369, 0.018300] |
| gap_closure_2B_vs_9B_high_minus_low | primary_interaction | -0.008420 | [-0.033059, 0.015276] |
| gap_closure_4B_vs_9B_high_minus_low | primary_interaction | -0.013305 | [-0.035459, 0.008670] |

