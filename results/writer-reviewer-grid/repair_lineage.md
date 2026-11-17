# Writer x Reviewer evaluator-repair lineage

Only invalid independent-evaluator rows were rerun with the same frozen Pixtral-12B evaluator protocol; valid rows were retained.

- Bundle Slurm job: `487444`
- Repaired rows: 111 of 9,000 evaluator rows
- Line numbers are one-based positions in each cell's canonical prediction file.
- Parent 111-row bundle job: `486933`
- Residual responses rerun by the finalizer: 9 (one-based bundle rows: 7, 8, 22, 32, 33, 36, 84, 90, 106)

| Writer | Reviewer | Repaired rows | One-based line numbers |
|---|---|---:|---|
| 2B | R-low | 12 | 20, 147, 223, 243, 351, 429, 441, 452, 495, 515, 907, 933 |
| 2B | R-mid | 11 | 10, 294, 341, 482, 485, 489, 511, 700, 713, 728, 906 |
| 2B | R-high | 17 | 24, 84, 124, 149, 267, 386, 393, 399, 449, 467, 551, 629, 681, 713, 715, 774, 905 |
| 4B | R-low | 5 | 331, 354, 474, 521, 666 |
| 4B | R-mid | 7 | 354, 681, 730, 752, 758, 795, 902 |
| 4B | R-high | 8 | 354, 510, 601, 690, 752, 849, 866, 938 |
| 9B | R-low | 12 | 510, 518, 520, 525, 538, 591, 609, 690, 694, 726, 824, 866 |
| 9B | R-mid | 19 | 511, 513, 518, 521, 526, 538, 609, 656, 658, 687, 701, 716, 821, 824, 825, 828, 829, 853, 939 |
| 9B | R-high | 20 | 99, 154, 505, 511, 518, 538, 591, 609, 684, 694, 695, 774, 831, 843, 848, 862, 935, 948, 950, 974 |

The complete machine-readable paths and replacement files are recorded in `repair_lineage.json`. Repair does not remove invalid outputs from the denominator.
