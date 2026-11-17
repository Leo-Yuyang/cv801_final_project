# CV801 Writer-Scale Extension

This additive release records the frozen Writer-only portion of the CV801
scaling extension. It does not replace the original 9B, detector, reviewer-swap,
or Agent Review results. The paired `3 x 3` Writer-by-reviewer analysis is
published only as an atomic audited bundle: `grid_samples.csv`,
`paired_analysis.json`, `paired_analysis.md`, report-ready tables, and the
repair-lineage record must all be present under `results/writer-reviewer-grid/`.
If that directory is absent from a release snapshot, the snapshot predates the
complete common 1,000-ID audit and must not be interpreted as a partial grid.

## Controlled setup

Qwen3.5-2B, 4B, and 9B were evaluated on the same 1,000-image Golden Test.
T0 is the unadapted whole-image model, T1 is a three-epoch domain LoRA, and T2
uses the same LoRA protocol with frozen YOLO26x proposal hints. Prompts, target
responses, deterministic decoding, the 4,096-token Golden generation cap, and
evaluation code were held fixed. Each adapted size uses one training seed.

## Frozen Writer-only results

| Writer | Variant | JSON valid | Schema valid | Count exact | Object-ID F1 | Caption F1 | Visual total | Unsupported share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2B | T0 | 82.4% | 0.0% | 45.4% | 35.5% | 20.7% | 3.429 | 44.6% |
| 2B | T1 | 0.0% | 0.0% | 0.0% | n/a | n/a | 3.588 | 32.9% |
| 2B | T2 | 0.1% | 0.1% | 0.1% | 0.0% | 31.3% | 3.505 | 41.0% |
| 4B | T0 | 69.4% | 0.0% | 4.3% | 27.1% | 16.7% | 3.078 | 44.6% |
| 4B | T1 | 0.0% | 0.0% | 0.0% | n/a | n/a | 3.248 | 56.3% |
| 4B | T2 | 5.2% | 5.2% | 4.5% | 5.8% | 45.5% | 4.631 | 33.7% |
| 9B | T0 | 97.6% | 0.0% | 57.2% | 48.9% | 5.8% | 2.650 | 61.9% |
| 9B | T1 | 88.8% | 88.8% | 52.9% | 59.7% | 49.1% | 4.172 | 30.2% |
| 9B | T2 | 99.5% | 99.5% | 64.1% | 76.7% | 51.3% | 4.175 | 33.2% |

All percentages use the full 1,000-image denominator. Invalid output is not
dropped. `Visual total` is a model-based independent score in `[0, 6]`, and
`unsupported share` is a model-based unsupported-claim fraction rather than a
human hallucination rate. Undefined structured metrics are preserved as `n/a`.

The main finding is an interface-validity boundary under the frozen protocol:
the adapted 2B and 4B Writers did not retain the 9B Writer's machine-readable
reliability. The 4B-T2 visual total of 4.631 must therefore not be interpreted
as end-to-end superiority because only 5.2% of its reports passed the schema.

## Artifact map

- `writer_results.csv`: machine-readable aggregate table above.
- `writer_inference_efficiency.json`: Slurm-derived GPU-hours, execution span,
  seconds per image, requested GPU count, and accepted post-inference audit
  failures for all nine Writer/variant runs.
- `2b/` and `4b/`: T0/T1/T2 Writer predictions, structured metrics, and
  independent-evaluator inputs, predictions, and metrics.
- `statistical_plan.md`: frozen paired-bootstrap and interaction analysis for
  the completed Writer-by-reviewer grid.
- `report_addendum_draft.tex`: additive report section with the Writer-only
  table and the completed grid interpretation.
- `../writer-reviewer-grid/`: complete paired-grid statistics, report-ready
  tables, per-image sufficient statistics, and the 111-row repair lineage.

The matching 2B/4B T1/T2 adapters are stored in
[`leo20000306/cv801_models`](https://huggingface.co/leo20000306/cv801_models).
The executable pipeline is in
[`Leo-Yuyang/cv801_final_project`](https://github.com/Leo-Yuyang/cv801_final_project).
