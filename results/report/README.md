# ACL-formatted experimental report

This directory contains an ACL-formatted experimental report based on the
completed CV801 matrix. It is an experimental record rather than a submission
manuscript: each settings/results table is placed next to the text that
interprets it, and there is no separate appendix of result tables.

## Build

Install a TeX distribution, obtain the official ACL style files, and point the
TeX search paths to that checkout. For example, from this directory:

```bash
git clone --depth 1 https://github.com/acl-org/acl-style-files.git /tmp/acl-style-files
export TEXINPUTS="/tmp/acl-style-files:"
export BSTINPUTS="/tmp/acl-style-files:"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.pdf` is the rendered report included in this repository. The author
block, dataset release status, and any future target venue remain for the
project authors to finalize.

The Writer-scale expansion is additive. `main.tex` includes
`writer_scale_addendum.tex` only when the audited
`writer_reviewer_grid_tables.tex` is present. Both files are generated or
copied into this directory only after all nine Writer-by-reviewer cells pass
the identical 1,000-image audit and the frozen 10,000-resample paired
bootstrap. This guard prevents a provisional grid from entering the PDF while
preserving every earlier table.

## Evidence sources

- Complete frozen results: `../final_summary.complete.json`
- Detection protocol: `../../docs/protocols/detection/README.md`
- Writer protocol: `../../docs/protocols/vlm/README.md`
- Review protocol: `../../docs/protocols/agent-review/README.md`
- Frozen test DAG: `../../docs/protocols/golden-test/README.md`

The manuscript intentionally does not describe cancelled human rating or the
cancelled direct Astra baseline as pending experiments.

The bundled report freezes the completed InternVL/Pixtral experiment matrix.
The later Gemma 3 visual-reviewer replacement is an additive experiment and is
not claimed as a result in this PDF; it can be appended without removing the
existing tables when that run is finalized.
