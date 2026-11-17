#!/usr/bin/env bash

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
report_dir="${project_root}/wkdrs/cv801-acl-paper"
tiny_bin="${project_root}/inits/tinytex/bin/x86_64-linux"
style_dir="${project_root}/inits/acl-style-files"

for executable in pdftex bibtex; do
  [[ -x "${tiny_bin}/${executable}" ]] || {
    echo "Missing TinyTeX executable: ${tiny_bin}/${executable}" >&2
    exit 1
  }
done
[[ -f "${style_dir}/acl.sty" && -f "${style_dir}/acl_natbib.bst" ]] || {
  echo "Missing ACL style files under ${style_dir}." >&2
  exit 1
}

cd "${report_dir}"
export TEXINPUTS=".:${style_dir}:"
export BSTINPUTS="${style_dir}:"
export BIBINPUTS=".:"

"${tiny_bin}/pdftex" --fmt=pdflatex -interaction=nonstopmode -halt-on-error main.tex
"${tiny_bin}/bibtex" main
"${tiny_bin}/pdftex" --fmt=pdflatex -interaction=nonstopmode -halt-on-error main.tex
"${tiny_bin}/pdftex" --fmt=pdflatex -interaction=nonstopmode -halt-on-error main.tex

[[ -s main.pdf ]] || { echo "Report PDF was not produced." >&2; exit 1; }
if rg -q 'undefined citations|There were undefined references|! LaTeX Error' main.log; then
  echo "Report contains unresolved citations, references, or LaTeX errors." >&2
  exit 1
fi

echo "cv801_report_build_status=PASS pdf=${report_dir}/main.pdf"
