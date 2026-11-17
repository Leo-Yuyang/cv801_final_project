#!/usr/bin/env bash
#SBATCH --job-name=cv801-grid-stats
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=wkdrs/cv801-writer-reviewer-statistics/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_root}/.env"
cd "${project_root}"

output_dir="${project_root}/wkdrs/cv801-writer-reviewer-statistics"
samples="${output_dir}/grid_samples.csv"
analysis_json="${output_dir}/paired_analysis.json"
analysis_markdown="${output_dir}/paired_analysis.md"
report_tex="${output_dir}/report_tables.tex"
report_markdown="${output_dir}/report_tables.md"

for output in "${samples}" "${analysis_json}" "${analysis_markdown}" "${report_tex}" "${report_markdown}"; do
  [[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}." >&2; exit 1; }
done
mkdir -p "${output_dir}"

"${PYTHON_HOME}/bin/python" CV801/scripts/build_writer_reviewer_grid_samples.py \
  --cell '2B,low,wkdrs/cv801-agent-review/revision-full-grid-w2b-low-judge-repaired-r7-467067/metrics.golden.labeled.jsonl,wkdrs/cv801-independent-eval/grid-w2b-low-judge-repaired-r7/predictions.jsonl' \
  --cell '2B,mid,wkdrs/cv801-agent-review/revision-full-grid-w2b-mid-repaired-r3-462622/metrics.golden.labeled.jsonl,wkdrs/cv801-independent-eval/grid-w2b-mid-repaired-r3/predictions.jsonl' \
  --cell '2B,high,wkdrs/cv801-agent-review/revision-full-grid-w2b-high-judge-repaired-r3-467083/metrics.golden.labeled.jsonl,wkdrs/cv801-independent-eval/grid-w2b-high-judge-repaired-r3/predictions.jsonl' \
  --cell '4B,low,wkdrs/cv801-writer-reviewer-grid-r1/grid-w4b-low/revision-sharded-r7/downstream/metrics.golden.labeled.jsonl,wkdrs/cv801-independent-eval/grid-w4b-low-sharded-r7/predictions.jsonl' \
  --cell '4B,mid,wkdrs/cv801-writer-reviewer-grid-r1/grid-w4b-mid/revision-sharded-r4/downstream/metrics.golden.labeled.jsonl,wkdrs/cv801-independent-eval/grid-w4b-mid-sharded-r4/predictions.jsonl' \
  --cell '4B,high,wkdrs/cv801-writer-reviewer-grid-r1/grid-w4b-high/revision-sharded-r4/downstream/metrics.golden.labeled.jsonl,wkdrs/cv801-independent-eval/grid-w4b-high-sharded-r4/predictions.jsonl' \
  --cell '9B,low,wkdrs/cv801-agent-review/revision-full-grid-w9b-low-judge-repaired-v1-449692/metrics.golden.labeled.jsonl,wkdrs/cv801-independent-eval/grid-w9b-low-judge-repaired-v1/predictions.jsonl' \
  --cell '9B,mid,wkdrs/cv801-agent-review/revision-full-gemma12-internvl14-judge-repaired-v1-449662/metrics.golden.labeled.jsonl,wkdrs/cv801-independent-eval/gemma12-internvl14-judge-repaired-v1/predictions.jsonl' \
  --cell '9B,high,wkdrs/cv801-agent-review/revision-full-gemma27-internvl38-judge-repaired-v1-449647/metrics.golden.labeled.jsonl,wkdrs/cv801-independent-eval/gemma27-internvl38-judge-repaired-v1/predictions.jsonl' \
  --output "${samples}"

"${PYTHON_HOME}/bin/python" CV801/scripts/analyze_writer_reviewer_grid.py "${samples}" \
  --output-json "${analysis_json}" \
  --output-markdown "${analysis_markdown}" \
  --iterations 10000 \
  --seed 20261104 \
  --expected-samples 1000

"${PYTHON_HOME}/bin/python" CV801/scripts/render_writer_reviewer_report.py \
  "${analysis_json}" \
  --output-tex "${report_tex}" \
  --output-markdown "${report_markdown}"

install -m 0644 "${report_tex}" \
  "${project_root}/wkdrs/cv801-acl-paper/writer_reviewer_grid_tables.tex"
install -m 0644 \
  "${project_root}/wkdrs/cv801-writer-scale-analysis/report_addendum_draft.tex" \
  "${project_root}/wkdrs/cv801-acl-paper/writer_scale_addendum.tex"

bash execs/scpts/cv801_build_report.sh

echo "writer_reviewer_statistics_status=PASS samples=${samples} analysis=${analysis_json} report=${report_tex}"
