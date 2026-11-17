#!/usr/bin/env bash
#SBATCH --job-name=cv801-grid-finalize
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=wkdrs/cv801-writer-reviewer-statistics/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${project_root}"
source "${project_root}/.env"

parent_bundle_job="${PARENT_BUNDLE_JOB:-486933}"
parent_run_dir="${project_root}/wkdrs/cv801-writer-reviewer-statistics/repair-bundle-${parent_bundle_job}"
parent_inputs="${parent_run_dir}/inputs.jsonl"
parent_predictions="${project_root}/wkdrs/cv801-independent-eval/paired-grid-independent-repair-bundle-${parent_bundle_job}/predictions.jsonl"
run_dir="${project_root}/wkdrs/cv801-writer-reviewer-statistics/residual-finalize-${SLURM_JOB_ID}"
mkdir -p "${run_dir}/cells"

# These are positions in the frozen 111-row repair bundle. A CPU pre-audit found
# that only these nine responses remained invalid after the first Pixtral pass.
residual_bundle_lines=(7 8 22 32 33 36 84 90 106)
raw_inputs="${run_dir}/inputs.raw.jsonl"
strict_inputs="${run_dir}/inputs.jsonl"
"${PYTHON_HOME}/bin/python" CV801/scripts/jsonl_row_ops.py select-lines \
  --source "${parent_inputs}" \
  --output "${raw_inputs}" \
  --line-numbers "${residual_bundle_lines[@]}"
"${PYTHON_HOME}/bin/python" CV801/scripts/jsonl_row_ops.py append-instruction \
  --source "${raw_inputs}" \
  --output "${strict_inputs}" \
  --instruction 'Return one compact RFC 8259 JSON object only. Do not use comments or Markdown. Aggregate duplicate object_ids into one object_review entry, keep rationale to at most three short strings, and close the JSON object before the token limit.'
test "$(wc -l < "${strict_inputs}")" -eq 9

run_label="paired-grid-independent-residual-final"
INPUT_FILE="${strict_inputs}" \
  RUN_LABEL="${run_label}" \
  MAX_NEW_TOKENS=4096 \
  bash execs/scpts/cv801_pixtral_evaluator_infer.sh
residual_predictions="${project_root}/wkdrs/cv801-independent-eval/${run_label}-${SLURM_JOB_ID}/predictions.jsonl"
test "$(wc -l < "${residual_predictions}")" -eq 9

final_bundle="${run_dir}/bundle-predictions.jsonl"
"${PYTHON_HOME}/bin/python" CV801/scripts/jsonl_row_ops.py replace-responses \
  --source "${parent_predictions}" \
  --replacements "${residual_predictions}" \
  --line-numbers "${residual_bundle_lines[@]}" \
  --output "${final_bundle}"
test "$(wc -l < "${final_bundle}")" -eq 111

default_repair_envs="wkdrs/cv801-independent-eval/grid-w2b-low-judge-repaired-r7/repairs/paired-grid-2b-low-r1-20261117T090714Z-1862026/repair.env:wkdrs/cv801-independent-eval/grid-w2b-mid-repaired-r3/repairs/paired-grid-2b-mid-r1-20261117T090714Z-1862057/repair.env:wkdrs/cv801-independent-eval/grid-w2b-high-judge-repaired-r3/repairs/paired-grid-2b-high-r1-20261117T090715Z-1862084/repair.env:wkdrs/cv801-independent-eval/grid-w4b-low-sharded-r7/repairs/paired-grid-4b-low-r1-20261117T090715Z-1862181/repair.env:wkdrs/cv801-independent-eval/grid-w4b-mid-sharded-r4/repairs/paired-grid-4b-mid-r1-20261117T090715Z-1862214/repair.env:wkdrs/cv801-independent-eval/grid-w4b-high-sharded-r4/repairs/paired-grid-4b-high-r1-20261117T090715Z-1862245/repair.env:wkdrs/cv801-independent-eval/grid-w9b-low-judge-repaired-v1/repairs/paired-grid-9b-low-r2-20261117T090716Z-1862280/repair.env:wkdrs/cv801-independent-eval/gemma12-internvl14-judge-repaired-v1/repairs/paired-grid-9b-mid-r2-20261117T090716Z-1862528/repair.env:wkdrs/cv801-independent-eval/gemma27-internvl38-judge-repaired-v1/repairs/paired-grid-9b-high-r2-20261117T090716Z-1862596/repair.env"
IFS=: read -r -a repair_envs <<< "${REPAIR_ENVS:-${default_repair_envs}}"
cell_keys=(2b-low 2b-mid 2b-high 4b-low 4b-mid 4b-high 9b-low 9b-mid 9b-high)
lineage_keys=("2B,low" "2B,mid" "2B,high" "4B,low" "4B,mid" "4B,high" "9B,low" "9B,mid" "9B,high")
[[ "${#repair_envs[@]}" -eq 9 ]] || { echo 'Expected nine repair environments.' >&2; exit 1; }

# Construct and strictly audit all nine canonical candidates before publishing
# any of them. This prevents a partial result grid from reaching statistics.
start=1
for index in "${!repair_envs[@]}"; do
  env_file="${repair_envs[index]}"
  source "${env_file}"
  strict_cell_inputs="$(dirname "${env_file}")/strict-inputs.jsonl"
  count="$(wc -l < "${strict_cell_inputs}")"
  end=$((start + count - 1))
  replacements="$(dirname "${env_file}")/bundled-predictions-${SLURM_JOB_ID}.jsonl"
  candidate_dir="${run_dir}/cells/${cell_keys[index]}"
  mkdir -p "${candidate_dir}"
  "${PYTHON_HOME}/bin/python" CV801/scripts/jsonl_row_ops.py select \
    --source "${final_bundle}" \
    --output "${replacements}" \
    --start "${start}" \
    --end "${end}"
  read -r -a line_numbers <<< "${invalid_lines//:/ }"
  "${PYTHON_HOME}/bin/python" CV801/scripts/jsonl_row_ops.py replace \
    --source "${source_predictions}" \
    --replacements "${replacements}" \
    --line-numbers "${line_numbers[@]}" \
    --output "${candidate_dir}/predictions.jsonl"
  "${PYTHON_HOME}/bin/python" CV801/scripts/audit_independent_vlm_evaluator_outputs.py \
    --predictions "${candidate_dir}/predictions.jsonl" \
    --inputs "${output_dir}/inputs.jsonl" \
    --output "${candidate_dir}/metrics.json" \
    --require-all-valid
  start=$((end + 1))
done
test "${start}" -eq 112

# All candidates passed. Preserve the previous canonical files and publish the
# fully valid replacements under the stable names consumed by analysis.
for index in "${!repair_envs[@]}"; do
  source "${repair_envs[index]}"
  candidate_dir="${run_dir}/cells/${cell_keys[index]}"
  test -f "${output_dir}/predictions.jsonl"
  test -f "${output_dir}/metrics.json"
  mv "${output_dir}/predictions.jsonl" "${output_dir}/predictions.pre-final-${SLURM_JOB_ID}.jsonl"
  mv "${output_dir}/metrics.json" "${output_dir}/metrics.pre-final-${SLURM_JOB_ID}.json"
  mv "${candidate_dir}/predictions.jsonl" "${output_dir}/predictions.jsonl"
  mv "${candidate_dir}/metrics.json" "${output_dir}/metrics.json"
done

lineage_args=()
for index in "${!repair_envs[@]}"; do
  lineage_args+=(--cell "${lineage_keys[index]},${repair_envs[index]}")
done
"${PYTHON_HOME}/bin/python" CV801/scripts/build_writer_reviewer_repair_lineage.py \
  "${lineage_args[@]}" \
  --bundle-job-id "${SLURM_JOB_ID}" \
  --parent-bundle-job-id "${parent_bundle_job}" \
  $(printf -- '--residual-bundle-line %s ' "${residual_bundle_lines[@]}") \
  --project-root "${project_root}" \
  --output-json "${project_root}/wkdrs/cv801-writer-reviewer-statistics/repair_lineage.json" \
  --output-markdown "${project_root}/wkdrs/cv801-writer-reviewer-statistics/repair_lineage.md"

bash execs/scpts/cv801_writer_reviewer_statistics.sh
echo "writer_reviewer_residual_finalize_status=PASS parent_bundle=${parent_bundle_job}"
