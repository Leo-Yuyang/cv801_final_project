#!/usr/bin/env bash
#SBATCH --job-name=cv801-grid-repair1
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:15:00
#SBATCH --output=wkdrs/cv801-writer-reviewer-statistics/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${project_root}"
source "${project_root}/.env"

default_repair_envs="wkdrs/cv801-independent-eval/grid-w2b-low-judge-repaired-r7/repairs/paired-grid-2b-low-r1-20261117T090714Z-1862026/repair.env:wkdrs/cv801-independent-eval/grid-w2b-mid-repaired-r3/repairs/paired-grid-2b-mid-r1-20261117T090714Z-1862057/repair.env:wkdrs/cv801-independent-eval/grid-w2b-high-judge-repaired-r3/repairs/paired-grid-2b-high-r1-20261117T090715Z-1862084/repair.env:wkdrs/cv801-independent-eval/grid-w4b-low-sharded-r7/repairs/paired-grid-4b-low-r1-20261117T090715Z-1862181/repair.env:wkdrs/cv801-independent-eval/grid-w4b-mid-sharded-r4/repairs/paired-grid-4b-mid-r1-20261117T090715Z-1862214/repair.env:wkdrs/cv801-independent-eval/grid-w4b-high-sharded-r4/repairs/paired-grid-4b-high-r1-20261117T090715Z-1862245/repair.env:wkdrs/cv801-independent-eval/grid-w9b-low-judge-repaired-v1/repairs/paired-grid-9b-low-r2-20261117T090716Z-1862280/repair.env:wkdrs/cv801-independent-eval/gemma12-internvl14-judge-repaired-v1/repairs/paired-grid-9b-mid-r2-20261117T090716Z-1862528/repair.env:wkdrs/cv801-independent-eval/gemma27-internvl38-judge-repaired-v1/repairs/paired-grid-9b-high-r2-20261117T090716Z-1862596/repair.env"
IFS=: read -r -a repair_envs <<< "${REPAIR_ENVS:-${default_repair_envs}}"
[[ "${#repair_envs[@]}" -eq 9 ]] || { echo "Expected nine repair.env paths." >&2; exit 1; }
cell_keys=("2B,low" "2B,mid" "2B,high" "4B,low" "4B,mid" "4B,high" "9B,low" "9B,mid" "9B,high")

run_dir="${project_root}/wkdrs/cv801-writer-reviewer-statistics/repair-bundle-${SLURM_JOB_ID}"
mkdir -p "${run_dir}"
strict_inputs=()
for env_file in "${repair_envs[@]}"; do
  [[ -f "${env_file}" ]] || { echo "Missing repair metadata: ${env_file}" >&2; exit 1; }
  input="$(dirname "${env_file}")/strict-inputs.jsonl"
  [[ -f "${input}" ]] || { echo "Missing strict repair inputs: ${input}" >&2; exit 1; }
  strict_inputs+=("${input}")
done

combined_inputs="${run_dir}/inputs.jsonl"
"${PYTHON_HOME}/bin/python" CV801/scripts/jsonl_row_ops.py concat \
  --sources "${strict_inputs[@]}" \
  --output "${combined_inputs}"
test "$(wc -l < "${combined_inputs}")" -eq 111

run_label="paired-grid-independent-repair-bundle"
INPUT_FILE="${combined_inputs}" \
  RUN_LABEL="${run_label}" \
  MAX_NEW_TOKENS=2048 \
  bash execs/scpts/cv801_pixtral_evaluator_infer.sh
combined_predictions="${project_root}/wkdrs/cv801-independent-eval/${run_label}-${SLURM_JOB_ID}/predictions.jsonl"
test "$(wc -l < "${combined_predictions}")" -eq 111

start=1
for index in "${!repair_envs[@]}"; do
  env_file="${repair_envs[index]}"
  count="$(wc -l < "${strict_inputs[index]}")"
  end=$((start + count - 1))
  replacements="$(dirname "${env_file}")/bundled-predictions-${SLURM_JOB_ID}.jsonl"
  "${PYTHON_HOME}/bin/python" CV801/scripts/jsonl_row_ops.py select \
    --source "${combined_predictions}" \
    --output "${replacements}" \
    --start "${start}" \
    --end "${end}"
  (
    source "${env_file}"
    line_numbers_space="${invalid_lines//:/ }"
    STAGE=repair \
      OUTPUT_DIR="${output_dir}" \
      SOURCE_PREDICTIONS="${source_predictions}" \
      REPAIR_PREDICTIONS="${replacements}" \
      LINE_NUMBERS="${line_numbers_space}" \
      bash execs/scpts/cv801_independent_eval_stage.sh
  )
  start=$((end + 1))
done
test "${start}" -eq 112

lineage_args=()
for index in "${!repair_envs[@]}"; do
  lineage_args+=(--cell "${cell_keys[index]},${repair_envs[index]}")
done
"${PYTHON_HOME}/bin/python" CV801/scripts/build_writer_reviewer_repair_lineage.py \
  "${lineage_args[@]}" \
  --bundle-job-id "${SLURM_JOB_ID}" \
  --project-root "${project_root}" \
  --output-json "${project_root}/wkdrs/cv801-writer-reviewer-statistics/repair_lineage.json" \
  --output-markdown "${project_root}/wkdrs/cv801-writer-reviewer-statistics/repair_lineage.md"

bash execs/scpts/cv801_writer_reviewer_statistics.sh
echo "writer_reviewer_single_gpu_repair_status=PASS"
