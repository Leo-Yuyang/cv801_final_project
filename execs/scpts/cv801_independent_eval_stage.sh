#!/usr/bin/env bash
#SBATCH --job-name=cv801-independent-stage
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=wkdrs/cv801-independent-eval/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_root}/.env"
stage="${STAGE:?Set STAGE to prepare, merge, or repair.}"
prepare_script="${project_root}/CV801/scripts/prepare_independent_vlm_evaluator_inputs.py"
row_ops="${project_root}/CV801/scripts/jsonl_row_ops.py"
audit_script="${project_root}/CV801/scripts/audit_independent_vlm_evaluator_outputs.py"
mkdir -p "${project_root}/wkdrs/cv801-independent-eval"

case "${stage}" in
  prepare)
    predictions="${PREDICTIONS:?Set PREDICTIONS.}"
    output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR.}"
    shards="${SHARDS:-4}"
    [[ -f "${predictions}" ]] || { echo "Predictions are missing: ${predictions}" >&2; exit 1; }
    [[ ! -e "${output_dir}" ]] || { echo "Refusing to overwrite ${output_dir}." >&2; exit 1; }
    mkdir -p "${output_dir}/shards"
    "${PYTHON_HOME}/bin/python" "${prepare_script}" \
      --predictions "${predictions}" \
      --output "${output_dir}/inputs.jsonl"
    test "$(wc -l < "${output_dir}/inputs.jsonl")" -eq 1000
    lines_per_shard=$(( (1000 + shards - 1) / shards ))
    split -d -a 2 -l "${lines_per_shard}" --additional-suffix=.jsonl \
      "${output_dir}/inputs.jsonl" "${output_dir}/shards/input-"
    test "$(find "${output_dir}/shards" -maxdepth 1 -type f -name 'input-*.jsonl' | wc -l)" -eq "${shards}"
    ;;
  merge)
    output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR.}"
    output="${output_dir}/predictions.jsonl"
    IFS=: read -r -a sources <<< "${SOURCE_PREDICTIONS:?Set SOURCE_PREDICTIONS.}"
    "${PYTHON_HOME}/bin/python" "${row_ops}" merge \
      --sources "${sources[@]}" \
      --reference-input "${output_dir}/inputs.jsonl" \
      --output "${output}"
    "${PYTHON_HOME}/bin/python" "${audit_script}" \
      --predictions "${output}" \
      --inputs "${output_dir}/inputs.jsonl" \
      --output "${output_dir}/metrics.json" \
      --require-all-valid
    ;;
  repair)
    output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR.}"
    repair_predictions="${REPAIR_PREDICTIONS:?Set REPAIR_PREDICTIONS.}"
    read -r -a line_numbers <<< "${LINE_NUMBERS:?Set LINE_NUMBERS.}"
    output="${output_dir}/predictions.jsonl"
    metrics="${output_dir}/metrics.json"
    source_predictions="${SOURCE_PREDICTIONS:-${output}}"
    repaired_output="${output_dir}/predictions.repaired-${SLURM_JOB_ID}.jsonl"
    repaired_metrics="${output_dir}/metrics.repaired-${SLURM_JOB_ID}.json"
    [[ -f "${output}" && -f "${metrics}" && -f "${source_predictions}" && -f "${repair_predictions}" ]] || {
      echo "Independent evaluator repair input is missing." >&2
      exit 1
    }
    "${PYTHON_HOME}/bin/python" "${row_ops}" replace \
      --source "${source_predictions}" \
      --replacements "${repair_predictions}" \
      --line-numbers "${line_numbers[@]}" \
      --output "${repaired_output}"
    "${PYTHON_HOME}/bin/python" "${audit_script}" \
      --predictions "${repaired_output}" \
      --inputs "${output_dir}/inputs.jsonl" \
      --output "${repaired_metrics}" \
      --require-all-valid
    mv "${output}" "${output_dir}/predictions.pre-repair-${SLURM_JOB_ID}.jsonl"
    mv "${metrics}" "${output_dir}/metrics.pre-repair-${SLURM_JOB_ID}.json"
    mv "${repaired_output}" "${output}"
    mv "${repaired_metrics}" "${metrics}"
    ;;
  *)
    echo "Unsupported STAGE: ${stage}" >&2
    exit 1
    ;;
esac
echo "independent_eval_stage_status=PASS stage=${stage}"
