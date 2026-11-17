#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

run_tag="${RUN_TAG:?Set RUN_TAG.}"
output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR.}"
line_numbers_value="${LINE_NUMBERS:?Set colon-separated LINE_NUMBERS.}"
source_predictions="${SOURCE_PREDICTIONS:-${output_dir}/predictions.jsonl}"

[[ "${run_tag}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || exit 1
[[ "${line_numbers_value}" =~ ^[0-9]+(:[0-9]+)*$ ]] || exit 1
for required in "${output_dir}/inputs.jsonl" "${output_dir}/predictions.jsonl" \
  "${output_dir}/metrics.json" "${source_predictions}"; do
  [[ -f "${required}" ]] || { echo "Missing repair input: ${required}" >&2; exit 1; }
done

python_bin="${project_root}/.venv-cv801/bin/python"
row_ops="${project_root}/CV801/scripts/jsonl_row_ops.py"
repair_root="${output_dir}/repairs/${run_tag}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "${repair_root}"

IFS=: read -r -a line_numbers <<< "${line_numbers_value}"
selected_inputs="${repair_root}/selected-inputs.jsonl"
strict_inputs="${repair_root}/strict-inputs.jsonl"
"${python_bin}" "${row_ops}" select-lines \
  --source "${output_dir}/inputs.jsonl" --output "${selected_inputs}" \
  --line-numbers "${line_numbers[@]}"
"${python_bin}" "${row_ops}" append-instruction \
  --source "${selected_inputs}" --output "${strict_inputs}" \
  --instruction 'Return exactly one strict JSON object and no prose. Use exactly these top-level keys: sample_id, object_reviews, scores, total, and rationale. Each object review must contain object_id, supported_claims, unsupported_claims, and object_linkage. supported_claims and unsupported_claims must be JSON arrays of plain quoted strings. object_linkage must be exactly correct, incorrect, or uncertain. scores must contain exactly visual_evidence, conclusion_support, and confidence_appropriateness as JSON integers from 0 to 2. total must be the integer sum of those three scores. rationale must be a JSON array of plain quoted strings. Never place key:value fragments or nested objects inside a string array.'

infer_job="$(sbatch --parsable --time=00:30:00 --exclude=auh7-1b-gpu-197 \
  --export="ALL,INPUT_FILE=${strict_inputs},RUN_LABEL=${run_tag},MAX_NEW_TOKENS=2048" \
  execs/scpts/cv801_pixtral_evaluator_infer.sh)"
replacement_predictions="${project_root}/wkdrs/cv801-independent-eval/${run_tag}-${infer_job}/predictions.jsonl"
line_numbers_space="${line_numbers[*]}"
repair_job="$(sbatch --parsable --dependency="afterok:${infer_job}" \
  --export="ALL,STAGE=repair,OUTPUT_DIR=${output_dir},SOURCE_PREDICTIONS=${source_predictions},REPAIR_PREDICTIONS=${replacement_predictions},LINE_NUMBERS=${line_numbers_space}" \
  execs/scpts/cv801_independent_eval_stage.sh)"

{
  printf 'submitted_uae=%q\n' "$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'run_tag=%q\n' "${run_tag}"
  printf 'output_dir=%q\n' "${output_dir}"
  printf 'source_predictions=%q\n' "${source_predictions}"
  printf 'invalid_lines=%q\n' "${line_numbers_value}"
  printf 'infer_job=%q\n' "${infer_job}"
  printf 'replacement_predictions=%q\n' "${replacement_predictions}"
  printf 'repair_job=%q\n' "${repair_job}"
} > "${repair_root}/repair.env"

printf 'independent_eval_repair=SUBMITTED run=%s infer=%s repair=%s rows=%s\n' \
  "${run_tag}" "${infer_job}" "${repair_job}" "${#line_numbers[@]}"
