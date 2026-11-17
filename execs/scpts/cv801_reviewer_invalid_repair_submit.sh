#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

run_tag="${RUN_TAG:?Set RUN_TAG.}"
model_path="${MODEL_PATH:?Set MODEL_PATH.}"
model_gpus="${MODEL_GPUS:-1}"
source_input="${SOURCE_INPUT:?Set SOURCE_INPUT.}"
source_predictions="${SOURCE_PREDICTIONS:?Set SOURCE_PREDICTIONS.}"
line_numbers_value="${LINE_NUMBERS:?Set colon-separated LINE_NUMBERS.}"
fixed_rd_predictions="${FIXED_RD_PREDICTIONS:?Set FIXED_RD_PREDICTIONS.}"
run_dir="${RUN_DIR:?Set RUN_DIR.}"

for required in "${model_path}/config.json" "${source_input}" "${source_predictions}" \
  "${fixed_rd_predictions}"; do
  [[ -f "${required}" ]] || { echo "Required input is missing: ${required}" >&2; exit 1; }
done
[[ "${run_tag}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || exit 1
[[ "${line_numbers_value}" =~ ^[0-9]+(:[0-9]+)*$ ]] || exit 1

row_ops="${project_root}/CV801/scripts/jsonl_row_ops.py"
python_bin="${project_root}/.venv-cv801/bin/python"
agent_stage="${project_root}/execs/scpts/cv801_agent_review_stage.sh"
reviewer_script="${project_root}/execs/scpts/cv801_gemma3_reviewer_infer.sh"
repair_dir="${run_dir}/repairs/invalid-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "${repair_dir}"

IFS=: read -r -a line_numbers <<< "${line_numbers_value}"
selected_input="${repair_dir}/selected-inputs.jsonl"
strict_input="${repair_dir}/strict-inputs.jsonl"
"${python_bin}" "${row_ops}" select-lines \
  --source "${source_input}" --output "${selected_input}" --line-numbers "${line_numbers[@]}"
"${python_bin}" "${row_ops}" append-instruction \
  --source "${selected_input}" --output "${strict_input}" \
  --instruction 'Return exactly one JSON object. Both "global_issues" and "revision_instructions" must be JSON arrays of strings, never a single string. Every object review must contain object_id, supported_claims, unsupported_claims, missed_evidence, and confidence_issue; the three claim fields must be JSON arrays of strings and confidence_issue must be one of none, overconfident, or underconfident.'
test "$(wc -l < "${strict_input}")" -eq "${#line_numbers[@]}"

mode="repair-${run_tag}-invalid"
infer_job="$(sbatch --parsable --exclude=auh7-1b-gpu-197 --time=04:00:00 \
  --gres="gpu:mi210:${model_gpus}" --cpus-per-task="$((16 * model_gpus))" \
  --mem="$((128 * model_gpus))G" \
  --export="ALL,MODEL_PATH=${model_path},INPUT_FILE=${strict_input},MODE=${mode},MAX_NEW_TOKENS=4096" \
  "${reviewer_script}")"
replacement_predictions="${project_root}/wkdrs/cv801-agent-review/rc-gemma3-${mode}-${infer_job}/predictions.jsonl"
repaired_predictions="${repair_dir}/rc_predictions.repaired.jsonl"
backup_predictions="${repair_dir}/rc_predictions.pre-repair.jsonl"
promote_job="$(sbatch --parsable --dependency="afterok:${infer_job}" \
  --export="ALL,STAGE=repair_reviewer,SOURCE_PREDICTIONS=${source_predictions},REPLACEMENT_PREDICTIONS=${replacement_predictions},OUTPUT=${repaired_predictions},LINE_NUMBERS=${line_numbers_value},PROMOTE_TO=${source_predictions},BACKUP_OUTPUT=${backup_predictions}" \
  "${agent_stage}")"

RUN_TAG="${run_tag}-repaired" \
WRITER_PREDICTIONS="${project_root}/wkdrs/cv801-golden-test/writer-t2-403232/predictions.jsonl" \
RC_PREDICTIONS="${source_predictions}" \
RD_PREDICTIONS="${fixed_rd_predictions}" \
UPSTREAM_DEPENDENCY="${promote_job}" \
RUN_DIR="${run_dir}/downstream-repaired" \
execs/scpts/cv801_agent_review_full_submit.sh

submitted_uae="$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
{
  printf 'submitted_uae=%q\n' "${submitted_uae}"
  printf 'invalid_lines=%q\n' "${line_numbers_value}"
  printf 'strict_input=%q\n' "${strict_input}"
  printf 'infer_job=%q\n' "${infer_job}"
  printf 'replacement_predictions=%q\n' "${replacement_predictions}"
  printf 'promote_job=%q\n' "${promote_job}"
  printf 'promoted_predictions=%q\n' "${source_predictions}"
  printf 'backup_predictions=%q\n' "${backup_predictions}"
} > "${repair_dir}/repair.env"

printf 'reviewer_invalid_repair=SUBMITTED run_tag=%s infer_job=%s promote_job=%s\n' \
  "${run_tag}" "${infer_job}" "${promote_job}"
