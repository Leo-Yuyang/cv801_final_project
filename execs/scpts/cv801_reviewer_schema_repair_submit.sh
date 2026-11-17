#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

run_tag="${RUN_TAG:?Set RUN_TAG.}"
role="${ROLE:?Set ROLE to rc or rd.}"
model_family="${MODEL_FAMILY:?Set MODEL_FAMILY to gemma3 or internvl.}"
model_path="${MODEL_PATH:-}"
model_gpus="${MODEL_GPUS:-1}"
source_input="${SOURCE_INPUT:?Set SOURCE_INPUT.}"
source_predictions="${SOURCE_PREDICTIONS:?Set SOURCE_PREDICTIONS.}"
promote_to="${PROMOTE_TO:-${source_predictions}}"
line_numbers_value="${LINE_NUMBERS:?Set colon-separated LINE_NUMBERS.}"
replacement_predictions="${REPLACEMENT_PREDICTIONS:-}"
secondary_source="${SECONDARY_SOURCE_PREDICTIONS:-}"
run_dir="${RUN_DIR:?Set RUN_DIR.}"

[[ "${run_tag}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || exit 1
[[ "${role}" == rc || "${role}" == rd ]] || exit 1
[[ "${model_family}" == gemma3 || "${model_family}" == internvl ]] || exit 1
[[ "${line_numbers_value}" =~ ^[0-9]+(:[0-9]+)*$ ]] || exit 1
for required in "${source_input}" "${source_predictions}" "${promote_to}"; do
  [[ -f "${required}" ]] || { echo "Missing repair input: ${required}" >&2; exit 1; }
done
if [[ -n "${secondary_source}" ]]; then
  [[ -f "${secondary_source}" ]] || exit 1
fi

python_bin="${project_root}/.venv-cv801/bin/python"
row_ops="${project_root}/CV801/scripts/jsonl_row_ops.py"
agent_stage="${project_root}/execs/scpts/cv801_agent_review_stage.sh"
repair_dir="${run_dir}/schema-repair-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "${repair_dir}"
IFS=: read -r -a line_numbers <<< "${line_numbers_value}"

infer_job=""
if [[ -z "${replacement_predictions}" ]]; then
  [[ -f "${model_path}/config.json" ]] || { echo "Missing model: ${model_path}" >&2; exit 1; }
  selected_input="${repair_dir}/selected-inputs.jsonl"
  strict_input="${repair_dir}/strict-inputs.jsonl"
  "${python_bin}" "${row_ops}" select-lines --source "${source_input}" \
    --output "${selected_input}" --line-numbers "${line_numbers[@]}"
  "${python_bin}" "${row_ops}" append-instruction --source "${selected_input}" \
    --output "${strict_input}" \
    --instruction 'Return exactly one strict JSON object and no prose. global_issues, revision_instructions, supported_claims, unsupported_claims, and missed_evidence must always be JSON arrays whose elements are plain quoted strings, never key:value fragments or nested objects. Every direct element of object_reviews must contain only the reviewer fields object_id, supported_claims, unsupported_claims, missed_evidence, and confidence_issue; never copy plant_part, objects, caption, assessment, image_quality, confidence, or action from the draft. confidence_issue must be exactly none, overconfident, or underconfident. If the writer draft is malformed or too repetitive to review safely, return object_reviews as an empty array, put "malformed writer draft" in global_issues, and request regeneration in revision_instructions.'
  mode="schema-repair-${run_tag}"
  if [[ "${model_family}" == gemma3 ]]; then
    reviewer_script="execs/scpts/cv801_gemma3_reviewer_infer.sh"
  else
    reviewer_script="execs/scpts/cv801_reviewer_infer.sh"
  fi
  infer_job="$(sbatch --parsable --exclude=auh7-1b-gpu-197 --time=04:00:00 \
    --gres="gpu:mi210:${model_gpus}" --cpus-per-task="$((16 * model_gpus))" \
    --mem="$((160 * model_gpus))G" \
    --export="ALL,MODEL_PATH=${model_path},INPUT_FILE=${strict_input},ROLE=${role},MODE=${mode},MAX_NEW_TOKENS=4096" \
    "${reviewer_script}")"
  if [[ "${model_family}" == gemma3 ]]; then
    replacement_predictions="${project_root}/wkdrs/cv801-agent-review/rc-gemma3-${mode}-${infer_job}/predictions.jsonl"
  else
    replacement_predictions="${project_root}/wkdrs/cv801-agent-review/${role}-${mode}-${infer_job}/predictions.jsonl"
  fi
else
  [[ -f "${replacement_predictions}" ]] || exit 1
fi

dependency_args=()
if [[ -n "${infer_job}" ]]; then dependency_args+=(--dependency="afterok:${infer_job}"); fi
primary_output="${repair_dir}/primary.repaired.jsonl"
primary_backup="${repair_dir}/primary.pre-repair.jsonl"
primary_job="$(sbatch --parsable "${dependency_args[@]}" \
  --export="ALL,STAGE=repair_reviewer,SOURCE_PREDICTIONS=${source_predictions},REPLACEMENT_PREDICTIONS=${replacement_predictions},OUTPUT=${primary_output},LINE_NUMBERS=${line_numbers_value},PROMOTE_TO=${promote_to},BACKUP_OUTPUT=${primary_backup}" \
  "${agent_stage}")"

secondary_job=""
if [[ -n "${secondary_source}" ]]; then
  secondary_output="${repair_dir}/secondary.repaired.jsonl"
  secondary_backup="${repair_dir}/secondary.pre-repair.jsonl"
  secondary_job="$(sbatch --parsable "${dependency_args[@]}" \
    --export="ALL,STAGE=repair_reviewer,SOURCE_PREDICTIONS=${secondary_source},REPLACEMENT_PREDICTIONS=${replacement_predictions},OUTPUT=${secondary_output},LINE_NUMBERS=${line_numbers_value},PROMOTE_TO=${secondary_source},BACKUP_OUTPUT=${secondary_backup}" \
    "${agent_stage}")"
fi

submitted_uae="$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
{
  printf 'submitted_uae=%q\n' "${submitted_uae}"
  printf 'run_tag=%q\n' "${run_tag}"
  printf 'invalid_lines=%q\n' "${line_numbers_value}"
  printf 'infer_job=%q\n' "${infer_job}"
  printf 'replacement_predictions=%q\n' "${replacement_predictions}"
  printf 'promote_to=%q\n' "${promote_to}"
  printf 'primary_job=%q\n' "${primary_job}"
  printf 'secondary_job=%q\n' "${secondary_job}"
} > "${repair_dir}/repair.env"

printf 'reviewer_schema_repair=SUBMITTED run=%s infer=%s primary=%s secondary=%s\n' \
  "${run_tag}" "${infer_job:-reused}" "${primary_job}" "${secondary_job:-none}"
