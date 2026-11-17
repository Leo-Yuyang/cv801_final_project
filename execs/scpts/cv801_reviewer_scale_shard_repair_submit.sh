#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

run_tag="${RUN_TAG:?Set RUN_TAG.}"
variable_role="${VARIABLE_ROLE:?Set VARIABLE_ROLE to rc or rd.}"
model_family="${MODEL_FAMILY:?Set MODEL_FAMILY to gemma3 or internvl.}"
model_path="${MODEL_PATH:?Set MODEL_PATH.}"
model_gpus="${MODEL_GPUS:-1}"
failed_input="${FAILED_INPUT:?Set FAILED_INPUT to the 250-row shard.}"
reference_input="${REFERENCE_INPUT:?Set REFERENCE_INPUT.}"
prefix_outputs="${PREFIX_OUTPUTS:-}"
suffix_outputs="${SUFFIX_OUTPUTS:-}"
merged_output="${MERGED_OUTPUT:?Set MERGED_OUTPUT.}"
fixed_rc_predictions="${FIXED_RC_PREDICTIONS:-}"
fixed_rd_predictions="${FIXED_RD_PREDICTIONS:-}"
run_dir="${RUN_DIR:?Set RUN_DIR to the existing scale run directory.}"

[[ "${variable_role}" == "rc" || "${variable_role}" == "rd" ]] || exit 1
[[ "${model_family}" == "gemma3" || "${model_family}" == "internvl" ]] || exit 1
[[ -f "${failed_input}" && -f "${reference_input}" && -f "${model_path}/config.json" ]] || {
  echo "Repair input, reference, or model is missing." >&2
  exit 1
}
test "$(wc -l < "${failed_input}")" -eq 250
[[ ! -e "${merged_output}" ]] || { echo "Refusing to overwrite ${merged_output}." >&2; exit 1; }

if [[ "${model_family}" == "gemma3" ]]; then
  reviewer_script="execs/scpts/cv801_gemma3_reviewer_infer.sh"
else
  reviewer_script="execs/scpts/cv801_reviewer_infer.sh"
fi
agent_stage="execs/scpts/cv801_agent_review_stage.sh"

repair_dir="${run_dir}/repairs/shard-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "${repair_dir}/inputs"
split -d -a 2 -l 50 --additional-suffix=.jsonl \
  "${failed_input}" "${repair_dir}/inputs/input-"
test "$(find "${repair_dir}/inputs" -maxdepth 1 -type f -name 'input-*.jsonl' | wc -l)" -eq 5

prediction_path() {
  local mode="$1"
  local job_id="$2"
  if [[ "${model_family}" == "gemma3" ]]; then
    printf '%s/wkdrs/cv801-agent-review/rc-gemma3-%s-%s/predictions.jsonl' \
      "${project_root}" "${mode}" "${job_id}"
  else
    printf '%s/wkdrs/cv801-agent-review/%s-%s-%s/predictions.jsonl' \
      "${project_root}" "${variable_role}" "${mode}" "${job_id}"
  fi
}

repair_jobs=()
repair_outputs=()
for part in 0 1 2 3 4; do
  printf -v part_id '%02d' "${part}"
  input_file="${repair_dir}/inputs/input-${part_id}.jsonl"
  mode="repair-${run_tag}-p${part_id}"
  job_id="$(sbatch --parsable --exclude=auh7-1b-gpu-197 --time=12:00:00 \
    --gres="gpu:mi210:${model_gpus}" \
    --cpus-per-task="$((16 * model_gpus))" --mem="$((128 * model_gpus))G" \
    --export="ALL,MODEL_PATH=${model_path},INPUT_FILE=${input_file},ROLE=${variable_role},MODE=${mode},MAX_NEW_TOKENS=4096" \
    "${reviewer_script}")"
  repair_jobs+=("${job_id}")
  repair_outputs+=("$(prediction_path "${mode}" "${job_id}")")
done

sources=()
if [[ -n "${prefix_outputs}" ]]; then
  IFS=: read -r -a prefix_array <<< "${prefix_outputs}"
  sources+=("${prefix_array[@]}")
fi
sources+=("${repair_outputs[@]}")
if [[ -n "${suffix_outputs}" ]]; then
  IFS=: read -r -a suffix_array <<< "${suffix_outputs}"
  sources+=("${suffix_array[@]}")
fi
join_colon() { local IFS=:; echo "$*"; }
repair_dependency="$(join_colon "${repair_jobs[@]}")"
source_list="$(join_colon "${sources[@]}")"
merge_job="$(sbatch --parsable --dependency="afterok:${repair_dependency}" \
  --export="ALL,STAGE=merge_reviewer,OUTPUT=${merged_output},REFERENCE_INPUT=${reference_input},SOURCE_PREDICTIONS=${source_list}" \
  "${agent_stage}")"

if [[ "${variable_role}" == "rc" ]]; then
  rc_predictions="${merged_output}"
  rd_predictions="${fixed_rd_predictions:?Set FIXED_RD_PREDICTIONS for Rc repair.}"
else
  rc_predictions="${fixed_rc_predictions:?Set FIXED_RC_PREDICTIONS for Rd repair.}"
  rd_predictions="${merged_output}"
fi

RUN_TAG="${run_tag}-repaired" \
WRITER_PREDICTIONS="${project_root}/wkdrs/cv801-golden-test/writer-t2-403232/predictions.jsonl" \
RC_PREDICTIONS="${rc_predictions}" \
RD_PREDICTIONS="${rd_predictions}" \
UPSTREAM_DEPENDENCY="${merge_job}" \
RUN_DIR="${run_dir}/downstream-repaired" \
execs/scpts/cv801_agent_review_full_submit.sh

submitted_uae="$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
{
  printf 'repair_submitted_uae=%q\n' "${submitted_uae}"
  printf 'repair_jobs=%q\n' "${repair_dependency}"
  printf 'repair_outputs=%q\n' "$(join_colon "${repair_outputs[@]}")"
  printf 'repair_merge_job=%q\n' "${merge_job}"
  printf 'repair_merged_output=%q\n' "${merged_output}"
} > "${repair_dir}/repair.env"

printf 'reviewer_scale_shard_repair=SUBMITTED run_tag=%s repair_jobs=%s merge_job=%s\n' \
  "${run_tag}" "${repair_dependency}" "${merge_job}"
