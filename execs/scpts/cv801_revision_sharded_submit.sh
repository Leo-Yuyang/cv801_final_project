#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

run_tag="${RUN_TAG:?Set RUN_TAG.}"
input_file="${INPUT_FILE:?Set INPUT_FILE.}"
model_path="${MODEL_PATH:?Set MODEL_PATH.}"
adapter_path="${ADAPTER_PATH:?Set ADAPTER_PATH.}"
upstream_dependency="${UPSTREAM_DEPENDENCY:-}"
shards="${SHARDS:-8}"
run_dir="${RUN_DIR:?Set RUN_DIR.}"

[[ "${run_tag}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || exit 1
[[ "${shards}" =~ ^[1-9][0-9]*$ ]] || exit 1
if [[ -n "${upstream_dependency}" ]]; then
  [[ "${upstream_dependency}" =~ ^[0-9]+(:[0-9]+)*$ ]] || exit 1
fi
[[ -f "${model_path}/config.json" && -f "${adapter_path}/adapter_model.safetensors" ]] || exit 1
[[ -f "${input_file}" || -n "${upstream_dependency}" ]] || { echo "Input is missing without dependency: ${input_file}" >&2; exit 1; }
[[ ! -e "${run_dir}" ]] || { echo "Refusing to reuse ${run_dir}." >&2; exit 1; }
mkdir -p "${run_dir}"

agent_stage="execs/scpts/cv801_agent_review_stage.sh"
revision_script="execs/scpts/cv801_qwen35_revision_infer.sh"
dependency_args=()
if [[ -n "${upstream_dependency}" ]]; then
  dependency_args+=(--dependency="afterok:${upstream_dependency}")
fi
shard_dir="${run_dir}/revision-shards"
split_job="$(sbatch --parsable "${dependency_args[@]}" \
  --export="ALL,STAGE=split_jsonl,SOURCE_INPUT=${input_file},OUTPUT_DIR=${shard_dir},SHARDS=${shards}" \
  "${agent_stage}")"

revision_jobs=(); revision_outputs=()
for ((rank=0; rank<shards; rank++)); do
  printf -v shard '%02d' "${rank}"
  mode="${run_tag}-s${shard}"
  job_id="$(sbatch --parsable --time=16:00:00 --dependency="afterok:${split_job}" \
    --export="ALL,MODEL_PATH=${model_path},ADAPTER_PATH=${adapter_path},INPUT_FILE=${shard_dir}/input-${shard}.jsonl,CONFIGURATION=full,MODE=${mode},MAX_NEW_TOKENS=4096" \
    "${revision_script}")"
  revision_jobs+=("${job_id}")
  revision_outputs+=("${project_root}/wkdrs/cv801-agent-review/revision-full-${mode}-${job_id}/predictions.jsonl")
done
join_colon() { local IFS=:; echo "$*"; }
merged_predictions="${run_dir}/predictions.jsonl"
merge_job="$(sbatch --parsable --dependency="afterok:$(join_colon "${revision_jobs[@]}")" \
  --export="ALL,STAGE=merge_predictions,REFERENCE_INPUT=${input_file},SOURCE_PREDICTIONS=$(join_colon "${revision_outputs[@]}"),OUTPUT=${merged_predictions}" \
  "${agent_stage}")"

RUN_TAG="${run_tag}" \
REVISION_PREDICTIONS="${merged_predictions}" \
RC_PREDICTIONS="${RC_PREDICTIONS:?Set RC_PREDICTIONS.}" \
RD_PREDICTIONS="${RD_PREDICTIONS:?Set RD_PREDICTIONS.}" \
JUDGE_PREDICTIONS="${JUDGE_PREDICTIONS:?Set JUDGE_PREDICTIONS.}" \
MANIFEST="${MANIFEST:?Set MANIFEST.}" \
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}" \
UPSTREAM_DEPENDENCY="${merge_job}" \
RUN_DIR="${run_dir}/downstream" \
RECHECK_SHARDS="${RECHECK_SHARDS:-8}" \
EVAL_SHARDS="${EVAL_SHARDS:-8}" \
bash execs/scpts/cv801_agent_review_after_revision_sharded_submit.sh

{
  printf 'submitted_uae=%q\n' "$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'run_tag=%q\n' "${run_tag}"
  printf 'input_file=%q\n' "${input_file}"
  printf 'upstream_dependency=%q\n' "${upstream_dependency}"
  printf 'shards=%q\n' "${shards}"
  printf 'split_job=%q\n' "${split_job}"
  printf 'revision_jobs=%q\n' "$(join_colon "${revision_jobs[@]}")"
  printf 'revision_outputs=%q\n' "$(join_colon "${revision_outputs[@]}")"
  printf 'merge_job=%q\n' "${merge_job}"
  printf 'merged_predictions=%q\n' "${merged_predictions}"
} > "${run_dir}/sharded.env"

printf 'revision_sharded=SUBMITTED run=%s shards=%s jobs=%s merge=%s\n' \
  "${run_tag}" "${shards}" "$(join_colon "${revision_jobs[@]}")" "${merge_job}"
