#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

run_tag="${RUN_TAG:?Set RUN_TAG.}"
revision_predictions="${REVISION_PREDICTIONS:?Set REVISION_PREDICTIONS.}"
rc_predictions="${RC_PREDICTIONS:?Set RC_PREDICTIONS.}"
rd_predictions="${RD_PREDICTIONS:?Set RD_PREDICTIONS.}"
judge_predictions="${JUDGE_PREDICTIONS:?Set JUDGE_PREDICTIONS.}"
upstream_dependency="${UPSTREAM_DEPENDENCY:?Set UPSTREAM_DEPENDENCY.}"
manifest="${MANIFEST:?Set MANIFEST.}"
judge_model_path="${JUDGE_MODEL_PATH:-${project_root}/inits/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
reference_predictions="${REFERENCE_PREDICTIONS:-${project_root}/wkdrs/cv801-golden-test/vlm-references-final.jsonl}"
run_dir="${RUN_DIR:?Set RUN_DIR.}"
recheck_shards="${RECHECK_SHARDS:-8}"
eval_shards="${EVAL_SHARDS:-8}"

[[ "${run_tag}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || exit 1
[[ "${upstream_dependency}" =~ ^[0-9]+(:[0-9]+)*$ ]] || exit 1
[[ "${recheck_shards}" =~ ^[1-9][0-9]*$ && "${eval_shards}" =~ ^[1-9][0-9]*$ ]] || exit 1
for required in "${rc_predictions}" "${rd_predictions}" "${manifest}" \
  "${reference_predictions}" "${judge_model_path}/config.json"; do
  [[ -f "${required}" ]] || { echo "Missing downstream input: ${required}" >&2; exit 1; }
done
[[ -f "${judge_predictions}" || -n "${upstream_dependency}" ]] || exit 1
[[ -f "${revision_predictions}" || -n "${upstream_dependency}" ]] || exit 1
[[ ! -e "${run_dir}" ]] || { echo "Refusing to reuse ${run_dir}." >&2; exit 1; }
mkdir -p "${run_dir}"

agent_stage="execs/scpts/cv801_agent_review_stage.sh"
judge_script="execs/scpts/cv801_qwen35_judge_infer.sh"
independent_stage="execs/scpts/cv801_independent_eval_stage.sh"
evaluator_script="execs/scpts/cv801_pixtral_evaluator_infer.sh"

structured_eval_job="$(sbatch --parsable --dependency="afterok:${upstream_dependency}" \
  --export="ALL,STAGE=evaluate,PREDICTIONS=${revision_predictions},REFERENCE_PREDICTIONS=${reference_predictions},LABELED_OUTPUT=${run_dir}/metrics.golden.labeled.jsonl,METRICS_OUTPUT=${run_dir}/metrics.golden.json" \
  "${agent_stage}")"

recheck_input="${run_dir}/recheck_inputs.jsonl"
recheck_prepare_job="$(sbatch --parsable --dependency="afterok:${upstream_dependency}" \
  --export="ALL,STAGE=recheck,MANIFEST=${manifest},RC_PREDICTIONS=${rc_predictions},RD_PREDICTIONS=${rd_predictions},JUDGE_PREDICTIONS=${judge_predictions},REVISED_PREDICTIONS=${revision_predictions},OUTPUT=${recheck_input}" \
  "${agent_stage}")"
recheck_shard_dir="${run_dir}/recheck-shards"
recheck_split_job="$(sbatch --parsable --dependency="afterok:${recheck_prepare_job}" \
  --export="ALL,STAGE=split_jsonl,SOURCE_INPUT=${recheck_input},OUTPUT_DIR=${recheck_shard_dir},SHARDS=${recheck_shards}" \
  "${agent_stage}")"
recheck_jobs=(); recheck_outputs=()
for ((rank=0; rank<recheck_shards; rank++)); do
  printf -v shard '%02d' "${rank}"
  mode="${run_tag}-recheck-s${shard}"
  job_id="$(sbatch --parsable --dependency="afterok:${recheck_split_job}" \
    --export="ALL,MODEL_PATH=${judge_model_path},INPUT_FILE=${recheck_shard_dir}/input-${shard}.jsonl,STAGE=recheck,MODE=${mode},MAX_NEW_TOKENS=4096" \
    "${judge_script}")"
  recheck_jobs+=("${job_id}")
  recheck_outputs+=("${project_root}/wkdrs/cv801-agent-review/recheck-${mode}-${job_id}/predictions.jsonl")
done
join_colon() { local IFS=:; echo "$*"; }
recheck_raw_predictions="${run_dir}/recheck_predictions.raw.jsonl"
recheck_merge_job="$(sbatch --parsable --dependency="afterok:$(join_colon "${recheck_jobs[@]}")" \
  --export="ALL,STAGE=merge_predictions,REFERENCE_INPUT=${recheck_input},SOURCE_PREDICTIONS=$(join_colon "${recheck_outputs[@]}"),OUTPUT=${recheck_raw_predictions}" \
  "${agent_stage}")"
recheck_predictions="${run_dir}/recheck_predictions.jsonl"
recheck_audit_job="$(sbatch --parsable --dependency="afterok:${recheck_merge_job}" \
  --export="ALL,STAGE=recover_decisions,PREDICTIONS=${recheck_raw_predictions},OUTPUT=${recheck_predictions},AUDIT_KIND=recheck,AUDIT_OUTPUT=${run_dir}/recheck.audit.json" \
  "${agent_stage}")"

independent_dir="${project_root}/wkdrs/cv801-independent-eval/${run_tag}"
independent_prepare_job="$(sbatch --parsable --dependency="afterok:${upstream_dependency}" \
  --export="ALL,STAGE=prepare,PREDICTIONS=${revision_predictions},OUTPUT_DIR=${independent_dir},SHARDS=${eval_shards}" \
  "${independent_stage}")"
evaluator_jobs=(); evaluator_outputs=()
for ((rank=0; rank<eval_shards; rank++)); do
  printf -v shard '%02d' "${rank}"
  run_label="${run_tag}-eval-s${shard}"
  job_id="$(sbatch --parsable --dependency="afterok:${independent_prepare_job}" \
    --export="ALL,INPUT_FILE=${independent_dir}/shards/input-${shard}.jsonl,RUN_LABEL=${run_label},MAX_NEW_TOKENS=2048" \
    "${evaluator_script}")"
  evaluator_jobs+=("${job_id}")
  evaluator_outputs+=("${project_root}/wkdrs/cv801-independent-eval/${run_label}-${job_id}/predictions.jsonl")
done
independent_merge_job="$(sbatch --parsable --dependency="afterok:$(join_colon "${evaluator_jobs[@]}")" \
  --export="ALL,STAGE=merge,OUTPUT_DIR=${independent_dir},SOURCE_PREDICTIONS=$(join_colon "${evaluator_outputs[@]}"),REQUIRE_ALL_VALID=NO" \
  "${independent_stage}")"

{
  printf 'submitted_uae=%q\n' "$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'run_tag=%q\n' "${run_tag}"
  printf 'revision_predictions=%q\n' "${revision_predictions}"
  printf 'upstream_dependency=%q\n' "${upstream_dependency}"
  printf 'structured_eval_job=%q\n' "${structured_eval_job}"
  printf 'recheck_prepare_job=%q\n' "${recheck_prepare_job}"
  printf 'recheck_split_job=%q\n' "${recheck_split_job}"
  printf 'recheck_jobs=%q\n' "$(join_colon "${recheck_jobs[@]}")"
  printf 'recheck_merge_job=%q\n' "${recheck_merge_job}"
  printf 'recheck_raw_predictions=%q\n' "${recheck_raw_predictions}"
  printf 'recheck_predictions=%q\n' "${recheck_predictions}"
  printf 'recheck_audit_job=%q\n' "${recheck_audit_job}"
  printf 'independent_prepare_job=%q\n' "${independent_prepare_job}"
  printf 'evaluator_jobs=%q\n' "$(join_colon "${evaluator_jobs[@]}")"
  printf 'independent_merge_job=%q\n' "${independent_merge_job}"
} > "${run_dir}/downstream.env"

printf 'agent_review_after_revision_sharded=SUBMITTED run=%s recheck_shards=%s eval_shards=%s terminal=%s:%s:%s\n' \
  "${run_tag}" "${recheck_shards}" "${eval_shards}" \
  "${structured_eval_job}" "${recheck_audit_job}" "${independent_merge_job}"
