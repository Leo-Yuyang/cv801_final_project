#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

run_tag="${RUN_TAG:?Set RUN_TAG.}"
writer_predictions="${WRITER_PREDICTIONS:?Set WRITER_PREDICTIONS.}"
rc_predictions="${RC_PREDICTIONS:?Set RC_PREDICTIONS.}"
rd_predictions="${RD_PREDICTIONS:?Set RD_PREDICTIONS.}"
upstream_dependency="${UPSTREAM_DEPENDENCY:-}"
writer_model_path="${WRITER_MODEL_PATH:-${project_root}/inits/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
writer_adapter_path="${WRITER_ADAPTER_PATH:-${project_root}/wkdrs/cv801-qwen35/t2-train-382484/v0-20261109-131602/checkpoint-1197}"
judge_model_path="${JUDGE_MODEL_PATH:-${project_root}/inits/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"

[[ "${run_tag}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || {
  echo "RUN_TAG must contain lowercase letters, digits, and hyphens." >&2
  exit 1
}
if [[ -n "${upstream_dependency}" ]]; then
  [[ "${upstream_dependency}" =~ ^[0-9]+(:[0-9]+)*$ ]] || {
    echo "UPSTREAM_DEPENDENCY must be colon-separated Slurm job IDs." >&2
    exit 1
  }
fi

review_dir="${project_root}/wkdrs/cv801-golden-test/reviewer-inputs-403233"
manifest="${MANIFEST:-${review_dir}/review_manifest.jsonl}"
reference_predictions="${REFERENCE_PREDICTIONS:-${project_root}/wkdrs/cv801-golden-test/vlm-references-final.jsonl}"
run_dir="${RUN_DIR:-${project_root}/wkdrs/cv801-agent-review-full/${run_tag}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

agent_stage="execs/scpts/cv801_agent_review_stage.sh"
judge_script="execs/scpts/cv801_qwen35_judge_infer.sh"
revision_script="execs/scpts/cv801_qwen35_revision_infer.sh"
independent_stage="execs/scpts/cv801_independent_eval_stage.sh"
evaluator_script="execs/scpts/cv801_pixtral_evaluator_infer.sh"

for required in "${reference_predictions}" "${writer_model_path}/config.json" \
  "${judge_model_path}/config.json" \
  "${writer_adapter_path}/adapter_model.safetensors"; do
  [[ -f "${required}" ]] || { echo "Required frozen input is missing: ${required}" >&2; exit 1; }
done
for future_input in "${writer_predictions}" "${manifest}" "${rc_predictions}" "${rd_predictions}"; do
  if [[ ! -f "${future_input}" && -z "${upstream_dependency}" ]]; then
    echo "Input is missing without an upstream dependency: ${future_input}" >&2
    exit 1
  fi
done
[[ ! -e "${run_dir}" ]] || {
  if [[ "${ALLOW_EXISTING_RUN_DIR:-NO}" != "YES" ]]; then
    echo "Refusing to reuse ${run_dir}." >&2
    exit 1
  fi
}
mkdir -p "${run_dir}"

dependency_args=()
if [[ -n "${upstream_dependency}" ]]; then
  dependency_args+=(--dependency="afterok:${upstream_dependency}")
fi

judge_input="${run_dir}/judge_inputs.jsonl"
judge_prepare_job="$(sbatch --parsable "${dependency_args[@]}" \
  --export="ALL,STAGE=judge,MANIFEST=${manifest},RC_PREDICTIONS=${rc_predictions},RD_PREDICTIONS=${rd_predictions},OUTPUT=${judge_input}" \
  "${agent_stage}")"
judge_job="$(sbatch --parsable --dependency="afterok:${judge_prepare_job}" \
  --export="ALL,MODEL_PATH=${judge_model_path},INPUT_FILE=${judge_input},STAGE=judge,MODE=${run_tag},MAX_NEW_TOKENS=4096" \
  "${judge_script}")"
judge_predictions="${project_root}/wkdrs/cv801-agent-review/judge-${run_tag}-${judge_job}/predictions.jsonl"
judge_audit="${run_dir}/judge.audit.json"
judge_audit_job="$(sbatch --parsable --dependency="afterok:${judge_job}" \
  --export="ALL,STAGE=audit,PREDICTIONS=${judge_predictions},AUDIT_KIND=judge,AUDIT_OUTPUT=${judge_audit}" \
  "${agent_stage}")"

revision_input="${run_dir}/revision-full-inputs.jsonl"
revision_prepare_job="$(sbatch --parsable --dependency="afterok:${judge_audit_job}" \
  --export="ALL,STAGE=revision,WRITER_PREDICTIONS=${writer_predictions},FEEDBACK_SOURCE=judge,JUDGE_PREDICTIONS=${judge_predictions},OUTPUT=${revision_input}" \
  "${agent_stage}")"
revision_job="$(sbatch --parsable --dependency="afterok:${revision_prepare_job}" \
  --export="ALL,MODEL_PATH=${writer_model_path},ADAPTER_PATH=${writer_adapter_path},INPUT_FILE=${revision_input},CONFIGURATION=full,MODE=${run_tag},MAX_NEW_TOKENS=4096" \
  "${revision_script}")"
revision_dir="${project_root}/wkdrs/cv801-agent-review/revision-full-${run_tag}-${revision_job}"
revision_predictions="${revision_dir}/predictions.jsonl"
structured_eval_job="$(sbatch --parsable --dependency="afterok:${revision_job}" \
  --export="ALL,STAGE=evaluate,PREDICTIONS=${revision_predictions},REFERENCE_PREDICTIONS=${reference_predictions},LABELED_OUTPUT=${revision_dir}/metrics.golden.labeled.jsonl,METRICS_OUTPUT=${revision_dir}/metrics.golden.json" \
  "${agent_stage}")"

recheck_input="${run_dir}/recheck_inputs.jsonl"
recheck_prepare_job="$(sbatch --parsable --dependency="afterok:${revision_job}" \
  --export="ALL,STAGE=recheck,MANIFEST=${manifest},RC_PREDICTIONS=${rc_predictions},RD_PREDICTIONS=${rd_predictions},JUDGE_PREDICTIONS=${judge_predictions},REVISED_PREDICTIONS=${revision_predictions},OUTPUT=${recheck_input}" \
  "${agent_stage}")"
recheck_job="$(sbatch --parsable --dependency="afterok:${recheck_prepare_job}" \
  --export="ALL,MODEL_PATH=${judge_model_path},INPUT_FILE=${recheck_input},STAGE=recheck,MODE=${run_tag},MAX_NEW_TOKENS=4096" \
  "${judge_script}")"
recheck_predictions="${project_root}/wkdrs/cv801-agent-review/recheck-${run_tag}-${recheck_job}/predictions.jsonl"
recheck_audit="${run_dir}/recheck.audit.json"
recheck_audit_job="$(sbatch --parsable --dependency="afterok:${recheck_job}" \
  --export="ALL,STAGE=audit,PREDICTIONS=${recheck_predictions},AUDIT_KIND=recheck,AUDIT_OUTPUT=${recheck_audit}" \
  "${agent_stage}")"

independent_dir="${project_root}/wkdrs/cv801-independent-eval/${run_tag}"
independent_prepare_job="$(sbatch --parsable --dependency="afterok:${revision_job}" \
  --export="ALL,STAGE=prepare,PREDICTIONS=${revision_predictions},OUTPUT_DIR=${independent_dir},SHARDS=4" \
  "${independent_stage}")"
evaluator_jobs=()
evaluator_outputs=()
for rank in 0 1 2 3; do
  printf -v shard '%02d' "${rank}"
  run_label="${run_tag}-eval-s${shard}"
  evaluator_job="$(sbatch --parsable --dependency="afterok:${independent_prepare_job}" \
    --export="ALL,INPUT_FILE=${independent_dir}/shards/input-${shard}.jsonl,RUN_LABEL=${run_label},MAX_NEW_TOKENS=2048" \
    "${evaluator_script}")"
  evaluator_jobs+=("${evaluator_job}")
  evaluator_outputs+=("${project_root}/wkdrs/cv801-independent-eval/${run_label}-${evaluator_job}/predictions.jsonl")
done
join_colon() { local IFS=:; echo "$*"; }
evaluator_dependency="$(join_colon "${evaluator_jobs[@]}")"
evaluator_sources="$(join_colon "${evaluator_outputs[@]}")"
independent_merge_job="$(sbatch --parsable --dependency="afterok:${evaluator_dependency}" \
  --export="ALL,STAGE=merge,OUTPUT_DIR=${independent_dir},SOURCE_PREDICTIONS=${evaluator_sources}" \
  "${independent_stage}")"

submitted_uae="$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
{
  printf 'submitted_uae=%q\n' "${submitted_uae}"
  printf 'run_tag=%q\n' "${run_tag}"
  printf 'writer_predictions=%q\n' "${writer_predictions}"
  printf 'writer_model_path=%q\n' "${writer_model_path}"
  printf 'writer_adapter_path=%q\n' "${writer_adapter_path}"
  printf 'judge_model_path=%q\n' "${judge_model_path}"
  printf 'rc_predictions=%q\n' "${rc_predictions}"
  printf 'rd_predictions=%q\n' "${rd_predictions}"
  printf 'upstream_dependency=%q\n' "${upstream_dependency}"
  printf 'judge_prepare_job=%q\n' "${judge_prepare_job}"
  printf 'judge_job=%q\n' "${judge_job}"
  printf 'judge_audit_job=%q\n' "${judge_audit_job}"
  printf 'revision_prepare_job=%q\n' "${revision_prepare_job}"
  printf 'revision_job=%q\n' "${revision_job}"
  printf 'structured_eval_job=%q\n' "${structured_eval_job}"
  printf 'recheck_prepare_job=%q\n' "${recheck_prepare_job}"
  printf 'recheck_job=%q\n' "${recheck_job}"
  printf 'recheck_audit_job=%q\n' "${recheck_audit_job}"
  printf 'independent_prepare_job=%q\n' "${independent_prepare_job}"
  printf 'evaluator_jobs=%q\n' "$(join_colon "${evaluator_jobs[@]}")"
  printf 'independent_merge_job=%q\n' "${independent_merge_job}"
} > "${run_dir}/downstream.env"

printf 'agent_review_full_status=SUBMITTED run_tag=%s run_dir=%s\n' "${run_tag}" "${run_dir}"
printf 'terminal_jobs structured_eval=%s recheck_audit=%s independent_eval=%s\n' \
  "${structured_eval_job}" "${recheck_audit_job}" "${independent_merge_job}"
