#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

run_tag="${RUN_TAG:?Set RUN_TAG, for example gemma4-internvl38.}"
variable_role="${VARIABLE_ROLE:?Set VARIABLE_ROLE to rc or rd.}"
model_family="${MODEL_FAMILY:?Set MODEL_FAMILY to gemma3 or internvl.}"
model_path="${MODEL_PATH:?Set MODEL_PATH to an immutable local model snapshot.}"
model_gpus="${MODEL_GPUS:-1}"
model_cpus="${MODEL_CPUS:-16}"
model_mem="${MODEL_MEM:-160G}"

[[ "${run_tag}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || {
  echo "RUN_TAG must contain only lowercase letters, digits, and hyphens." >&2
  exit 1
}
[[ "${variable_role}" == "rc" || "${variable_role}" == "rd" ]] || {
  echo "VARIABLE_ROLE must be rc or rd." >&2
  exit 1
}
[[ "${model_family}" == "gemma3" || "${model_family}" == "internvl" ]] || {
  echo "MODEL_FAMILY must be gemma3 or internvl." >&2
  exit 1
}
if [[ "${model_family}" == "gemma3" && "${variable_role}" != "rc" ]]; then
  echo "Gemma 3 is only supported as Rc in this controlled sweep." >&2
  exit 1
fi
[[ "${model_gpus}" =~ ^[12]$ ]] || { echo "MODEL_GPUS must be 1 or 2." >&2; exit 1; }

review_dir="${project_root}/wkdrs/cv801-golden-test/reviewer-inputs-403233"
writer_predictions="${project_root}/wkdrs/cv801-golden-test/writer-t2-403232/predictions.jsonl"
reference_predictions="${project_root}/wkdrs/cv801-golden-test/vlm-references-final.jsonl"
manifest="${review_dir}/review_manifest.jsonl"
fixed_rc_predictions="${FIXED_RC_PREDICTIONS:-${project_root}/wkdrs/cv801-golden-test/gemma3-reviewer-swap-422455/rc_predictions.jsonl}"
fixed_rd_predictions="${FIXED_RD_PREDICTIONS:-${review_dir}/rd_predictions.jsonl}"

agent_stage="execs/scpts/cv801_agent_review_stage.sh"
judge_script="execs/scpts/cv801_qwen35_judge_infer.sh"
revision_script="execs/scpts/cv801_qwen35_revision_infer.sh"
independent_stage="execs/scpts/cv801_independent_eval_stage.sh"
evaluator_script="execs/scpts/cv801_pixtral_evaluator_infer.sh"

if [[ "${model_family}" == "gemma3" ]]; then
  reviewer_script="execs/scpts/cv801_gemma3_reviewer_infer.sh"
else
  reviewer_script="execs/scpts/cv801_reviewer_infer.sh"
fi

for required in "${writer_predictions}" "${reference_predictions}" "${manifest}" \
  "${fixed_rc_predictions}" "${fixed_rd_predictions}" "${model_path}/config.json"; do
  [[ -f "${required}" ]] || { echo "Required frozen input is missing: ${required}" >&2; exit 1; }
done
test "$(wc -l < "${writer_predictions}")" -eq 1000
test "$(wc -l < "${fixed_rc_predictions}")" -eq 1000
test "$(wc -l < "${fixed_rd_predictions}")" -eq 1000
test "$(wc -l < "${manifest}")" -eq 1000

if [[ "${variable_role}" == "rc" ]]; then
  smoke_input="${project_root}/wkdrs/cv801-agent-review/t2-validation-smoke/visual_reviewer_inputs.jsonl"
  shard_dir="${review_dir}/visual_shards"
  reference_input="${review_dir}/visual_reviewer_inputs.jsonl"
else
  smoke_input="${project_root}/wkdrs/cv801-agent-review/t2-validation-smoke/domain_reviewer_inputs.jsonl"
  shard_dir="${review_dir}/domain_shards"
  reference_input="${review_dir}/domain_reviewer_inputs.jsonl"
fi
test "$(wc -l < "${smoke_input}")" -eq 8

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

submit_reviewer() {
  local input_file="$1"
  local mode="$2"
  local dependency="${3:-}"
  local args=(
    --parsable
    --gres="gpu:mi210:${model_gpus}"
    --cpus-per-task="${model_cpus}"
    --mem="${model_mem}"
    --export="ALL,MODEL_PATH=${model_path},INPUT_FILE=${input_file},ROLE=${variable_role},MODE=${mode},MAX_NEW_TOKENS=4096"
  )
  if [[ -n "${dependency}" ]]; then
    args+=(--dependency="afterok:${dependency}")
  fi
  sbatch "${args[@]}" "${reviewer_script}"
}

smoke_mode="scale-${run_tag}-smoke"
smoke_job="$(submit_reviewer "${smoke_input}" "${smoke_mode}")"
smoke_predictions="$(prediction_path "${smoke_mode}" "${smoke_job}")"
run_dir="${project_root}/wkdrs/cv801-reviewer-scale/${run_tag}-${smoke_job}"
[[ ! -e "${run_dir}" ]] || { echo "Refusing to reuse ${run_dir}." >&2; exit 1; }
mkdir -p "${run_dir}"

smoke_audit="${run_dir}/smoke.audit.json"
smoke_audit_job="$(sbatch --parsable --dependency="afterok:${smoke_job}" \
  --export="ALL,STAGE=audit,PREDICTIONS=${smoke_predictions},AUDIT_KIND=reviewer,AUDIT_OUTPUT=${smoke_audit}" \
  "${agent_stage}")"

reviewer_jobs=()
reviewer_outputs=()
for rank in 0 1 2 3; do
  printf -v shard '%02d' "${rank}"
  input_file="${shard_dir}/input-${shard}.jsonl"
  test "$(wc -l < "${input_file}")" -eq 250
  mode="scale-${run_tag}-s${shard}"
  job_id="$(submit_reviewer "${input_file}" "${mode}" "${smoke_audit_job}")"
  reviewer_jobs+=("${job_id}")
  reviewer_outputs+=("$(prediction_path "${mode}" "${job_id}")")
done

join_colon() { local IFS=:; echo "$*"; }
reviewer_dependency="$(join_colon "${reviewer_jobs[@]}")"
reviewer_sources="$(join_colon "${reviewer_outputs[@]}")"
variable_predictions="${run_dir}/${variable_role}_predictions.jsonl"
reviewer_merge_job="$(sbatch --parsable --dependency="afterok:${reviewer_dependency}" \
  --export="ALL,STAGE=merge_reviewer,OUTPUT=${variable_predictions},REFERENCE_INPUT=${reference_input},SOURCE_PREDICTIONS=${reviewer_sources}" \
  "${agent_stage}")"

if [[ "${variable_role}" == "rc" ]]; then
  rc_predictions="${variable_predictions}"
  rd_predictions="${fixed_rd_predictions}"
else
  rc_predictions="${fixed_rc_predictions}"
  rd_predictions="${variable_predictions}"
fi

judge_input="${run_dir}/judge_inputs.jsonl"
judge_prepare_job="$(sbatch --parsable --dependency="afterok:${reviewer_merge_job}" \
  --export="ALL,STAGE=judge,MANIFEST=${manifest},RC_PREDICTIONS=${rc_predictions},RD_PREDICTIONS=${rd_predictions},OUTPUT=${judge_input}" \
  "${agent_stage}")"
judge_job="$(sbatch --parsable --dependency="afterok:${judge_prepare_job}" \
  --export="ALL,INPUT_FILE=${judge_input},STAGE=judge,MODE=scale-${run_tag},MAX_NEW_TOKENS=4096" \
  "${judge_script}")"
judge_predictions="${project_root}/wkdrs/cv801-agent-review/judge-scale-${run_tag}-${judge_job}/predictions.jsonl"
judge_audit="${run_dir}/judge.audit.json"
judge_audit_job="$(sbatch --parsable --dependency="afterok:${judge_job}" \
  --export="ALL,STAGE=audit,PREDICTIONS=${judge_predictions},AUDIT_KIND=judge,AUDIT_OUTPUT=${judge_audit}" \
  "${agent_stage}")"

revision_input="${run_dir}/revision-full-inputs.jsonl"
revision_prepare_job="$(sbatch --parsable --dependency="afterok:${judge_audit_job}" \
  --export="ALL,STAGE=revision,WRITER_PREDICTIONS=${writer_predictions},FEEDBACK_SOURCE=judge,JUDGE_PREDICTIONS=${judge_predictions},OUTPUT=${revision_input}" \
  "${agent_stage}")"
revision_job="$(sbatch --parsable --dependency="afterok:${revision_prepare_job}" \
  --export="ALL,INPUT_FILE=${revision_input},CONFIGURATION=full,MODE=scale-${run_tag},MAX_NEW_TOKENS=4096" \
  "${revision_script}")"
revision_dir="${project_root}/wkdrs/cv801-agent-review/revision-full-scale-${run_tag}-${revision_job}"
revision_predictions="${revision_dir}/predictions.jsonl"
structured_eval_job="$(sbatch --parsable --dependency="afterok:${revision_job}" \
  --export="ALL,STAGE=evaluate,PREDICTIONS=${revision_predictions},REFERENCE_PREDICTIONS=${reference_predictions},LABELED_OUTPUT=${revision_dir}/metrics.golden.labeled.jsonl,METRICS_OUTPUT=${revision_dir}/metrics.golden.json" \
  "${agent_stage}")"

recheck_input="${run_dir}/recheck_inputs.jsonl"
recheck_prepare_job="$(sbatch --parsable --dependency="afterok:${revision_job}" \
  --export="ALL,STAGE=recheck,MANIFEST=${manifest},RC_PREDICTIONS=${rc_predictions},RD_PREDICTIONS=${rd_predictions},JUDGE_PREDICTIONS=${judge_predictions},REVISED_PREDICTIONS=${revision_predictions},OUTPUT=${recheck_input}" \
  "${agent_stage}")"
recheck_job="$(sbatch --parsable --dependency="afterok:${recheck_prepare_job}" \
  --export="ALL,INPUT_FILE=${recheck_input},STAGE=recheck,MODE=scale-${run_tag},MAX_NEW_TOKENS=4096" \
  "${judge_script}")"
recheck_predictions="${project_root}/wkdrs/cv801-agent-review/recheck-scale-${run_tag}-${recheck_job}/predictions.jsonl"
recheck_audit="${run_dir}/recheck.audit.json"
recheck_audit_job="$(sbatch --parsable --dependency="afterok:${recheck_job}" \
  --export="ALL,STAGE=audit,PREDICTIONS=${recheck_predictions},AUDIT_KIND=recheck,AUDIT_OUTPUT=${recheck_audit}" \
  "${agent_stage}")"

independent_dir="${project_root}/wkdrs/cv801-independent-eval/reviewer-scale-${run_tag}-${smoke_job}"
independent_prepare_job="$(sbatch --parsable --dependency="afterok:${revision_job}" \
  --export="ALL,STAGE=prepare,PREDICTIONS=${revision_predictions},OUTPUT_DIR=${independent_dir},SHARDS=4" \
  "${independent_stage}")"
evaluator_jobs=()
evaluator_outputs=()
for rank in 0 1 2 3; do
  printf -v shard '%02d' "${rank}"
  run_label="reviewer-scale-${run_tag}-s${shard}"
  evaluator_job="$(sbatch --parsable --dependency="afterok:${independent_prepare_job}" \
    --export="ALL,INPUT_FILE=${independent_dir}/shards/input-${shard}.jsonl,RUN_LABEL=${run_label},MAX_NEW_TOKENS=2048" \
    "${evaluator_script}")"
  evaluator_jobs+=("${evaluator_job}")
  evaluator_outputs+=("${project_root}/wkdrs/cv801-independent-eval/${run_label}-${evaluator_job}/predictions.jsonl")
done
evaluator_dependency="$(join_colon "${evaluator_jobs[@]}")"
evaluator_sources="$(join_colon "${evaluator_outputs[@]}")"
independent_merge_job="$(sbatch --parsable --dependency="afterok:${evaluator_dependency}" \
  --export="ALL,STAGE=merge,OUTPUT_DIR=${independent_dir},SOURCE_PREDICTIONS=${evaluator_sources}" \
  "${independent_stage}")"

submitted_uae="$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
{
  printf 'submitted_uae=%q\n' "${submitted_uae}"
  printf 'run_tag=%q\n' "${run_tag}"
  printf 'variable_role=%q\n' "${variable_role}"
  printf 'model_family=%q\n' "${model_family}"
  printf 'model_path=%q\n' "${model_path}"
  printf 'model_commit=%q\n' "$(basename "${model_path}")"
  printf 'model_gpus=%q\n' "${model_gpus}"
  printf 'smoke_job=%q\n' "${smoke_job}"
  printf 'smoke_audit_job=%q\n' "${smoke_audit_job}"
  printf 'reviewer_jobs=%q\n' "$(join_colon "${reviewer_jobs[@]}")"
  printf 'reviewer_merge_job=%q\n' "${reviewer_merge_job}"
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
} > "${run_dir}/submission.env"

printf 'reviewer_scale_status=SUBMITTED run_tag=%s run_dir=%s\n' "${run_tag}" "${run_dir}"
printf 'smoke_job=%s smoke_audit_job=%s reviewer_jobs=%s\n' \
  "${smoke_job}" "${smoke_audit_job}" "$(join_colon "${reviewer_jobs[@]}")"
printf 'terminal_jobs structured_eval=%s recheck_audit=%s independent_eval=%s\n' \
  "${structured_eval_job}" "${recheck_audit_job}" "${independent_merge_job}"
