#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

smoke_audit="${SMOKE_AUDIT:?Set SMOKE_AUDIT to a passed Gemma 3 reviewer smoke audit.}"
[[ -f "${smoke_audit}" ]] || { echo "Smoke audit is missing: ${smoke_audit}" >&2; exit 1; }
jq -e '.samples == 8 and .schema_valid == 8' "${smoke_audit}" >/dev/null

review_dir="${project_root}/wkdrs/cv801-golden-test/reviewer-inputs-403233"
writer_predictions="${project_root}/wkdrs/cv801-golden-test/writer-t2-403232/predictions.jsonl"
reference_predictions="${project_root}/wkdrs/cv801-golden-test/vlm-references-final.jsonl"
rd_predictions="${review_dir}/rd_predictions.jsonl"
manifest="${review_dir}/review_manifest.jsonl"
model_path="${project_root}/inits/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"

reviewer_script="execs/scpts/cv801_gemma3_reviewer_infer.sh"
agent_stage="execs/scpts/cv801_agent_review_stage.sh"
judge_script="execs/scpts/cv801_qwen35_judge_infer.sh"
revision_script="execs/scpts/cv801_qwen35_revision_infer.sh"
independent_stage="execs/scpts/cv801_independent_eval_stage.sh"
evaluator_script="execs/scpts/cv801_pixtral_evaluator_infer.sh"

for required in "${writer_predictions}" "${reference_predictions}" "${rd_predictions}" \
  "${manifest}" "${model_path}/config.json"; do
  [[ -f "${required}" ]] || { echo "Required frozen input is missing: ${required}" >&2; exit 1; }
done
test "$(wc -l < "${writer_predictions}")" -eq 1000
test "$(wc -l < "${rd_predictions}")" -eq 1000
test "$(wc -l < "${manifest}")" -eq 1000

rc_jobs=()
rc_outputs=()
for rank in 0 1 2 3; do
  printf -v shard '%02d' "${rank}"
  input_file="${review_dir}/visual_shards/input-${shard}.jsonl"
  test "$(wc -l < "${input_file}")" -eq 250
  rc_job=$(sbatch --parsable \
    --export=ALL,MODEL_PATH="${model_path}",INPUT_FILE="${input_file}",MODE="golden-s${shard}",MAX_NEW_TOKENS=4096 \
    "${reviewer_script}")
  rc_jobs+=("${rc_job}")
  rc_outputs+=("${project_root}/wkdrs/cv801-agent-review/rc-gemma3-golden-s${shard}-${rc_job}/predictions.jsonl")
done

run_dir="${project_root}/wkdrs/cv801-golden-test/gemma3-reviewer-swap-${rc_jobs[0]}"
[[ ! -e "${run_dir}" ]] || { echo "Refusing to reuse ${run_dir}." >&2; exit 1; }
mkdir -p "${run_dir}"

join_colon() { local IFS=:; echo "$*"; }
rc_dependency=$(join_colon "${rc_jobs[@]}")
rc_sources=$(join_colon "${rc_outputs[@]}")
rc_predictions="${run_dir}/rc_predictions.jsonl"
rc_merge_job=$(sbatch --parsable --dependency="afterok:${rc_dependency}" \
  --export=ALL,STAGE=merge_reviewer,OUTPUT="${rc_predictions}",REFERENCE_INPUT="${review_dir}/visual_reviewer_inputs.jsonl",SOURCE_PREDICTIONS="${rc_sources}" \
  "${agent_stage}")
rc_audit="${run_dir}/rc.audit.json"
rc_audit_job=$(sbatch --parsable --dependency="afterok:${rc_merge_job}" \
  --export=ALL,STAGE=audit,PREDICTIONS="${rc_predictions}",AUDIT_KIND=reviewer,AUDIT_OUTPUT="${rc_audit}" \
  "${agent_stage}")

judge_input="${run_dir}/judge_inputs.jsonl"
judge_prepare_job=$(sbatch --parsable --dependency="afterok:${rc_audit_job}" \
  --export=ALL,STAGE=judge,MANIFEST="${manifest}",RC_PREDICTIONS="${rc_predictions}",RD_PREDICTIONS="${rd_predictions}",OUTPUT="${judge_input}" \
  "${agent_stage}")
judge_job=$(sbatch --parsable --dependency="afterok:${judge_prepare_job}" \
  --export=ALL,INPUT_FILE="${judge_input}",STAGE=judge,MODE=golden-gemma3-rc,MAX_NEW_TOKENS=4096 \
  "${judge_script}")
judge_predictions="${project_root}/wkdrs/cv801-agent-review/judge-golden-gemma3-rc-${judge_job}/predictions.jsonl"
judge_audit="${run_dir}/judge.audit.json"
judge_audit_job=$(sbatch --parsable --dependency="afterok:${judge_job}" \
  --export=ALL,STAGE=audit,PREDICTIONS="${judge_predictions}",AUDIT_KIND=judge,AUDIT_OUTPUT="${judge_audit}" \
  "${agent_stage}")

revision_input="${run_dir}/revision-full-inputs.jsonl"
revision_prepare_job=$(sbatch --parsable --dependency="afterok:${judge_audit_job}" \
  --export=ALL,STAGE=revision,WRITER_PREDICTIONS="${writer_predictions}",FEEDBACK_SOURCE=judge,JUDGE_PREDICTIONS="${judge_predictions}",OUTPUT="${revision_input}" \
  "${agent_stage}")
revision_job=$(sbatch --parsable --dependency="afterok:${revision_prepare_job}" \
  --export=ALL,INPUT_FILE="${revision_input}",CONFIGURATION=full,MODE=golden-gemma3-rc,MAX_NEW_TOKENS=4096 \
  "${revision_script}")
revision_dir="${project_root}/wkdrs/cv801-agent-review/revision-full-golden-gemma3-rc-${revision_job}"
revision_predictions="${revision_dir}/predictions.jsonl"
structured_eval_job=$(sbatch --parsable --dependency="afterok:${revision_job}" \
  --export=ALL,STAGE=evaluate,PREDICTIONS="${revision_predictions}",REFERENCE_PREDICTIONS="${reference_predictions}",LABELED_OUTPUT="${revision_dir}/metrics.golden.labeled.jsonl",METRICS_OUTPUT="${revision_dir}/metrics.golden.json" \
  "${agent_stage}")

recheck_input="${run_dir}/recheck_inputs.jsonl"
recheck_prepare_job=$(sbatch --parsable --dependency="afterok:${revision_job}" \
  --export=ALL,STAGE=recheck,MANIFEST="${manifest}",RC_PREDICTIONS="${rc_predictions}",RD_PREDICTIONS="${rd_predictions}",JUDGE_PREDICTIONS="${judge_predictions}",REVISED_PREDICTIONS="${revision_predictions}",OUTPUT="${recheck_input}" \
  "${agent_stage}")
recheck_job=$(sbatch --parsable --dependency="afterok:${recheck_prepare_job}" \
  --export=ALL,INPUT_FILE="${recheck_input}",STAGE=recheck,MODE=golden-gemma3-rc,MAX_NEW_TOKENS=4096 \
  "${judge_script}")
recheck_predictions="${project_root}/wkdrs/cv801-agent-review/recheck-golden-gemma3-rc-${recheck_job}/predictions.jsonl"
recheck_audit="${run_dir}/recheck.audit.json"
recheck_audit_job=$(sbatch --parsable --dependency="afterok:${recheck_job}" \
  --export=ALL,STAGE=audit,PREDICTIONS="${recheck_predictions}",AUDIT_KIND=recheck,AUDIT_OUTPUT="${recheck_audit}" \
  "${agent_stage}")

independent_dir="${project_root}/wkdrs/cv801-independent-eval/golden-gemma3-full-${rc_jobs[0]}"
independent_prepare_job=$(sbatch --parsable --dependency="afterok:${revision_job}" \
  --export=ALL,STAGE=prepare,PREDICTIONS="${revision_predictions}",OUTPUT_DIR="${independent_dir}",SHARDS=4 \
  "${independent_stage}")
evaluator_jobs=()
evaluator_outputs=()
for rank in 0 1 2 3; do
  printf -v shard '%02d' "${rank}"
  run_label="golden-gemma3-full-s${shard}"
  evaluator_job=$(sbatch --parsable --dependency="afterok:${independent_prepare_job}" \
    --export=ALL,INPUT_FILE="${independent_dir}/shards/input-${shard}.jsonl",RUN_LABEL="${run_label}",MAX_NEW_TOKENS=2048 \
    "${evaluator_script}")
  evaluator_jobs+=("${evaluator_job}")
  evaluator_outputs+=("${project_root}/wkdrs/cv801-independent-eval/${run_label}-${evaluator_job}/predictions.jsonl")
done
evaluator_dependency=$(join_colon "${evaluator_jobs[@]}")
evaluator_sources=$(join_colon "${evaluator_outputs[@]}")
independent_merge_job=$(sbatch --parsable --dependency="afterok:${evaluator_dependency}" \
  --export=ALL,STAGE=merge,OUTPUT_DIR="${independent_dir}",SOURCE_PREDICTIONS="${evaluator_sources}" \
  "${independent_stage}")

submitted_uae=$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')
{
  printf 'submitted_uae=%q\n' "${submitted_uae}"
  printf 'model_commit=%q\n' '96b6f1eccf38110c56df3a15bffe176da04bfd80'
  printf 'smoke_audit=%q\n' "${smoke_audit}"
  printf 'rc_jobs=%q\n' "$(join_colon "${rc_jobs[@]}")"
  printf 'rc_merge_job=%q\n' "${rc_merge_job}"
  printf 'rc_audit_job=%q\n' "${rc_audit_job}"
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

printf 'gemma3_reviewer_swap_status=SUBMITTED run_dir=%s\n' "${run_dir}"
printf 'rc_jobs=%s\n' "$(join_colon "${rc_jobs[@]}")"
printf 'terminal_jobs structured_eval=%s recheck_audit=%s independent_eval=%s\n' \
  "${structured_eval_job}" "${recheck_audit_job}" "${independent_merge_job}"
