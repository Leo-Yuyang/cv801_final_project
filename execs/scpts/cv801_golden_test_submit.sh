#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${project_root}/.env"
safe_root="${project_root}/datas/cv801_golden_test_inputs_v1"
review_configuration="${REVIEW_CONFIGURATION:-}"

"${PYTHON_HOME}/bin/python" "${project_root}/CV801/scripts/prepare_golden_test_inputs.py" \
  audit --input-root "${safe_root}" --expected-images 1000

cat <<'EOF'
Golden Test inference DAG (no evaluation jobs are included):

  YOLO26x ──> build reference-free T2 inputs ──> T2 Writer ──> Rc/Rd shards
      │                                                        │
      ├────────────────────────────────────────────────────────> Judge
      │                                                        │
      │                                                        └─> selected revision ─> optional recheck
      │
      ├─ Grounding DINO     (parallel detection output only)
      └─ LocateAnything     (parallel detection output only)

  T0 Writer ─┐
  T1 Writer ─┴─ run in parallel with the detector and T2 branches

Golden references and all metric evaluation remain outside this DAG.
EOF

if [[ "${SUBMIT:-0}" != "1" ]]; then
  echo "golden_test_submit_status=DRY_RUN jobs_submitted=0"
  echo "After validation freezes the protocol, set PROTOCOL_FROZEN=YES, REVIEW_CONFIGURATION, and SUBMIT=1."
  exit 0
fi

[[ "${PROTOCOL_FROZEN:-NO}" == "YES" ]] || {
  echo "Refusing formal Golden Test inference until PROTOCOL_FROZEN=YES." >&2
  exit 1
}
[[ "${review_configuration}" =~ ^(rc|rd|rc_rd|full)$ ]] || {
  echo "Set REVIEW_CONFIGURATION to the validation-selected rc, rd, rc_rd, or full." >&2
  exit 1
}
cd "${project_root}"
mkdir -p wkdrs/cv801-golden-test

detection_script="execs/scpts/cv801_golden_detection_infer.sh"
writer_script="execs/scpts/cv801_golden_writer_infer.sh"
golden_stage="execs/scpts/cv801_golden_test_stage.sh"
reviewer_script="execs/scpts/cv801_reviewer_infer.sh"
agent_stage="execs/scpts/cv801_agent_review_stage.sh"
judge_script="execs/scpts/cv801_qwen35_judge_infer.sh"
revision_script="execs/scpts/cv801_qwen35_revision_infer.sh"

yolo_job=$(sbatch --parsable --export=ALL,MODEL_FAMILY=yolo "${detection_script}")
gdino_job=$(sbatch --parsable --export=ALL,MODEL_FAMILY=grounding_dino "${detection_script}")
locate_job=$(sbatch --parsable --export=ALL,MODEL_FAMILY=locateanything "${detection_script}")
t0_job=$(sbatch --parsable --export=ALL,VARIANT=t0 "${writer_script}")
t1_job=$(sbatch --parsable --export=ALL,VARIANT=t1 "${writer_script}")

yolo_predictions="${project_root}/wkdrs/cv801-golden-test/detection-yolo-${yolo_job}/predictions.coco.json"
t2_prepare_job=$(sbatch --parsable --dependency="afterok:${yolo_job}" \
  --export=ALL,STAGE=prepare_t2,YOLO_PREDICTIONS="${yolo_predictions}" \
  "${golden_stage}")
t2_job=$(sbatch --parsable --dependency="afterok:${t2_prepare_job}" \
  --export=ALL,VARIANT=t2 "${writer_script}")
t2_predictions="${project_root}/wkdrs/cv801-golden-test/writer-t2-${t2_job}/predictions.jsonl"

review_input_job=$(sbatch --parsable --dependency="afterok:${t2_job}" \
  --export=ALL,STAGE=reviewers,WRITER_PREDICTIONS="${t2_predictions}",SHARDS=4 \
  "${golden_stage}")
review_dir="${project_root}/wkdrs/cv801-golden-test/reviewer-inputs-${review_input_job}"

rc_model="${project_root}/inits/huggingface/hub/models--OpenGVLab--InternVL3_5-8B-HF/snapshots/741a7d03020411e666c6109218ab71e08151ef86"
rd_model="${project_root}/inits/huggingface/hub/models--OpenGVLab--InternVL3_5-38B-HF/snapshots/7c830fc25e874f9a861087e3c70587f2b9b555f3"
rc_jobs=()
rd_jobs=()
rc_outputs=()
rd_outputs=()
for rank in 0 1 2 3; do
  printf -v shard '%02d' "${rank}"
  rc_job=$(sbatch --parsable --dependency="afterok:${review_input_job}" \
    --export=ALL,MODEL_PATH="${rc_model}",INPUT_FILE="${review_dir}/visual_shards/input-${shard}.jsonl",ROLE=rc,MODE="golden-s${shard}",MAX_NEW_TOKENS=4096 \
    "${reviewer_script}")
  rd_job=$(sbatch --parsable --dependency="afterok:${review_input_job}" \
    --gres=gpu:mi210:2 --cpus-per-task=32 --mem=320G \
    --export=ALL,MODEL_PATH="${rd_model}",INPUT_FILE="${review_dir}/domain_shards/input-${shard}.jsonl",ROLE=rd,MODE="golden-s${shard}",MAX_NEW_TOKENS=4096 \
    "${reviewer_script}")
  rc_jobs+=("${rc_job}")
  rd_jobs+=("${rd_job}")
  rc_outputs+=("${project_root}/wkdrs/cv801-agent-review/rc-golden-s${shard}-${rc_job}/predictions.jsonl")
  rd_outputs+=("${project_root}/wkdrs/cv801-agent-review/rd-golden-s${shard}-${rd_job}/predictions.jsonl")
done

join_colon() { local IFS=:; echo "$*"; }
rc_dependency=$(join_colon "${rc_jobs[@]}")
rd_dependency=$(join_colon "${rd_jobs[@]}")
rc_sources=$(join_colon "${rc_outputs[@]}")
rd_sources=$(join_colon "${rd_outputs[@]}")
rc_merged="${review_dir}/rc_predictions.jsonl"
rd_merged="${review_dir}/rd_predictions.jsonl"
rc_merge_job=$(sbatch --parsable --dependency="afterok:${rc_dependency}" \
  --export=ALL,STAGE=merge_reviewer,OUTPUT="${rc_merged}",REFERENCE_INPUT="${review_dir}/visual_reviewer_inputs.jsonl",SOURCE_PREDICTIONS="${rc_sources}" \
  "${agent_stage}")
rd_merge_job=$(sbatch --parsable --dependency="afterok:${rd_dependency}" \
  --export=ALL,STAGE=merge_reviewer,OUTPUT="${rd_merged}",REFERENCE_INPUT="${review_dir}/domain_reviewer_inputs.jsonl",SOURCE_PREDICTIONS="${rd_sources}" \
  "${agent_stage}")

judge_input="${review_dir}/judge_inputs.jsonl"
judge_prepare_job=$(sbatch --parsable --dependency="afterok:${rc_merge_job}:${rd_merge_job}" \
  --export=ALL,STAGE=judge,MANIFEST="${review_dir}/review_manifest.jsonl",RC_PREDICTIONS="${rc_merged}",RD_PREDICTIONS="${rd_merged}",OUTPUT="${judge_input}" \
  "${agent_stage}")
judge_job=$(sbatch --parsable --dependency="afterok:${judge_prepare_job}" \
  --export=ALL,INPUT_FILE="${judge_input}",STAGE=judge,MODE=golden,MAX_NEW_TOKENS=4096 \
  "${judge_script}")
judge_predictions="${project_root}/wkdrs/cv801-agent-review/judge-golden-${judge_job}/predictions.jsonl"

revision_input="${review_dir}/revision-${review_configuration}-inputs.jsonl"
revision_exports="ALL,STAGE=revision,WRITER_PREDICTIONS=${t2_predictions},OUTPUT=${revision_input}"
case "${review_configuration}" in
  rc)
    revision_exports+=",FEEDBACK_SOURCE=rc,RC_PREDICTIONS=${rc_merged}"
    ;;
  rd)
    revision_exports+=",FEEDBACK_SOURCE=rd,RD_PREDICTIONS=${rd_merged}"
    ;;
  rc_rd)
    revision_exports+=",FEEDBACK_SOURCE=rc_rd,RC_PREDICTIONS=${rc_merged},RD_PREDICTIONS=${rd_merged}"
    ;;
  full)
    revision_exports+=",FEEDBACK_SOURCE=judge,JUDGE_PREDICTIONS=${judge_predictions}"
    ;;
esac
revision_prepare_job=$(sbatch --parsable --dependency="afterok:${judge_job}" \
  --export="${revision_exports}" "${agent_stage}")
revision_job=$(sbatch --parsable --dependency="afterok:${revision_prepare_job}" \
  --export=ALL,INPUT_FILE="${revision_input}",CONFIGURATION="${review_configuration}",MODE=golden,MAX_NEW_TOKENS=4096 \
  "${revision_script}")
revision_predictions="${project_root}/wkdrs/cv801-agent-review/revision-${review_configuration}-golden-${revision_job}/predictions.jsonl"

terminal_job="${revision_job}"
if [[ "${review_configuration}" == "full" ]]; then
  recheck_input="${review_dir}/recheck_inputs.jsonl"
  recheck_prepare_job=$(sbatch --parsable --dependency="afterok:${revision_job}" \
    --export=ALL,STAGE=recheck,MANIFEST="${review_dir}/review_manifest.jsonl",RC_PREDICTIONS="${rc_merged}",RD_PREDICTIONS="${rd_merged}",JUDGE_PREDICTIONS="${judge_predictions}",REVISED_PREDICTIONS="${revision_predictions}",OUTPUT="${recheck_input}" \
    "${agent_stage}")
  recheck_job=$(sbatch --parsable --dependency="afterok:${recheck_prepare_job}" \
    --export=ALL,INPUT_FILE="${recheck_input}",STAGE=recheck,MODE=golden,MAX_NEW_TOKENS=4096 \
    "${judge_script}")
  recheck_predictions="${project_root}/wkdrs/cv801-agent-review/recheck-golden-${recheck_job}/predictions.jsonl"
  terminal_job=$(sbatch --parsable --dependency="afterok:${recheck_job}" \
    --export=ALL,STAGE=audit,PREDICTIONS="${recheck_predictions}",AUDIT_KIND=judge \
    "${agent_stage}")
fi

printf 'golden_test_submit_status=SUBMITTED protocol_frozen=YES review_configuration=%s\n' "${review_configuration}"
printf 'detectors yolo=%s grounding_dino=%s locateanything=%s\n' "${yolo_job}" "${gdino_job}" "${locate_job}"
printf 'writers t0=%s t1=%s t2=%s\n' "${t0_job}" "${t1_job}" "${t2_job}"
printf 'agent_review_terminal_job=%s evaluation_jobs_submitted=0\n' "${terminal_job}"
