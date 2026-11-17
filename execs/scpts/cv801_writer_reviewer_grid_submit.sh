#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

writer_tag="${WRITER_TAG:?Set WRITER_TAG, for example w2b.}"
writer_predictions="${WRITER_PREDICTIONS:?Set WRITER_PREDICTIONS.}"
writer_model_path="${WRITER_MODEL_PATH:?Set WRITER_MODEL_PATH.}"
writer_adapter_path="${WRITER_ADAPTER_PATH:?Set WRITER_ADAPTER_PATH.}"
review_dir="${REVIEW_DIR:?Set REVIEW_DIR.}"
review_input_dependency="${REVIEW_INPUT_DEPENDENCY:-}"
tiers_value="${TIERS:-low mid high}"
run_root="${RUN_ROOT:-${project_root}/wkdrs/cv801-writer-reviewer-grid}"

[[ "${writer_tag}" =~ ^w(2|4|9)b$ ]] || exit 1
if [[ -n "${review_input_dependency}" ]]; then
  [[ "${review_input_dependency}" =~ ^[0-9]+$ ]] || exit 1
fi
[[ -f "${writer_model_path}/config.json" ]] || exit 1
[[ -f "${writer_adapter_path}/adapter_model.safetensors" ]] || exit 1
mkdir -p "${run_root}"

gemma4="${project_root}/inits/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
gemma12="${project_root}/inits/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
gemma27="${project_root}/inits/huggingface/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a"
intern8="${project_root}/inits/huggingface/hub/models--OpenGVLab--InternVL3_5-8B-HF/snapshots/741a7d03020411e666c6109218ab71e08151ef86"
intern14="${project_root}/inits/huggingface/hub/models--OpenGVLab--InternVL3_5-14B-HF/snapshots/226b96d5912e69159abc0384cefcbd51487fdce0"
intern38="${project_root}/inits/huggingface/hub/models--OpenGVLab--InternVL3_5-38B-HF/snapshots/7c830fc25e874f9a861087e3c70587f2b9b555f3"
for model in "${gemma4}" "${gemma12}" "${gemma27}" "${intern8}" "${intern14}" "${intern38}"; do
  [[ -f "${model}/config.json" ]] || { echo "Reviewer model is missing: ${model}" >&2; exit 1; }
done

join_colon() { local IFS=:; echo "$*"; }
for tier in ${tiers_value}; do
  case "${tier}" in
    low)
      rc_model="${gemma4}"; rc_gpus=1; rd_model="${intern8}"; rd_gpus=1
      ;;
    mid)
      rc_model="${gemma12}"; rc_gpus=1; rd_model="${intern14}"; rd_gpus=1
      ;;
    high)
      rc_model="${gemma27}"; rc_gpus=2; rd_model="${intern38}"; rd_gpus=2
      ;;
    *)
      echo "Unsupported reviewer tier: ${tier}" >&2
      exit 1
      ;;
  esac

  combo_tag="grid-${writer_tag}-${tier}"
  combo_dir="${run_root}/${combo_tag}"
  [[ ! -e "${combo_dir}" ]] || { echo "Refusing to reuse ${combo_dir}." >&2; exit 1; }
  mkdir -p "${combo_dir}"
  reviewer_dependency_args=()
  if [[ -n "${review_input_dependency}" ]]; then
    reviewer_dependency_args+=(--dependency="afterok:${review_input_dependency}")
  fi
  rc_jobs=(); rd_jobs=(); rc_outputs=(); rd_outputs=()
  for rank in 0 1 2 3; do
    printf -v shard '%02d' "${rank}"
    rc_mode="${combo_tag}-rc-s${shard}"
    rd_mode="${combo_tag}-rd-s${shard}"
    rc_job="$(sbatch --parsable "${reviewer_dependency_args[@]}" \
      --exclude=auh7-1b-gpu-197 --time=12:00:00 --job-name="cv801-${writer_tag}-${tier}-rc${shard}" \
      --gres="gpu:mi210:${rc_gpus}" --cpus-per-task="$((16 * rc_gpus))" --mem="$((160 * rc_gpus))G" \
      --export="ALL,MODEL_PATH=${rc_model},INPUT_FILE=${review_dir}/visual_shards/input-${shard}.jsonl,MODE=${rc_mode},MAX_NEW_TOKENS=4096" \
      execs/scpts/cv801_gemma3_reviewer_infer.sh)"
    rd_job="$(sbatch --parsable "${reviewer_dependency_args[@]}" \
      --exclude=auh7-1b-gpu-197 --time=12:00:00 --job-name="cv801-${writer_tag}-${tier}-rd${shard}" \
      --gres="gpu:mi210:${rd_gpus}" --cpus-per-task="$((16 * rd_gpus))" --mem="$((160 * rd_gpus))G" \
      --export="ALL,MODEL_PATH=${rd_model},INPUT_FILE=${review_dir}/domain_shards/input-${shard}.jsonl,ROLE=rd,MODE=${rd_mode},MAX_NEW_TOKENS=4096" \
      execs/scpts/cv801_reviewer_infer.sh)"
    rc_jobs+=("${rc_job}"); rd_jobs+=("${rd_job}")
    rc_outputs+=("${project_root}/wkdrs/cv801-agent-review/rc-gemma3-${rc_mode}-${rc_job}/predictions.jsonl")
    rd_outputs+=("${project_root}/wkdrs/cv801-agent-review/rd-${rd_mode}-${rd_job}/predictions.jsonl")
  done

  rc_predictions="${combo_dir}/rc_predictions.jsonl"
  rd_predictions="${combo_dir}/rd_predictions.jsonl"
  rc_merge_job="$(sbatch --parsable --dependency="afterok:$(join_colon "${rc_jobs[@]}")" \
    --export="ALL,STAGE=merge_reviewer,OUTPUT=${rc_predictions},REFERENCE_INPUT=${review_dir}/visual_reviewer_inputs.jsonl,SOURCE_PREDICTIONS=$(join_colon "${rc_outputs[@]}")" \
    execs/scpts/cv801_agent_review_stage.sh)"
  rd_merge_job="$(sbatch --parsable --dependency="afterok:$(join_colon "${rd_jobs[@]}")" \
    --export="ALL,STAGE=merge_reviewer,OUTPUT=${rd_predictions},REFERENCE_INPUT=${review_dir}/domain_reviewer_inputs.jsonl,SOURCE_PREDICTIONS=$(join_colon "${rd_outputs[@]}")" \
    execs/scpts/cv801_agent_review_stage.sh)"

  RUN_TAG="${combo_tag}" \
  WRITER_PREDICTIONS="${writer_predictions}" \
  WRITER_MODEL_PATH="${writer_model_path}" \
  WRITER_ADAPTER_PATH="${writer_adapter_path}" \
  RC_PREDICTIONS="${rc_predictions}" \
  RD_PREDICTIONS="${rd_predictions}" \
  MANIFEST="${review_dir}/review_manifest.jsonl" \
  UPSTREAM_DEPENDENCY="${rc_merge_job}:${rd_merge_job}" \
  RUN_DIR="${combo_dir}/downstream" \
  execs/scpts/cv801_agent_review_full_submit.sh

  submitted_uae="$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
  {
    printf 'submitted_uae=%q\n' "${submitted_uae}"
    printf 'combo_tag=%q\n' "${combo_tag}"
    printf 'writer_predictions=%q\n' "${writer_predictions}"
    printf 'writer_model_path=%q\n' "${writer_model_path}"
    printf 'writer_adapter_path=%q\n' "${writer_adapter_path}"
    printf 'rc_model=%q\n' "${rc_model}"
    printf 'rd_model=%q\n' "${rd_model}"
    printf 'rc_jobs=%q\n' "$(join_colon "${rc_jobs[@]}")"
    printf 'rd_jobs=%q\n' "$(join_colon "${rd_jobs[@]}")"
    printf 'rc_merge_job=%q\n' "${rc_merge_job}"
    printf 'rd_merge_job=%q\n' "${rd_merge_job}"
  } > "${combo_dir}/submission.env"
  printf 'writer_reviewer_grid=SUBMITTED combo=%s reviewer_jobs=%s:%s merge=%s:%s\n' \
    "${combo_tag}" "$(join_colon "${rc_jobs[@]}")" "$(join_colon "${rd_jobs[@]}")" \
    "${rc_merge_job}" "${rd_merge_job}"
done
