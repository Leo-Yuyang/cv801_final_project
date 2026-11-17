#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

writer_tag="${WRITER_TAG:?Set WRITER_TAG.}"
variant="${VARIANT:?Set VARIANT to t0, t1, or t2.}"
model_path="${MODEL_PATH:?Set MODEL_PATH.}"
adapter_path="${ADAPTER_PATH:-}"
upstream_dependency="${UPSTREAM_DEPENDENCY:-}"
writer_gpus="${WRITER_GPUS:-4}"
time_limit="${TIME_LIMIT:-12:00:00}"
run_dir="${RUN_DIR:-${project_root}/wkdrs/cv801-writer-scale/${writer_tag}/eval-${variant}}"

[[ "${writer_tag}" =~ ^w(2|4|9)b$ ]] || exit 1
[[ "${variant}" =~ ^t[012]$ ]] || exit 1
[[ "${writer_gpus}" =~ ^[1-8]$ ]] || exit 1
[[ "${time_limit}" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] || exit 1
if [[ -n "${upstream_dependency}" ]]; then
  [[ "${upstream_dependency}" =~ ^[0-9]+(:[0-9]+)*$ ]] || exit 1
fi
if [[ "${variant}" != "t0" ]]; then
  [[ -f "${adapter_path}/adapter_model.safetensors" ]] || {
    echo "Adapter is missing for ${variant}: ${adapter_path}" >&2
    exit 1
  }
fi
[[ ! -e "${run_dir}" ]] || { echo "Refusing to reuse ${run_dir}." >&2; exit 1; }
mkdir -p "${run_dir}"

dependency_args=()
if [[ -n "${upstream_dependency}" ]]; then
  dependency_args+=(--dependency="afterok:${upstream_dependency}")
fi
run_label="${RUN_LABEL:-${writer_tag}-${variant}}"
[[ "${run_label}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || exit 1
writer_export="ALL,VARIANT=${variant},RUN_LABEL=${run_label},MODEL_PATH=${model_path},GPUS=${writer_gpus},MAX_NEW_TOKENS=4096"
if [[ -n "${adapter_path}" ]]; then
  writer_export+=",ADAPTER_PATH=${adapter_path}"
fi
writer_job="$(sbatch --parsable "${dependency_args[@]}" --exclude=auh7-1b-gpu-197 \
  --gres="gpu:mi210:${writer_gpus}" --cpus-per-task="$((16 * writer_gpus))" \
  --mem="$((64 * writer_gpus))G" --time="${time_limit}" \
  --job-name="cv801-${run_label}" --export="${writer_export}" \
  execs/scpts/cv801_golden_writer_infer.sh)"
writer_predictions="${project_root}/wkdrs/cv801-golden-test/writer-${run_label}-${writer_job}/predictions.jsonl"

reference_predictions="${project_root}/wkdrs/cv801-golden-test/vlm-references-final.jsonl"
structured_job="$(sbatch --parsable --dependency="afterok:${writer_job}" \
  --export="ALL,STAGE=evaluate,PREDICTIONS=${writer_predictions},REFERENCE_PREDICTIONS=${reference_predictions},LABELED_OUTPUT=${run_dir}/metrics.golden.labeled.jsonl,METRICS_OUTPUT=${run_dir}/metrics.golden.json" \
  execs/scpts/cv801_agent_review_stage.sh)"

independent_dir="${project_root}/wkdrs/cv801-independent-eval/writer-${run_label}"
independent_prepare_job="$(sbatch --parsable --dependency="afterok:${writer_job}" \
  --export="ALL,STAGE=prepare,PREDICTIONS=${writer_predictions},OUTPUT_DIR=${independent_dir},SHARDS=4" \
  execs/scpts/cv801_independent_eval_stage.sh)"
evaluator_jobs=(); evaluator_outputs=()
for rank in 0 1 2 3; do
  printf -v shard '%02d' "${rank}"
  evaluator_label="writer-${run_label}-eval-s${shard}"
  evaluator_job="$(sbatch --parsable --dependency="afterok:${independent_prepare_job}" \
    --export="ALL,INPUT_FILE=${independent_dir}/shards/input-${shard}.jsonl,RUN_LABEL=${evaluator_label},MAX_NEW_TOKENS=2048" \
    execs/scpts/cv801_pixtral_evaluator_infer.sh)"
  evaluator_jobs+=("${evaluator_job}")
  evaluator_outputs+=("${project_root}/wkdrs/cv801-independent-eval/${evaluator_label}-${evaluator_job}/predictions.jsonl")
done
join_colon() { local IFS=:; echo "$*"; }
independent_merge_job="$(sbatch --parsable --dependency="afterok:$(join_colon "${evaluator_jobs[@]}")" \
  --export="ALL,STAGE=merge,OUTPUT_DIR=${independent_dir},SOURCE_PREDICTIONS=$(join_colon "${evaluator_outputs[@]}")" \
  execs/scpts/cv801_independent_eval_stage.sh)"

submitted_uae="$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
{
  printf 'submitted_uae=%q\n' "${submitted_uae}"
  printf 'writer_tag=%q\n' "${writer_tag}"
  printf 'variant=%q\n' "${variant}"
  printf 'model_path=%q\n' "${model_path}"
  printf 'adapter_path=%q\n' "${adapter_path}"
  printf 'writer_gpus=%q\n' "${writer_gpus}"
  printf 'time_limit=%q\n' "${time_limit}"
  printf 'writer_job=%q\n' "${writer_job}"
  printf 'writer_predictions=%q\n' "${writer_predictions}"
  printf 'structured_job=%q\n' "${structured_job}"
  printf 'independent_prepare_job=%q\n' "${independent_prepare_job}"
  printf 'evaluator_jobs=%q\n' "$(join_colon "${evaluator_jobs[@]}")"
  printf 'independent_merge_job=%q\n' "${independent_merge_job}"
} > "${run_dir}/submission.env"

printf 'writer_eval=SUBMITTED writer=%s variant=%s inference=%s structured=%s independent=%s predictions=%s\n' \
  "${writer_tag}" "${variant}" "${writer_job}" "${structured_job}" \
  "${independent_merge_job}" "${writer_predictions}"
