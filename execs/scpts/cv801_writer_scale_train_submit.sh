#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

model_tag="${MODEL_TAG:?Set MODEL_TAG, for example qwen35-2b.}"
model_id="${MODEL_ID:?Set MODEL_ID.}"
revision="${REVISION:?Set immutable Hugging Face revision.}"
model_path="${MODEL_PATH:?Set the expected local snapshot path.}"
[[ "${model_tag}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || exit 1
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]] || exit 1

download_script="execs/scpts/cv801_download_hf_model.sh"
t1_script="execs/scpts/cv801_qwen35_t1_train.sh"
t2_script="execs/scpts/cv801_qwen35_t2_train.sh"
run_dir="${project_root}/wkdrs/cv801-writer-scale/${model_tag}"
[[ ! -e "${run_dir}" ]] || { echo "Refusing to reuse ${run_dir}." >&2; exit 1; }
mkdir -p "${run_dir}" "${project_root}/wkdrs/cv801-model-downloads" \
  "${project_root}/wkdrs/cv801-qwen35"

download_job="$(sbatch --parsable --exclude=auh7-1b-gpu-197 \
  --job-name="cv801-${model_tag}-download" \
  --export="ALL,MODEL_ID=${model_id},REVISION=${revision}" \
  "${download_script}")"

submit_probe() {
  local task="$1"
  local probe="$2"
  local script="$3"
  local max_length="$4"
  local stress=0
  [[ "${probe}" == "stress" ]] && stress=1
  sbatch --parsable --dependency="afterok:${download_job}" \
    --exclude=auh7-1b-gpu-197 --gres=gpu:mi210:1 --cpus-per-task=16 --mem=128G \
    --time=04:00:00 --job-name="cv801-${model_tag}-${task}-${probe}" \
    --export="ALL,MODEL_PATH=${model_path},GPUS=1,SMOKE=1,STRESS=${stress},MAX_LENGTH=${max_length}" \
    "${script}"
}

t1_smoke_job="$(submit_probe t1 smoke "${t1_script}" 4096)"
t1_stress_job="$(submit_probe t1 stress "${t1_script}" 4096)"
t2_smoke_job="$(submit_probe t2 smoke "${t2_script}" 6144)"
t2_stress_job="$(submit_probe t2 stress "${t2_script}" 6144)"

t1_train_job="$(sbatch --parsable --dependency="afterok:${t1_smoke_job}:${t1_stress_job}" \
  --exclude=auh7-1b-gpu-197 --gres=gpu:mi210:4 --cpus-per-task=64 --mem=256G \
  --time=1-00:00:00 --job-name="cv801-${model_tag}-t1-train" \
  --export="ALL,MODEL_PATH=${model_path},GPUS=4,SMOKE=0,EPOCHS=3,GRADIENT_ACCUMULATION=4,MAX_LENGTH=4096,LEARNING_RATE=1e-4,LORA_RANK=64,LORA_ALPHA=128,LORA_DROPOUT=0.05" \
  "${t1_script}")"
t2_train_job="$(sbatch --parsable --dependency="afterok:${t2_smoke_job}:${t2_stress_job}" \
  --exclude=auh7-1b-gpu-197 --gres=gpu:mi210:4 --cpus-per-task=64 --mem=256G \
  --time=1-00:00:00 --job-name="cv801-${model_tag}-t2-train" \
  --export="ALL,MODEL_PATH=${model_path},GPUS=4,SMOKE=0,EPOCHS=3,GRADIENT_ACCUMULATION=4,MAX_LENGTH=6144,LEARNING_RATE=1e-4,LORA_RANK=64,LORA_ALPHA=128,LORA_DROPOUT=0.05" \
  "${t2_script}")"

submitted_uae="$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
{
  printf 'submitted_uae=%q\n' "${submitted_uae}"
  printf 'model_tag=%q\n' "${model_tag}"
  printf 'model_id=%q\n' "${model_id}"
  printf 'revision=%q\n' "${revision}"
  printf 'model_path=%q\n' "${model_path}"
  printf 'download_job=%q\n' "${download_job}"
  printf 't1_smoke_job=%q\n' "${t1_smoke_job}"
  printf 't1_stress_job=%q\n' "${t1_stress_job}"
  printf 't2_smoke_job=%q\n' "${t2_smoke_job}"
  printf 't2_stress_job=%q\n' "${t2_stress_job}"
  printf 't1_train_job=%q\n' "${t1_train_job}"
  printf 't2_train_job=%q\n' "${t2_train_job}"
} > "${run_dir}/submission.env"

printf 'writer_scale_training=SUBMITTED model=%s download=%s probes=%s:%s:%s:%s train=%s:%s\n' \
  "${model_tag}" "${download_job}" "${t1_smoke_job}" "${t1_stress_job}" \
  "${t2_smoke_job}" "${t2_stress_job}" "${t1_train_job}" "${t2_train_job}"
