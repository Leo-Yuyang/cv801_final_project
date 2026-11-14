#!/usr/bin/env bash
#SBATCH --job-name=cv801-qwen-t2i
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=16:00:00

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
swift_bin="${project_root}/inits/envs/cv801-qwen35/bin/swift"
dataset_root="${project_root}/datas/cv801_vlm_t2_yolo26x_v1"
train_job_id="${TRAIN_JOB_ID:?Set TRAIN_JOB_ID to the completed formal T2 job}"
train_root="${project_root}/wkdrs/cv801-qwen35/t2-train-${train_job_id}"
smoke="${SMOKE:-0}"

[[ -f "${container}" && -f "${swift_bin}" && -d "${train_root}" ]] || {
  echo "Qwen runtime or T2 training output is missing." >&2
  exit 1
}

latest_state=$(find "${train_root}" -mindepth 2 -maxdepth 3 -type f -name trainer_state.json | sort -V | tail -n 1)
adapter=""
if [[ -n "${latest_state}" ]]; then
  adapter=$(jq -r '.best_model_checkpoint // empty' "${latest_state}")
fi
if [[ -z "${adapter}" || ! -f "${adapter}/adapter_config.json" ]]; then
  adapter=$(find "${train_root}" -mindepth 2 -maxdepth 3 -type f -name adapter_config.json -printf '%h\n' | sort -V | tail -n 1)
fi
[[ -n "${adapter}" && -f "${adapter}/adapter_model.safetensors" ]] || {
  echo "No complete LoRA checkpoint found under ${train_root}." >&2
  exit 1
}

if [[ "${smoke}" == "1" ]]; then
  val_data="${dataset_root}/t2_val_smoke.jsonl"
  run_name="t2-infer-smoke-${SLURM_JOB_ID}"
else
  val_data="${dataset_root}/t2_val.jsonl"
  run_name="t2-val-${SLURM_JOB_ID}"
fi
output_dir="${project_root}/wkdrs/cv801-qwen35/${run_name}"
result_path="${output_dir}/predictions.jsonl"
mkdir -p "${output_dir}"
[[ ! -e "${result_path}" ]] || { echo "Refusing to append to existing ${result_path}" >&2; exit 1; }
printf '%s\n' "${adapter}" > "${output_dir}/adapter_path.txt"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=true
export APPTAINERENV_IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
export APPTAINERENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${swift_bin}" infer \
  --model "${MODEL_PATH:-Qwen/Qwen3.5-9B}" \
  --adapters "${adapter}" \
  --use_hf true \
  --val_dataset "${val_data}" \
  --infer_backend transformers \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --max_batch_size 1 \
  --max_new_tokens "${MAX_NEW_TOKENS:-4096}" \
  --temperature 0 \
  --stream false \
  --result_path "${result_path}"

test "$(wc -l < "${result_path}")" -eq "$(wc -l < "${val_data}")"
echo "qwen35_t2_infer_status=PASS mode=$([[ "${smoke}" == "1" ]] && echo smoke || echo full) adapter=${adapter}"
