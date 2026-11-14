#!/usr/bin/env bash
#SBATCH --job-name=cv801-qwen-t1i8
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=16:00:00
#SBATCH --output=wkdrs/cv801-qwen35/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
swift_bin="${project_root}/inits/envs/cv801-qwen35/bin/swift"
dataset_root="${project_root}/datas/cv801_vlm_v1"
adapter="${ADAPTER_PATH:-}"
train_root=""
gpus="${GPUS:-8}"

[[ "${gpus}" =~ ^[1-8]$ ]] || { echo "GPUS must be in [1, 8]." >&2; exit 1; }
[[ -f "${container}" && -f "${swift_bin}" && -f "${dataset_root}/t1_val.jsonl" ]] || {
  echo "Qwen runtime or validation data is missing." >&2
  exit 1
}

if [[ -z "${adapter}" ]]; then
  train_job_id="${TRAIN_JOB_ID:?Set ADAPTER_PATH or TRAIN_JOB_ID to the completed formal T1 job}"
  train_root="${project_root}/wkdrs/cv801-qwen35/t1-train-${train_job_id}"
  [[ -d "${train_root}" ]] || { echo "T1 training output is missing: ${train_root}" >&2; exit 1; }
  latest_state=$(
    find "${train_root}" -mindepth 2 -maxdepth 3 -type f -name trainer_state.json \
      | sort -V | tail -n 1
  )
  if [[ -n "${latest_state}" ]]; then
    adapter=$(jq -r '.best_model_checkpoint // empty' "${latest_state}")
  fi
  if [[ -z "${adapter}" || ! -f "${adapter}/adapter_config.json" ]]; then
    adapter=$(
      find "${train_root}" -mindepth 2 -maxdepth 3 -type f -name adapter_config.json -printf '%h\n' \
        | sort -V | tail -n 1
    )
  fi
fi
[[ -f "${adapter}/adapter_config.json" && -f "${adapter}/adapter_model.safetensors" ]] || {
  echo "No complete LoRA checkpoint found at ${adapter}." >&2
  exit 1
}

val_data="${dataset_root}/t1_val.jsonl"
output_dir="${project_root}/wkdrs/cv801-qwen35/t1-val-sharded-${SLURM_JOB_ID}"
shard_dir="${output_dir}/shards"
result_path="${output_dir}/predictions.jsonl"
mkdir -p "${shard_dir}"
[[ ! -e "${result_path}" ]] || { echo "Refusing to overwrite ${result_path}" >&2; exit 1; }
printf '%s\n' "${adapter}" > "${output_dir}/adapter_path.txt"

# Equal-sized contiguous chunks preserve validation ordering after concatenation.
total_lines=$(wc -l < "${val_data}")
lines_per_shard=$(( (total_lines + gpus - 1) / gpus ))
split -d -a 2 -l "${lines_per_shard}" --additional-suffix=.jsonl \
  "${val_data}" "${shard_dir}/input-"
test "$(find "${shard_dir}" -maxdepth 1 -type f -name 'input-*.jsonl' | wc -l)" -eq "${gpus}"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=true
export APPTAINERENV_IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
export APPTAINERENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

pids=()
for rank in $(seq 0 $((gpus - 1))); do
  printf -v shard_id '%02d' "${rank}"
  shard_input="${shard_dir}/input-${shard_id}.jsonl"
  shard_output="${shard_dir}/predictions-${shard_id}.jsonl"
  (
    apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
      "${container}" env ROCR_VISIBLE_DEVICES="${rank}" \
      "${swift_bin}" infer \
      --model "${MODEL_PATH:-Qwen/Qwen3.5-9B}" \
      --adapters "${adapter}" \
      --use_hf true \
      --val_dataset "${shard_input}" \
      --infer_backend transformers \
      --torch_dtype bfloat16 \
      --attn_impl sdpa \
      --max_batch_size 1 \
      --max_new_tokens "${MAX_NEW_TOKENS:-4096}" \
      --temperature 0 \
      --stream false \
      --result_path "${shard_output}"
    test "$(wc -l < "${shard_output}")" -eq "$(wc -l < "${shard_input}")"
  ) > "${shard_dir}/rank-${shard_id}.out" 2> "${shard_dir}/rank-${shard_id}.err" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
[[ "${failed}" -eq 0 ]] || { echo "At least one inference shard failed." >&2; exit 1; }

for rank in $(seq 0 $((gpus - 1))); do
  printf -v shard_id '%02d' "${rank}"
  cat "${shard_dir}/predictions-${shard_id}.jsonl" >> "${result_path}"
done
test "$(wc -l < "${result_path}")" -eq "$(wc -l < "${val_data}")"

echo "qwen35_t1_infer_status=PASS mode=full-sharded gpus=${gpus} adapter=${adapter} result=${result_path}"
