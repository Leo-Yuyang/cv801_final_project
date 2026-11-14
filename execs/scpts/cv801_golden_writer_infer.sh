#!/usr/bin/env bash
#SBATCH --job-name=cv801-golden-writer
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=16:00:00
#SBATCH --output=wkdrs/cv801-golden-test/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_root}/.env"
variant="${VARIANT:?Set VARIANT to t0, t1, or t2.}"
run_label="${RUN_LABEL:-${variant}}"
gpus="${GPUS:-8}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
swift_bin="${project_root}/inits/envs/cv801-qwen35/bin/swift"
model_path="${MODEL_PATH:-${project_root}/inits/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
input_root="${project_root}/datas/cv801_golden_test_inputs_v1"

[[ "${variant}" =~ ^(t0|t1|t2)$ ]] || { echo "VARIANT must be t0, t1, or t2." >&2; exit 1; }
[[ "${run_label}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "RUN_LABEL must contain lowercase letters, digits, or hyphens." >&2; exit 1; }
[[ "${gpus}" =~ ^[1-8]$ ]] || { echo "GPUS must be in [1, 8]." >&2; exit 1; }
[[ -f "${container}" && -x "${swift_bin}" && -f "${model_path}/config.json" ]] || {
  echo "Golden Writer runtime or frozen base model is missing." >&2
  exit 1
}

if [[ "${variant}" == "t2" ]]; then
  input_file="${input_root}/writer_t2.jsonl"
  audit_args=(--require-t2)
  adapter="${project_root}/wkdrs/cv801-qwen35/t2-train-382484/v0-20261109-131602/checkpoint-1197"
  audit_variant=t2
elif [[ "${variant}" == "t1" ]]; then
  input_file="${input_root}/writer_t0_t1.jsonl"
  audit_args=()
  adapter="${project_root}/wkdrs/cv801-qwen35/t1-train-358388/v0-20261104-192033/checkpoint-1197"
  audit_variant=t0_t1
else
  input_file="${input_root}/writer_t0_t1.jsonl"
  audit_args=()
  adapter=""
  audit_variant=t0_t1
fi
"${PYTHON_HOME}/bin/python" "${project_root}/CV801/scripts/prepare_golden_test_inputs.py" \
  audit --input-root "${input_root}" --expected-images 1000 "${audit_args[@]}"
[[ -f "${input_file}" ]] || { echo "Golden Writer input is missing: ${input_file}" >&2; exit 1; }
if [[ -n "${adapter}" && ! -f "${adapter}/adapter_model.safetensors" ]]; then
  echo "Frozen ${variant} adapter is missing: ${adapter}" >&2
  exit 1
fi

output_dir="${project_root}/wkdrs/cv801-golden-test/writer-${run_label}-${SLURM_JOB_ID}"
shard_dir="${output_dir}/shards"
result_path="${output_dir}/predictions.jsonl"
mkdir -p "${shard_dir}"
[[ ! -e "${result_path}" ]] || { echo "Refusing to overwrite ${result_path}" >&2; exit 1; }
printf '%s\n' "${model_path}" > "${output_dir}/model_path.txt"
if [[ -n "${adapter}" ]]; then
  printf '%s\n' "${adapter}" > "${output_dir}/adapter_path.txt"
fi

total_lines=$(wc -l < "${input_file}")
test "${total_lines}" -eq 1000
lines_per_shard=$(( (total_lines + gpus - 1) / gpus ))
split -d -a 2 -l "${lines_per_shard}" --additional-suffix=.jsonl \
  "${input_file}" "${shard_dir}/input-"
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
  infer_args=(
    --model "${model_path}"
    --use_hf true
    --val_dataset "${shard_input}"
    --infer_backend transformers
    --torch_dtype bfloat16
    --attn_impl sdpa
    --max_batch_size 1
    --max_new_tokens "${MAX_NEW_TOKENS:-4096}"
    --temperature 0
    --stream false
    --result_path "${shard_output}"
  )
  if [[ -n "${adapter}" ]]; then
    infer_args+=(--adapters "${adapter}")
  fi
  (
    apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
      "${container}" env ROCR_VISIBLE_DEVICES="${rank}" \
      "${swift_bin}" infer "${infer_args[@]}"
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
[[ "${failed}" -eq 0 ]] || { echo "At least one Writer inference shard failed." >&2; exit 1; }

for rank in $(seq 0 $((gpus - 1))); do
  printf -v shard_id '%02d' "${rank}"
  cat "${shard_dir}/predictions-${shard_id}.jsonl" >> "${result_path}"
done
test "$(wc -l < "${result_path}")" -eq "${total_lines}"
"${PYTHON_HOME}/bin/python" "${project_root}/CV801/scripts/prepare_golden_test_inputs.py" \
  predictions \
  --predictions "${result_path}" \
  --input-root "${input_root}" \
  --variant "${audit_variant}" \
  --expected-images 1000

echo "golden_writer_status=PASS variant=${variant} images=1000 result=${result_path} evaluation=NOT_RUN"
