#!/usr/bin/env bash
#SBATCH --job-name=cv801-revision
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=wkdrs/cv801-agent-review/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }

container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
swift_bin="${project_root}/inits/envs/cv801-qwen35/bin/swift"
model_path="${MODEL_PATH:-${project_root}/inits/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
train_job_id="${TRAIN_JOB_ID:-382484}"
train_root="${project_root}/wkdrs/cv801-qwen35/t2-train-${train_job_id}"
input_file="${INPUT_FILE:?Set INPUT_FILE to a Writer revision JSONL dataset.}"
configuration="${CONFIGURATION:?Set CONFIGURATION to rc, rd, rc_rd, or full.}"
mode="${MODE:-smoke}"

[[ "${configuration}" =~ ^(rc|rd|rc_rd|full)$ ]] || { echo "Invalid CONFIGURATION." >&2; exit 1; }
[[ -f "${container}" && -x "${swift_bin}" && -f "${model_path}/config.json" && -d "${train_root}" ]] || {
  echo "Writer runtime, base model, or T2 training output is missing." >&2
  exit 1
}
[[ -f "${input_file}" ]] || { echo "Revision input is missing: ${input_file}" >&2; exit 1; }

latest_state=$(find "${train_root}" -mindepth 2 -maxdepth 3 -type f -name trainer_state.json | sort -V | tail -n 1)
adapter=""
if [[ -n "${latest_state}" ]]; then
  adapter=$(jq -r '.best_model_checkpoint // empty' "${latest_state}")
fi
if [[ -z "${adapter}" || ! -f "${adapter}/adapter_config.json" ]]; then
  adapter=$(find "${train_root}" -mindepth 2 -maxdepth 3 -type f -name adapter_config.json -printf '%h\n' | sort -V | tail -n 1)
fi
[[ -n "${adapter}" && -f "${adapter}/adapter_model.safetensors" ]] || {
  echo "No complete T2 LoRA checkpoint found under ${train_root}." >&2
  exit 1
}

run_name="revision-${configuration}-${mode}-${SLURM_JOB_ID}"
output_dir="${project_root}/wkdrs/cv801-agent-review/${run_name}"
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
  --model "${model_path}" \
  --adapters "${adapter}" \
  --use_hf true \
  --val_dataset "${input_file}" \
  --infer_backend transformers \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --max_batch_size 1 \
  --max_new_tokens "${MAX_NEW_TOKENS:-4096}" \
  --temperature 0 \
  --stream false \
  --result_path "${result_path}"

test "$(wc -l < "${result_path}")" -eq "$(wc -l < "${input_file}")"
echo "revision_infer_status=PASS configuration=${configuration} mode=${mode} adapter=${adapter} result=${result_path}"
