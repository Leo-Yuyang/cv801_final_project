#!/usr/bin/env bash
#SBATCH --job-name=cv801-pixtral-review
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=wkdrs/cv801-agent-review/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
swift_bin="${project_root}/inits/envs/cv801-qwen35/bin/swift"
model_path="${MODEL_PATH:-${project_root}/inits/huggingface/pixtral-12b}"
input_file="${INPUT_FILE:?Set INPUT_FILE to a visual reviewer JSONL dataset.}"
mode="${MODE:-smoke}"
output_dir="${project_root}/wkdrs/cv801-agent-review/rc-pixtral-${mode}-${SLURM_JOB_ID}"
result_path="${output_dir}/predictions.jsonl"

[[ -f "${container}" && -x "${swift_bin}" && -f "${model_path}/config.json" ]] || {
  echo "Pixtral runtime or model is missing." >&2
  exit 1
}
[[ -f "${input_file}" ]] || { echo "Reviewer input is missing: ${input_file}" >&2; exit 1; }
mkdir -p "${output_dir}"
[[ ! -e "${result_path}" ]] || { echo "Refusing to append to ${result_path}." >&2; exit 1; }

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=true
export APPTAINERENV_IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
export APPTAINERENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${swift_bin}" infer \
  --model "${model_path}" \
  --model_type pixtral \
  --use_hf true \
  --val_dataset "${input_file}" \
  --infer_backend transformers \
  --device_map auto \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --max_batch_size 1 \
  --response_prefix '{' \
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
  --temperature 0 \
  --stream false \
  --result_path "${result_path}"

test "$(wc -l < "${result_path}")" -eq "$(wc -l < "${input_file}")"
echo "pixtral_reviewer_status=PASS mode=${mode} result=${result_path}"
