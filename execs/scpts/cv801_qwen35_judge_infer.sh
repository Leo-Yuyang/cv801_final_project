#!/usr/bin/env bash
#SBATCH --job-name=cv801-judge
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=wkdrs/cv801-agent-review/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }

container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
swift_bin="${project_root}/inits/envs/cv801-qwen35/bin/swift"
model_path="${MODEL_PATH:-${project_root}/inits/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
input_file="${INPUT_FILE:?Set INPUT_FILE to a Judge JSONL dataset.}"
stage="${STAGE:-judge}"
mode="${MODE:-smoke}"

[[ "${stage}" == "judge" || "${stage}" == "recheck" ]] || { echo "STAGE must be judge or recheck." >&2; exit 1; }
[[ -f "${container}" && -x "${swift_bin}" && -f "${model_path}/config.json" ]] || {
  echo "Judge runtime or model snapshot is missing." >&2
  exit 1
}
[[ -f "${input_file}" ]] || { echo "Judge input is missing: ${input_file}" >&2; exit 1; }

run_name="${stage}-${mode}-${SLURM_JOB_ID}"
output_dir="${project_root}/wkdrs/cv801-agent-review/${run_name}"
result_path="${output_dir}/predictions.jsonl"
mkdir -p "${output_dir}"
[[ ! -e "${result_path}" ]] || { echo "Refusing to append to existing ${result_path}" >&2; exit 1; }

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=true
export APPTAINERENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${swift_bin}" infer \
  --model "${model_path}" \
  --model_type qwen3_5 \
  --use_hf true \
  --val_dataset "${input_file}" \
  --infer_backend transformers \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --max_batch_size 1 \
  --response_prefix '{' \
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
  --temperature 0 \
  --stream false \
  --result_path "${result_path}"

test "$(wc -l < "${result_path}")" -eq "$(wc -l < "${input_file}")"
echo "judge_infer_status=PASS stage=${stage} mode=${mode} model=${model_path} result=${result_path}"
