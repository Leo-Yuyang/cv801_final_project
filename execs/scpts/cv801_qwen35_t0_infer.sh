#!/usr/bin/env bash
#SBATCH --job-name=cv801-qwen-t0
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=wkdrs/cv801-qwen35/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
swift_bin="${project_root}/inits/envs/cv801-qwen35/bin/swift"
dataset_root="${project_root}/datas/cv801_vlm_v1"
smoke="${SMOKE:-0}"

[[ -f "${container}" && -f "${project_root}/inits/envs/cv801-qwen35/pyvenv.cfg" \
   && -f "${swift_bin}" && -f "${dataset_root}/t1_val.jsonl" ]] || {
  echo "Qwen runtime or validation dataset is missing." >&2
  exit 1
}

if [[ "${smoke}" == "1" ]]; then
  val_data="${dataset_root}/t1_val_smoke.jsonl"
  run_name="t0-smoke-${SLURM_JOB_ID}"
else
  val_data="${dataset_root}/t1_val.jsonl"
  run_name="t0-val-${SLURM_JOB_ID}"
fi
output_dir="${project_root}/wkdrs/cv801-qwen35/${run_name}"
result_path="${output_dir}/predictions.jsonl"
mkdir -p "${output_dir}"
[[ ! -e "${result_path}" ]] || { echo "Refusing to append to existing ${result_path}" >&2; exit 1; }

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=true
export APPTAINERENV_IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
export APPTAINERENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${swift_bin}" infer \
  --model "${MODEL_PATH:-Qwen/Qwen3.5-9B}" \
  --use_hf true \
  --val_dataset "${val_data}" \
  --infer_backend transformers \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --max_batch_size 1 \
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
  --temperature 0 \
  --stream false \
  --result_path "${result_path}"

test "$(wc -l < "${result_path}")" -eq "$(wc -l < "${val_data}")"
echo "qwen35_t0_status=PASS mode=$([[ "${smoke}" == "1" ]] && echo smoke || echo full) result=${result_path}"
