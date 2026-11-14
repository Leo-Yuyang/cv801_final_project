#!/usr/bin/env bash
#SBATCH --job-name=cv801-pixtral-download
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=wkdrs/cv801-models/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
hf_bin="${project_root}/inits/envs/cv801-qwen35/bin/hf"
model_dir="${project_root}/inits/huggingface/pixtral-12b"

[[ -f "${container}" && -x "${hf_bin}" ]] || {
  echo "Qwen runtime container or Hugging Face client is missing." >&2
  exit 1
}
mkdir -p "${project_root}/wkdrs/cv801-models" "${model_dir}"
if [[ ! -f "${model_dir}/config.json" ]]; then
  apptainer exec --bind "${project_root}:${project_root}" --pwd "${project_root}" \
    "${container}" "${hf_bin}" download mistral-community/pixtral-12b \
    --local-dir "${model_dir}"
fi
[[ -f "${model_dir}/config.json" ]] || { echo "Pixtral config is missing after download." >&2; exit 1; }
test "$(find -L "${model_dir}" -maxdepth 1 -type f -name '*.safetensors' | wc -l)" -gt 0
echo "pixtral_download_status=PASS model=${model_dir}"
