#!/usr/bin/env bash
#SBATCH --job-name=cv801-hf-download
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=wkdrs/cv801-model-downloads/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }

model_id="${MODEL_ID:?Set MODEL_ID to the Hugging Face repository ID.}"
revision="${REVISION:?Set REVISION to an immutable Hugging Face commit.}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
qwen_env="${project_root}/inits/envs/cv801-qwen35"
cache_dir="${project_root}/inits/huggingface/hub"

[[ -f "${container}" ]] || { echo "Missing container: ${container}" >&2; exit 1; }
[[ -x "${qwen_env}/bin/hf" ]] || { echo "Missing Hugging Face CLI: ${qwen_env}/bin/hf" >&2; exit 1; }
mkdir -p "${cache_dir}" "${project_root}/wkdrs/cv801-model-downloads"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"

echo "model_id=${model_id}"
echo "revision=${revision}"
echo "download_started_utc=$(date -u +%FT%TZ)"

snapshot_path="$({
  apptainer exec --bind "${project_root}:${project_root}" --pwd "${project_root}" \
    "${container}" "${qwen_env}/bin/hf" download "${model_id}" \
    --revision "${revision}" --cache-dir "${cache_dir}" --max-workers 8 --quiet
} | tail -n 1)"

[[ -d "${snapshot_path}" ]] || { echo "Downloaded snapshot path is missing: ${snapshot_path}" >&2; exit 1; }
if find -L "${snapshot_path}" -type l -print -quit | grep -q .; then
  echo "Downloaded snapshot contains broken links: ${snapshot_path}" >&2
  exit 1
fi

weight_files="$(find "${snapshot_path}" -maxdepth 1 -type l -name '*.safetensors' | wc -l)"
[[ "${weight_files}" -gt 0 ]] || { echo "No safetensors weights found in ${snapshot_path}" >&2; exit 1; }

echo "snapshot_path=${snapshot_path}"
echo "weight_files=${weight_files}"
echo "download_finished_utc=$(date -u +%FT%TZ)"
echo "download_status=PASS"
