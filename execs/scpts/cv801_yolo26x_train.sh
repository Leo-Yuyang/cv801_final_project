#!/usr/bin/env bash
#SBATCH --job-name=cv801-yolo26x
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=wkdrs/cv801-yolo26x/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
python_bin="${project_root}/inits/envs/cv801-detectors/bin/python"
dataset_yaml="${project_root}/datas/cv801_detection_v1/dataset.yaml"
model_dir="${project_root}/inits/yolo"
model_path="${MODEL_PATH:-${model_dir}/yolo26x.pt}"
smoke="${SMOKE:-0}"
gpus="${GPUS:-8}"

[[ -f "${container}" && -f "${project_root}/inits/envs/cv801-detectors/pyvenv.cfg" && -f "${dataset_yaml}" ]] || {
  echo "Training runtime or dataset is missing; run cv801_prepare_training_runtime.sh first." >&2
  exit 1
}
mkdir -p "${model_dir}" "${project_root}/wkdrs/cv801-yolo26x"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TORCH_HOME="${project_root}/inits/torch-cache"
export APPTAINERENV_YOLO_CONFIG_DIR="${project_root}/inits/ultralytics"
export APPTAINERENV_NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export APPTAINERENV_OMP_NUM_THREADS=8

if [[ ! -f "${model_path}" ]]; then
  apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${model_dir}" \
    "${container}" "${python_bin}" -c 'from ultralytics import YOLO; YOLO("yolo26x.pt")'
fi

if [[ "${smoke}" == "1" ]]; then
  epochs=1
  image_size=640
  batch=8
  workers=2
  fraction=0.01
  validate=False
  run_name="smoke-${SLURM_JOB_ID}"
else
  epochs="${EPOCHS:-100}"
  image_size="${IMAGE_SIZE:-960}"
  batch="${BATCH_SIZE:-64}"
  workers="${WORKERS:-8}"
  fraction=1.0
  validate=True
  run_name="train-${SLURM_JOB_ID}"
fi
if [[ "${gpus}" == "1" ]]; then
  devices=0
else
  devices="$(seq -s, 0 "$((gpus - 1))")"
fi

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${project_root}/inits/envs/cv801-detectors/bin/yolo" detect train \
  model="${model_path}" \
  data="${dataset_yaml}" \
  epochs="${epochs}" \
  imgsz="${image_size}" \
  batch="${batch}" \
  device="${devices}" \
  workers="${workers}" \
  fraction="${fraction}" \
  val="${validate}" \
  pretrained=True \
  amp=True \
  cos_lr=True \
  patience=30 \
  close_mosaic=10 \
  save_period=10 \
  seed=20261104 \
  deterministic=True \
  project="${project_root}/wkdrs/cv801-yolo26x" \
  name="${run_name}" \
  exist_ok=False

echo "yolo_status=PASS mode=$([[ "${smoke}" == "1" ]] && echo smoke || echo full)"
