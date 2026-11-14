#!/usr/bin/env bash
#SBATCH --job-name=cv801-gdino
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=wkdrs/cv801-grounding-dino/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
python_bin="${project_root}/inits/envs/cv801-detectors/bin/python"
dataset_root="${project_root}/datas/cv801_detection_v1"
training_script="${project_root}/CV801/scripts/train_grounding_dino.py"
smoke="${SMOKE:-0}"
gpus="${GPUS:-8}"

[[ -f "${container}" && -f "${project_root}/inits/envs/cv801-detectors/pyvenv.cfg" && -f "${dataset_root}/dataset.yaml" ]] || {
  echo "Training runtime or dataset is missing; run cv801_prepare_training_runtime.sh first." >&2
  exit 1
}
mkdir -p "${project_root}/wkdrs/cv801-grounding-dino"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export APPTAINERENV_OMP_NUM_THREADS=8

extra_args=()
if [[ "${smoke}" == "1" ]]; then
  epochs=1
  gradient_accumulation=1
  workers=0
  run_name="smoke-${SLURM_JOB_ID}"
  extra_args+=(--max-train-samples 32 --max-val-samples 16)
else
  epochs="${EPOCHS:-12}"
  gradient_accumulation="${GRADIENT_ACCUMULATION:-2}"
  workers="${WORKERS:-8}"
  run_name="train-${SLURM_JOB_ID}"
fi
output_dir="${project_root}/wkdrs/cv801-grounding-dino/${run_name}"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${python_bin}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${gpus}" \
  "${training_script}" \
  --dataset-root "${dataset_root}" \
  --output-dir "${output_dir}" \
  --model "${MODEL_PATH:-IDEA-Research/grounding-dino-tiny}" \
  --epochs "${epochs}" \
  --batch-size 1 \
  --gradient-accumulation "${gradient_accumulation}" \
  --learning-rate "${LEARNING_RATE:-2e-5}" \
  --workers "${workers}" \
  "${extra_args[@]}"

echo "grounding_dino_status=PASS mode=$([[ "${smoke}" == "1" ]] && echo smoke || echo full)"
