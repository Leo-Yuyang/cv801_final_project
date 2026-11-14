#!/usr/bin/env bash
#SBATCH --job-name=cv801-gdino-map
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=wkdrs/cv801-grounding-dino/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
python_bin="${project_root}/inits/envs/cv801-detectors/bin/python"
dataset_root="${project_root}/datas/cv801_detection_v1"
ground_truth="${dataset_root}/coco/instances_val.json"
model="${project_root}/wkdrs/cv801-grounding-dino/train-357715/final"
smoke="${SMOKE:-0}"
gpus="${GPUS:-8}"

[[ -f "${container}" && -f "${project_root}/inits/envs/cv801-detectors/pyvenv.cfg" && -f "${ground_truth}" \
   && -f "${model}/config.json" ]] || { echo "Grounding DINO evaluation inputs are missing." >&2; exit 1; }

if [[ "${smoke}" == "1" ]]; then
  run_name="coco-smoke-${SLURM_JOB_ID}"
  subset_args=(--max-images "${MAX_IMAGES:-16}")
  eval_args=(--only-predicted-images)
else
  run_name="coco-val-${SLURM_JOB_ID}"
  subset_args=()
  eval_args=()
fi
output_dir="${project_root}/wkdrs/cv801-grounding-dino/${run_name}"
predictions="${output_dir}/predictions.coco.json"
metrics="${output_dir}/metrics.coco.json"
mkdir -p "${output_dir}"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_OMP_NUM_THREADS=8

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${python_bin}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${gpus}" \
  CV801/scripts/predict_grounding_dino_coco.py \
  --model "${model}" \
  --ground-truth "${ground_truth}" \
  --dataset-root "${dataset_root}" \
  --output "${predictions}" \
  --box-threshold "${BOX_THRESHOLD:-0.01}" \
  --text-threshold "${TEXT_THRESHOLD:-0.01}" \
  --nms-iou "${NMS_IOU:-0.6}" \
  "${subset_args[@]}"

apptainer exec --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${python_bin}" CV801/scripts/evaluate_coco_predictions.py \
  --ground-truth "${ground_truth}" \
  --predictions "${predictions}" \
  --output "${metrics}" \
  --model-name "Grounding DINO Tiny fine-tuned" \
  --score-policy "native_model_score; classwise_nms_iou=${NMS_IOU:-0.6}" \
  "${eval_args[@]}"

echo "grounding_dino_coco_status=PASS mode=$([[ "${smoke}" == "1" ]] && echo smoke || echo full) metrics=${metrics}"
