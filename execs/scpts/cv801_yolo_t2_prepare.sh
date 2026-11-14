#!/usr/bin/env bash
#SBATCH --job-name=cv801-t2-data
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
# Eight independent ROCm inference workers transiently peak near 500 GiB of
# host memory while loading.  Leave headroom for allocator peaks and merging.
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=wkdrs/cv801-t2/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
python_bin="${project_root}/inits/envs/cv801-detectors/bin/python"
detection_root="${project_root}/datas/cv801_detection_v1"
t1_root="${project_root}/datas/cv801_vlm_v1"
t2_root="${project_root}/datas/cv801_vlm_t2_yolo26x_v1"
model="${project_root}/wkdrs/cv801-yolo26x/train-357714/weights/best.pt"
smoke="${SMOKE:-0}"
gpus="${GPUS:-8}"

[[ -f "${container}" && -f "${project_root}/inits/envs/cv801-detectors/pyvenv.cfg" \
   && -f "${model}" && -f "${t1_root}/summary.json" ]] || {
  echo "YOLO model, detector runtime, or T1 dataset is missing." >&2
  exit 1
}
mkdir -p "${project_root}/wkdrs/cv801-t2"

if [[ "${smoke}" == "1" ]]; then
  run_name="prepare-smoke-${SLURM_JOB_ID}"
  train_subset=(--max-images "${SMOKE_TRAIN_IMAGES:-16}")
  val_subset=(--max-images "${SMOKE_VAL_IMAGES:-8}")
  eval_subset=(--only-predicted-images)
else
  run_name="prepare-${SLURM_JOB_ID}"
  train_subset=()
  val_subset=()
  eval_subset=()
  [[ ! -e "${t2_root}/summary.json" ]] || {
    echo "Refusing to overwrite frozen T2 dataset ${t2_root}" >&2
    exit 1
  }
fi
output_dir="${project_root}/wkdrs/cv801-t2/${run_name}"
train_predictions="${output_dir}/yolo-train.coco.json"
val_predictions="${output_dir}/yolo-val.coco.json"
val_metrics="${output_dir}/yolo-val-metrics.coco.json"
mkdir -p "${output_dir}"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_YOLO_CONFIG_DIR="${project_root}/inits/ultralytics"
export APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

for split in val train; do
  if [[ "${split}" == "val" ]]; then
    predictions="${val_predictions}"
    subset_args=("${val_subset[@]}")
  else
    predictions="${train_predictions}"
    subset_args=("${train_subset[@]}")
  fi
  apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
    "${container}" "${python_bin}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${gpus}" \
    CV801/scripts/predict_yolo_coco.py \
    --model "${model}" \
    --ground-truth "${detection_root}/coco/instances_${split}.json" \
    --dataset-root "${detection_root}" \
    --output "${predictions}" \
    --image-size 960 \
    --confidence 0.001 \
    --nms-iou 0.7 \
    --max-detections 300 \
    --batch-size "${BATCH_SIZE:-1}" \
    "${subset_args[@]}"
done

apptainer exec --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${python_bin}" CV801/scripts/evaluate_coco_predictions.py \
  --ground-truth "${detection_root}/coco/instances_val.json" \
  --predictions "${val_predictions}" \
  --output "${val_metrics}" \
  --model-name "YOLO26x fine-tuned" \
  --score-policy "native_model_score; nms_iou=0.7; max_det=300" \
  "${eval_subset[@]}"

if [[ "${smoke}" != "1" ]]; then
  apptainer exec --bind "${project_root}:${project_root}" --pwd "${project_root}" \
    "${container}" "${python_bin}" CV801/scripts/prepare_t2_dataset.py \
    --t1-root "${t1_root}" \
    --detection-root "${detection_root}" \
    --train-predictions "${train_predictions}" \
    --val-predictions "${val_predictions}" \
    --output-root "${t2_root}" \
    --locator-model "${model}" \
    --locator-metrics "${val_metrics}" \
    --score-threshold "${T2_SCORE_THRESHOLD:-0.25}" \
    --max-proposals "${T2_MAX_PROPOSALS:-20}" \
    --coverage-iou 0.5
fi

echo "cv801_t2_data_status=PASS mode=$([[ "${smoke}" == "1" ]] && echo smoke || echo full) output=${output_dir}"
