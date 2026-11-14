#!/usr/bin/env bash
#SBATCH --job-name=cv801-golden-det
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=16:00:00
#SBATCH --output=wkdrs/cv801-golden-test/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_root}/.env"
family="${MODEL_FAMILY:?Set MODEL_FAMILY to yolo, grounding_dino, or locateanything.}"
gpus="${GPUS:-8}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
detector_python="${project_root}/inits/envs/cv801-detectors/bin/python"
locate_python="${project_root}/inits/envs/cv801-locateanything/bin/python"
dataset_root="${project_root}/datas/cv801_detection_v1"
safe_root="${project_root}/datas/cv801_golden_test_inputs_v1"
safe_coco="${safe_root}/detector_input.coco.json"
run_label="${RUN_LABEL:-${SLURM_JOB_ID}}"
[[ "${run_label}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "RUN_LABEL contains unsafe characters." >&2; exit 1; }
output_dir="${project_root}/wkdrs/cv801-golden-test/detection-${family}-${run_label}"
predictions="${output_dir}/predictions.coco.json"

[[ "${gpus}" =~ ^[1-8]$ ]] || { echo "GPUS must be in [1, 8]." >&2; exit 1; }
[[ -f "${container}" && ( -x "${detector_python}" || -L "${detector_python}" ) && -f "${safe_coco}" ]] || {
  echo "Golden detector runtime or reference-free input is missing." >&2
  exit 1
}
"${PYTHON_HOME}/bin/python" "${project_root}/CV801/scripts/prepare_golden_test_inputs.py" \
  audit --input-root "${safe_root}" --expected-images 1000
mkdir -p "${output_dir}"
[[ ! -e "${predictions}" ]] || { echo "Refusing to overwrite ${predictions}" >&2; exit 1; }

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

case "${family}" in
  yolo)
    model="${MODEL_PATH:-${project_root}/wkdrs/cv801-yolo26x/train-357714/weights/best.pt}"
    [[ -f "${model}" ]] || { echo "Frozen YOLO26x model is missing." >&2; exit 1; }
    export APPTAINERENV_YOLO_CONFIG_DIR="${project_root}/inits/ultralytics"
    apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
      "${container}" "${detector_python}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node="${gpus}" \
      CV801/scripts/predict_yolo_coco.py \
      --model "${model}" \
      --ground-truth "${safe_coco}" \
      --dataset-root "${dataset_root}" \
      --output "${predictions}" \
      --image-size 960 \
      --confidence 0.001 \
      --nms-iou 0.7 \
      --max-detections 300 \
      --batch-size "${BATCH_SIZE:-1}"
    ;;
  grounding_dino)
    model="${MODEL_PATH:-${project_root}/wkdrs/cv801-grounding-dino/train-357715/final}"
    [[ -f "${model}/config.json" ]] || { echo "Frozen Grounding DINO model is missing." >&2; exit 1; }
    apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
      "${container}" "${detector_python}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node="${gpus}" \
      CV801/scripts/predict_grounding_dino_coco.py \
      --model "${model}" \
      --ground-truth "${safe_coco}" \
      --dataset-root "${dataset_root}" \
      --output "${predictions}" \
      --box-threshold "${BOX_THRESHOLD:-0.01}" \
      --text-threshold "${TEXT_THRESHOLD:-0.01}" \
      --nms-iou "${NMS_IOU:-0.6}"
    ;;
  locateanything)
    model="${MODEL_PATH:-${project_root}/wkdrs/cv801-locateanything/train-357716}"
    eagle_root="${project_root}/inits/Eagle/Embodied"
    raw_output="${output_dir}/raw_predictions.jsonl"
    [[ ( -x "${locate_python}" || -L "${locate_python}" ) && -f "${model}/config.json" ]] || {
      echo "Frozen LocateAnything model or runtime is missing." >&2
      exit 1
    }
    export APPTAINERENV_PYTHONPATH="${eagle_root}"
    apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
      "${container}" "${locate_python}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node="${gpus}" \
      CV801/scripts/predict_locateanything_coco.py \
      --model "${model}" \
      --ground-truth "${safe_coco}" \
      --dataset-root "${dataset_root}" \
      --output "${predictions}" \
      --raw-output "${raw_output}" \
      --generation-mode "${GENERATION_MODE:-hybrid}" \
      --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
      --short-side-size "${SHORT_SIDE_SIZE:-1024}"
    test "$(wc -l < "${raw_output}")" -eq 1000
    ;;
  *)
    echo "Unsupported MODEL_FAMILY: ${family}" >&2
    exit 1
    ;;
esac

"${PYTHON_HOME}/bin/python" - "${safe_coco}" "${predictions}" <<'PY'
import json
import sys
from pathlib import Path

safe = json.loads(Path(sys.argv[1]).read_text())
predictions = json.loads(Path(sys.argv[2]).read_text())
image_ids = {int(record["id"]) for record in safe["images"]}
category_ids = {int(record["id"]) for record in safe["categories"]}
for record in predictions:
    if int(record["image_id"]) not in image_ids or int(record["category_id"]) not in category_ids:
        raise SystemExit("Prediction contains an unknown image/category ID")
print(json.dumps({"status": "PASS", "predictions": len(predictions)}))
PY
echo "golden_detection_status=PASS family=${family} images=1000 predictions=${predictions} evaluation=NOT_RUN"
