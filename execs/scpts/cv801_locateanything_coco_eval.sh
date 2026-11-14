#!/usr/bin/env bash
#SBATCH --job-name=cv801-locany-map
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=wkdrs/cv801-locateanything/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
locate_python="${project_root}/inits/envs/cv801-locateanything/bin/python"
eval_python="${project_root}/inits/envs/cv801-detectors/bin/python"
eagle_root="${project_root}/inits/Eagle/Embodied"
dataset_root="${project_root}/datas/cv801_detection_v1"
ground_truth="${dataset_root}/coco/instances_val.json"
model="${MODEL_PATH:-${project_root}/wkdrs/cv801-locateanything/train-357716}"
smoke="${SMOKE:-0}"
gpus="${GPUS:-8}"

[[ -f "${container}" && -f "${project_root}/inits/envs/cv801-locateanything/pyvenv.cfg" \
   && -f "${project_root}/inits/envs/cv801-detectors/pyvenv.cfg" \
   && -f "${ground_truth}" && -f "${model}/config.json" ]] || {
  echo "LocateAnything evaluation inputs are missing." >&2
  exit 1
}

if [[ "${smoke}" == "1" ]]; then
  run_name="coco-smoke-${SLURM_JOB_ID}"
  max_images="${MAX_IMAGES:-8}"
  subset_args=(--max-images "${max_images}")
  eval_args=(--only-predicted-images)
else
  run_name="coco-val-${SLURM_JOB_ID}"
  max_images=710
  subset_args=()
  eval_args=()
fi
output_dir="${project_root}/wkdrs/cv801-locateanything/${run_name}"
predictions="${output_dir}/predictions.coco.json"
raw_output="${output_dir}/raw_predictions.jsonl"
metrics="${output_dir}/metrics.coco.json"
mkdir -p "${output_dir}"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=true
export APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_PYTHONPATH="${eagle_root}"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${locate_python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${gpus}" \
  CV801/scripts/predict_locateanything_coco.py \
  --model "${model}" \
  --ground-truth "${ground_truth}" \
  --dataset-root "${dataset_root}" \
  --output "${predictions}" \
  --raw-output "${raw_output}" \
  --generation-mode "${GENERATION_MODE:-hybrid}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
  --short-side-size "${SHORT_SIDE_SIZE:-1024}" \
  "${subset_args[@]}"

test "$(wc -l < "${raw_output}")" -eq "${max_images}"
apptainer exec --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${eval_python}" CV801/scripts/evaluate_coco_predictions.py \
  --ground-truth "${ground_truth}" \
  --predictions "${predictions}" \
  --output "${metrics}" \
  --model-name "LocateAnything 3B fine-tuned" \
  --score-policy "constant_0.2_no_native_confidence; generation_order_ties" \
  "${eval_args[@]}"

echo "locateanything_coco_status=PASS mode=$([[ "${smoke}" == "1" ]] && echo smoke || echo full) metrics=${metrics}"
