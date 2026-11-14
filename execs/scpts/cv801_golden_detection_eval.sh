#!/usr/bin/env bash
#SBATCH --job-name=cv801-golden-map
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=wkdrs/cv801-golden-test/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
python_bin="${project_root}/inits/envs/cv801-detectors/bin/python"
ground_truth="${project_root}/datas/cv801_detection_v1/coco/instances_test.json"
predictions="${PREDICTIONS:?Set PREDICTIONS.}"
output="${OUTPUT:?Set OUTPUT.}"
model_name="${MODEL_NAME:?Set MODEL_NAME.}"
score_policy="${SCORE_POLICY:?Set SCORE_POLICY.}"
operating_score_threshold="${OPERATING_SCORE_THRESHOLD:?Set OPERATING_SCORE_THRESHOLD from validation.}"

missing=0
for required in "${container}" "${ground_truth}" "${predictions}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Golden detection evaluation input is missing: ${required}" >&2
    missing=1
  fi
done
if [[ ! -x "${python_bin}" && ! -L "${python_bin}" ]]; then
  echo "Golden detection evaluation Python is missing: ${python_bin}" >&2
  missing=1
fi
[[ "${missing}" -eq 0 ]] || exit 1
apptainer exec --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${python_bin}" CV801/scripts/evaluate_coco_predictions.py \
  --ground-truth "${ground_truth}" \
  --predictions "${predictions}" \
  --output "${output}" \
  --model-name "${model_name}" \
  --score-policy "${score_policy}" \
  --operating-score-threshold "${operating_score_threshold}"
echo "golden_detection_eval_status=PASS metrics=${output}"
