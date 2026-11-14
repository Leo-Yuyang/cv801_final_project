#!/usr/bin/env bash
#SBATCH --job-name=cv801-vlm-eval
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_root}/.env"
predictions="${PREDICTIONS:?Set PREDICTIONS to an ms-swift result JSONL path}"
output="${OUTPUT:-$(dirname "${predictions}")/metrics.json}"

[[ -f "${predictions}" ]] || { echo "Missing predictions: ${predictions}" >&2; exit 1; }
"${PYTHON_HOME}/bin/python" "${project_root}/CV801/scripts/evaluate_vlm_predictions.py" \
  "${predictions}" --output "${output}"
echo "cv801_vlm_eval_status=PASS output=${output}"
