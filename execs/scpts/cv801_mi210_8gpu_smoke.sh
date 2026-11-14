#!/usr/bin/env bash
#SBATCH --job-name=cv801-8gpu-smoke
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=wkdrs/cv801-mi210-smoke/slurm-%x-%j.out

set -euo pipefail

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "visible_devices=${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-unset}}"

command -v rocm-smi
command -v rocminfo

rocm-smi --showproductname --showmeminfo vram --showuse

gpu_count="$(rocm-smi --showproductname | awk '/Card Series:/ {count++} END {print count+0}')"
echo "detected_gpu_count=${gpu_count}"
if [[ "${gpu_count}" -ne 8 ]]; then
  echo "Expected 8 visible MI210 GPUs, detected ${gpu_count}." >&2
  exit 1
fi

rocm-smi --showtopo
rocminfo | awk '/^[[:space:]]*Name:[[:space:]]*gfx/ {print}' | sort | uniq -c

echo "smoke_status=PASS"
