#!/usr/bin/env bash
#SBATCH --job-name=cv801-rocm-probe
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=wkdrs/cv801-rocm-probe/slurm-%x-%j.out

set -euo pipefail

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "path=${PATH}"

rocm-smi --showdriverversion
rocminfo | awk '/^[[:space:]]*Name:[[:space:]]*gfx/ {print}'

for version_file in /opt/rocm/.info/version /opt/rocm/.info/version-dev /opt/rocm/.info/version-utils; do
  if [[ -r "${version_file}" ]]; then
    echo "${version_file}=$(<"${version_file}")"
  fi
done

for command_name in python python3 apptainer singularity docker; do
  command -v "${command_name}" || true
done

python3 - <<'PY' || true
try:
    import torch
except ImportError:
    print("torch=not-installed")
else:
    print(f"torch={torch.__version__}")
    print(f"torch_hip={torch.version.hip}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")
PY

echo "probe_status=PASS"
