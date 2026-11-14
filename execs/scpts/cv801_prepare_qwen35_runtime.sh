#!/usr/bin/env bash
#SBATCH --job-name=cv801-qwenrt
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=wkdrs/cv801-qwen35/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
qwen_env="${project_root}/inits/envs/cv801-qwen35"

[[ -f "${container}" ]] || { echo "Missing container: ${container}" >&2; exit 1; }
mkdir -p "${project_root}/inits/envs" "${project_root}/inits/pip-cache" \
  "${project_root}/inits/huggingface" "${project_root}/wkdrs/cv801-qwen35"

export APPTAINERENV_PIP_CACHE_DIR="${project_root}/inits/pip-cache"
export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" bash -s -- "${qwen_env}" <<'RUNTIME'
set -euo pipefail
qwen_env="$1"

if [[ ! -x "${qwen_env}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${qwen_env}"
fi
"${qwen_env}/bin/python" - <<'PY'
from pathlib import Path
import site
Path(site.getsitepackages()[0], "rocm_container.pth").write_text(
    "/opt/venv/lib/python3.12/site-packages\n",
    encoding="utf-8",
)
PY
"${qwen_env}/bin/python" -m pip install --disable-pip-version-check --upgrade \
  ms-swift==4.6.0 transformers==5.19.0 qwen-vl-utils==0.0.14 tensorboard

"${qwen_env}/bin/python" - <<'PY'
import torch
import transformers
import swift
print(f"torch={torch.__version__} hip={torch.version.hip}")
print(f"gpu_count={torch.cuda.device_count()}")
print(f"transformers={transformers.__version__}")
print(f"ms_swift={swift.__version__}")
assert torch.cuda.is_available()
assert transformers.__version__ == "5.19.0"
PY
"${qwen_env}/bin/swift" sft --help >/dev/null
RUNTIME

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${qwen_env}/bin/python" -m pip freeze \
  > "${project_root}/wkdrs/cv801-qwen35/requirements.freeze.txt"

echo "qwen35_runtime_status=PASS"
