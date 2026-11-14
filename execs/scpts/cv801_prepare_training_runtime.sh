#!/usr/bin/env bash
#SBATCH --job-name=cv801-runtime
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=wkdrs/cv801-runtime/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
detector_env="${project_root}/inits/envs/cv801-detectors"
locate_env="${project_root}/inits/envs/cv801-locateanything"
eagle_root="${project_root}/inits/Eagle"
eagle_patch="${project_root}/CV801/patches/eagle_mi210_sdpa.patch"

[[ -f "${container}" ]] || { echo "Missing container: ${container}" >&2; exit 1; }
[[ -d "${eagle_root}/.git" ]] || { echo "Missing Eagle checkout: ${eagle_root}" >&2; exit 1; }
mkdir -p "${project_root}/inits/envs" "${project_root}/inits/pip-cache" \
  "${project_root}/inits/huggingface" "${project_root}/inits/ultralytics" \
  "${project_root}/wkdrs/cv801-runtime"

if git -C "${eagle_root}" apply --reverse --check "${eagle_patch}" >/dev/null 2>&1; then
  echo "Eagle MI210 SDPA patch already applied."
elif git -C "${eagle_root}" apply --check "${eagle_patch}"; then
  git -C "${eagle_root}" apply "${eagle_patch}"
else
  echo "Eagle checkout does not match the pinned SDPA patch." >&2
  exit 1
fi

export APPTAINERENV_PIP_CACHE_DIR="${project_root}/inits/pip-cache"
export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_YOLO_CONFIG_DIR="${project_root}/inits/ultralytics"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" bash -s -- "${detector_env}" "${locate_env}" "${eagle_root}" <<'RUNTIME'
set -euo pipefail
detector_env="$1"
locate_env="$2"
eagle_root="$3"

if [[ ! -x "${detector_env}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${detector_env}"
fi
"${detector_env}/bin/python" - <<'PY'
from pathlib import Path
import site
Path(site.getsitepackages()[0], "rocm_container.pth").write_text(
    "/opt/venv/lib/python3.12/site-packages\n",
    encoding="utf-8",
)
PY
"${detector_env}/bin/python" -m pip install --disable-pip-version-check --upgrade \
  ultralytics==8.4.202 transformers==4.57.1 tokenizers==0.22.0 accelerate==1.5.2 \
  tensorboard scipy pycocotools

if [[ ! -x "${locate_env}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${locate_env}"
fi
"${locate_env}/bin/python" - <<'PY'
from pathlib import Path
import site
Path(site.getsitepackages()[0], "rocm_container.pth").write_text(
    "/opt/venv/lib/python3.12/site-packages\n",
    encoding="utf-8",
)
PY
"${locate_env}/bin/python" -m pip install --disable-pip-version-check --upgrade \
  transformers==4.57.1 tokenizers==0.22.0 accelerate==1.5.2 peft==0.12.0 \
  sentencepiece==0.2.0 shortuuid wandb pydantic==2.7.1 'numpy>=1.25,<2' \
  scikit-learn lmdb einops einops-exts timm av filetype \
  bitstring==4.3.1 ebmlite sortedcontainers scipy datasets==2.21.0 pyarrow==17.0.0 python-dotenv decord scikit-image imagehash \
  opencv-python-headless==4.11.0.86 tensorboard
"${locate_env}/bin/python" -m pip install --disable-pip-version-check --no-deps \
  liger_kernel==0.3.1
"${locate_env}/bin/python" -m pip install --disable-pip-version-check --no-deps \
  --editable "${eagle_root}/Embodied"

"${detector_env}/bin/python" - <<'PY'
import torch
import transformers
import ultralytics
print(f"torch={torch.__version__} hip={torch.version.hip}")
print(f"gpu_count={torch.cuda.device_count()}")
print(f"transformers={transformers.__version__}")
print(f"ultralytics={ultralytics.__version__}")
assert torch.cuda.is_available()
PY

"${locate_env}/bin/python" - <<'PY'
import torch
import transformers
import eaglevl
print(f"locateanything_import=PASS torch={torch.__version__} transformers={transformers.__version__}")
PY
RUNTIME

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${detector_env}/bin/python" -m pip freeze \
  > "${project_root}/wkdrs/cv801-runtime/detector-requirements.freeze.txt"
apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" "${locate_env}/bin/python" -m pip freeze \
  > "${project_root}/wkdrs/cv801-runtime/locateanything-requirements.freeze.txt"

echo "runtime_status=PASS"
