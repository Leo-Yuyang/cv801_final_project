#!/usr/bin/env bash

set -euo pipefail

package_dir="${1:?Usage: publish_hf.sh <release-package-directory>}"
package_dir="$(cd "${package_dir}" && pwd)"
map_file="${package_dir}/repo-map.env"
[[ -f "${package_dir}/release_inventory.json" && -d "${package_dir}/hf" ]] || {
  echo "Invalid release package: ${package_dir}" >&2
  exit 1
}
[[ -f "${map_file}" ]] || {
  echo "Copy ${package_dir}/repo-map.env.example to ${map_file} and fill it first." >&2
  exit 1
}

# shellcheck disable=SC1090
source "${map_file}"

execute="${EXECUTE:-0}"
namespace="${HF_NAMESPACE:-}"
private="${HF_PRIVATE:-0}"
allow_locateanything="${ALLOW_LOCATEANYTHING_WEIGHTS:-NO}"
hf_bin="${HF_BIN:-$(command -v hf || true)}"
[[ -n "${hf_bin}" && -x "${hf_bin}" ]] || { echo "hf CLI is not available." >&2; exit 1; }

if [[ "${execute}" == "1" && -z "${namespace}" ]]; then
  echo "HF_NAMESPACE is required for upload." >&2
  exit 1
fi
if [[ "${execute}" == "1" && "${allow_locateanything}" != "YES" ]]; then
  echo "Set ALLOW_LOCATEANYTHING_WEIGHTS=YES only after checking its redistribution terms." >&2
  exit 1
fi

display_namespace="${namespace:-<HF_NAMESPACE>}"
private_args=()
if [[ "${private}" == "1" ]]; then
  private_args+=(--private)
fi

repo_keys=(
  models
  data
)
repo_types=(model dataset)
repo_names=(
  "${HF_MODELS_REPO:-cv801_models}"
  "${HF_DATA_REPO:-cv801_data}"
)

echo "hf_publish_mode=$([[ "${execute}" == "1" ]] && echo execute || echo dry-run)"
echo "hf_visibility=$([[ "${private}" == "1" ]] && echo private || echo public)"

if [[ "${execute}" == "1" ]]; then
  "${hf_bin}" auth whoami >/dev/null
fi

for index in "${!repo_keys[@]}"; do
  key="${repo_keys[$index]}"
  repo_type="${repo_types[$index]}"
  repo_name="${repo_names[$index]}"
  local_dir="${package_dir}/hf/${key}"
  [[ -d "${local_dir}" && -f "${local_dir}/README.md" && -f "${local_dir}/MANIFEST.sha256" ]] || {
    echo "Prepared HF directory is incomplete: ${local_dir}" >&2
    exit 1
  }
  repo_id="${display_namespace}/${repo_name}"
  echo "hf_upload repo=${repo_id} type=${repo_type} source=${local_dir}"
  if [[ "${execute}" == "1" ]]; then
    (cd "${local_dir}" && sha256sum -c MANIFEST.sha256)
    "${hf_bin}" upload-large-folder "${namespace}/${repo_name}" "${local_dir}" \
      --repo-type "${repo_type}" "${private_args[@]}" --num-workers "${HF_UPLOAD_WORKERS:-8}"
  fi
done

echo "hf_publish_status=$([[ "${execute}" == "1" ]] && echo COMPLETE || echo DRY_RUN_PASS)"
