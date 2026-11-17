#!/usr/bin/env bash

set -euo pipefail

package_dir="${1:?Usage: publish_github.sh <release-package-directory>}"
package_dir="$(cd "${package_dir}" && pwd)"
map_file="${package_dir}/repo-map.env"
github_dir="${package_dir}/github"
[[ -d "${github_dir}" && -f "${github_dir}/README.md" && -f "${github_dir}/MANIFEST.sha256" ]] || {
  echo "Prepared GitHub directory is incomplete: ${github_dir}" >&2
  exit 1
}
[[ -f "${map_file}" ]] || {
  echo "Copy ${package_dir}/repo-map.env.example to ${map_file} and fill it first." >&2
  exit 1
}

# shellcheck disable=SC1090
source "${map_file}"
execute="${EXECUTE:-0}"
repo_url="${GITHUB_REPO_URL:-}"

echo "github_publish_mode=$([[ "${execute}" == "1" ]] && echo execute || echo dry-run)"
echo "github_source=${github_dir}"
echo "github_target=${repo_url:-<GITHUB_REPO_URL>}"
echo "github_files=$(find "${github_dir}" -type f | wc -l)"

if [[ "${execute}" != "1" ]]; then
  echo "github_publish_status=DRY_RUN_PASS"
  exit 0
fi

[[ -n "${repo_url}" ]] || { echo "GITHUB_REPO_URL is required for upload." >&2; exit 1; }
(cd "${github_dir}" && sha256sum -c MANIFEST.sha256)

remote_heads="$(git ls-remote --heads "${repo_url}")"
if [[ -z "${remote_heads}" ]]; then
  git -C "${github_dir}" init -b main
  git -C "${github_dir}" add .
  git -C "${github_dir}" commit -m "Release CV801 experiment code and reproducibility materials"
  git -C "${github_dir}" remote add origin "${repo_url}"
  git -C "${github_dir}" push -u origin main
  echo "github_publish_status=COMPLETE mode=initial"
  exit 0
fi

update_dir="$(mktemp -d "${package_dir}/github-update.XXXXXX")"
git clone --depth 1 "${repo_url}" "${update_dir}"
rsync -a --delete --exclude=.git "${github_dir}/" "${update_dir}/"
(cd "${update_dir}" && sha256sum -c MANIFEST.sha256)
git -C "${update_dir}" add -A
if git -C "${update_dir}" diff --cached --quiet; then
  echo "github_publish_status=NO_CHANGES mode=incremental"
  exit 0
fi
git -C "${update_dir}" commit -m "Add frozen Writer-scale artifacts and recovery tooling"
git -C "${update_dir}" push origin HEAD:main
echo "github_publish_status=COMPLETE mode=incremental commit=$(git -C "${update_dir}" rev-parse HEAD)"
