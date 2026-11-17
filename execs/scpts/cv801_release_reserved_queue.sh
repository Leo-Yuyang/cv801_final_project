#!/usr/bin/env bash
#SBATCH --job-name=cv801-release-reserve
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=wkdrs/cv801-writer-scale/slurm-%x-%j.out

set -euo pipefail

target_job="${TARGET_JOB:?Set TARGET_JOB.}"
target_state="$(squeue -j "${target_job}" -h -o '%T')"

mapfile -t held_jobs < <(
  squeue -r -u "${USER}" -h -t PENDING -o '%i|%R' |
    awk -F'|' '$2 == "(JobHeldUser)" {print $1}'
)
failures=0
if ((${#held_jobs[@]})); then
  for held_job in "${held_jobs[@]}"; do
    if ! scontrol release "${held_job}"; then
      failures=$((failures + 1))
    fi
  done
fi

printf 'reservation_release_status=%s target=%s target_state=%s released=%s failures=%s uae=%s\n' \
  "$([[ "${failures}" -eq 0 ]] && echo PASS || echo PARTIAL)" "${target_job}" \
  "${target_state:-not-queued}" "${#held_jobs[@]}" "${failures}" \
  "$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
[[ "${failures}" -eq 0 ]]
