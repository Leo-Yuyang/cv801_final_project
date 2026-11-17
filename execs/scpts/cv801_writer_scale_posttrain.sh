#!/usr/bin/env bash
#SBATCH --job-name=cv801-writer-posttrain
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=wkdrs/cv801-writer-scale/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${project_root}"

writer_tag="${WRITER_TAG:?Set WRITER_TAG.}"
model_tag="${MODEL_TAG:?Set MODEL_TAG.}"
model_path="${MODEL_PATH:?Set MODEL_PATH.}"
t1_train_job="${T1_TRAIN_JOB:?Set T1_TRAIN_JOB.}"
t2_train_job="${T2_TRAIN_JOB:?Set T2_TRAIN_JOB.}"
[[ "${writer_tag}" =~ ^w(2|4)b$ ]] || exit 1
[[ "${model_tag}" =~ ^qwen35-(2|4)b$ ]] || exit 1
[[ -f "${model_path}/config.json" ]] || exit 1

resolve_adapter() {
  local task="$1"
  local job_id="$2"
  local train_dir="${project_root}/wkdrs/cv801-qwen35/${task}-train-${job_id}"
  local latest_state adapter
  latest_state="$(find "${train_dir}" -type f -name trainer_state.json \
    -path '*/checkpoint-*/*' | sort -V | tail -n 1)"
  [[ -f "${latest_state}" ]] || {
    echo "Could not resolve trainer state under ${train_dir}." >&2
    exit 1
  }
  adapter="$(jq -r '.best_model_checkpoint // empty' "${latest_state}")"
  [[ -n "${adapter}" && -f "${adapter}/adapter_config.json" ]] || {
    echo "Could not resolve minimum-validation-loss adapter from ${latest_state}." >&2
    exit 1
  }
  printf '%s\n' "${adapter}"
}

t1_adapter="$(resolve_adapter t1 "${t1_train_job}")"
t2_adapter="$(resolve_adapter t2 "${t2_train_job}")"
base_dir="${project_root}/wkdrs/cv801-writer-scale/${model_tag}"

WRITER_TAG="${writer_tag}" VARIANT=t1 MODEL_PATH="${model_path}" \
ADAPTER_PATH="${t1_adapter}" RUN_DIR="${base_dir}/eval-t1" \
execs/scpts/cv801_writer_eval_submit.sh
WRITER_TAG="${writer_tag}" VARIANT=t2 MODEL_PATH="${model_path}" \
ADAPTER_PATH="${t2_adapter}" RUN_DIR="${base_dir}/eval-t2" \
execs/scpts/cv801_writer_eval_submit.sh

# Read the deterministic T2 submission record to connect the reviewer grid.
source "${base_dir}/eval-t2/submission.env"
t2_writer_job="${writer_job}"
t2_writer_predictions="${writer_predictions}"
review_dir="${base_dir}/reviewer-inputs"
review_input_job="$(sbatch --parsable --dependency="afterok:${t2_writer_job}" \
  --export="ALL,STAGE=reviewers,WRITER_PREDICTIONS=${t2_writer_predictions},OUTPUT_DIR=${review_dir},SHARDS=4" \
  execs/scpts/cv801_golden_test_stage.sh)"

WRITER_TAG="${writer_tag}" \
WRITER_PREDICTIONS="${t2_writer_predictions}" \
WRITER_MODEL_PATH="${model_path}" \
WRITER_ADAPTER_PATH="${t2_adapter}" \
REVIEW_DIR="${review_dir}" \
REVIEW_INPUT_DEPENDENCY="${review_input_job}" \
TIERS="low mid high" \
execs/scpts/cv801_writer_reviewer_grid_submit.sh

submitted_uae="$(TZ=Asia/Dubai date '+%Y-%m-%dT%H:%M:%S%z')"
{
  printf 'submitted_uae=%q\n' "${submitted_uae}"
  printf 'writer_tag=%q\n' "${writer_tag}"
  printf 'model_tag=%q\n' "${model_tag}"
  printf 'model_path=%q\n' "${model_path}"
  printf 't1_train_job=%q\n' "${t1_train_job}"
  printf 't2_train_job=%q\n' "${t2_train_job}"
  printf 't1_adapter=%q\n' "${t1_adapter}"
  printf 't2_adapter=%q\n' "${t2_adapter}"
  printf 't2_writer_job=%q\n' "${t2_writer_job}"
  printf 't2_writer_predictions=%q\n' "${t2_writer_predictions}"
  printf 'review_input_job=%q\n' "${review_input_job}"
  printf 'review_dir=%q\n' "${review_dir}"
} > "${base_dir}/posttrain.env"

echo "writer_scale_posttrain=SUBMITTED writer=${writer_tag} review_input=${review_input_job}"
