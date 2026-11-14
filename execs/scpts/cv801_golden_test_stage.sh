#!/usr/bin/env bash
#SBATCH --job-name=cv801-golden-stage
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
source "${project_root}/.env"
stage="${STAGE:?Set STAGE to prepare_t2, writer_audit, reviewers, or references.}"
input_root="${project_root}/datas/cv801_golden_test_inputs_v1"
prepare_golden="${project_root}/CV801/scripts/prepare_golden_test_inputs.py"
prepare_review="${project_root}/CV801/scripts/prepare_agent_review_inputs.py"

case "${stage}" in
  prepare_t2)
    predictions="${YOLO_PREDICTIONS:?Set YOLO_PREDICTIONS.}"
    model="${project_root}/wkdrs/cv801-yolo26x/train-357714/weights/best.pt"
    [[ -f "${predictions}" && -f "${model}" ]] || {
      echo "Frozen YOLO predictions or model are missing." >&2
      exit 1
    }
    [[ ! -e "${input_root}/writer_t2.jsonl" ]] || {
      echo "Refusing to overwrite frozen Golden T2 inputs." >&2
      exit 1
    }
    "${PYTHON_HOME}/bin/python" "${prepare_golden}" t2 \
      --input-root "${input_root}" \
      --predictions "${predictions}" \
      --locator-model "${model}" \
      --score-threshold "${T2_SCORE_THRESHOLD:-0.25}" \
      --max-proposals "${T2_MAX_PROPOSALS:-20}" \
      --expected-images 1000
    ;;
  writer_audit)
    predictions="${WRITER_PREDICTIONS:?Set WRITER_PREDICTIONS.}"
    variant="${WRITER_VARIANT:?Set WRITER_VARIANT to t0_t1 or t2.}"
    [[ "${variant}" == "t0_t1" || "${variant}" == "t2" ]] || {
      echo "WRITER_VARIANT must be t0_t1 or t2." >&2
      exit 1
    }
    [[ -f "${predictions}" ]] || { echo "Writer predictions are missing: ${predictions}" >&2; exit 1; }
    "${PYTHON_HOME}/bin/python" "${prepare_golden}" predictions \
      --predictions "${predictions}" \
      --input-root "${input_root}" \
      --variant "${variant}" \
      --expected-images 1000
    ;;
  reviewers)
    writer_predictions="${WRITER_PREDICTIONS:?Set WRITER_PREDICTIONS.}"
    output_dir="${OUTPUT_DIR:-${project_root}/wkdrs/cv801-golden-test/reviewer-inputs-${SLURM_JOB_ID}}"
    shards="${SHARDS:-4}"
    [[ "${shards}" =~ ^[1-9][0-9]*$ ]] || { echo "SHARDS must be positive." >&2; exit 1; }
    [[ -f "${writer_predictions}" ]] || {
      echo "Golden Writer predictions are missing: ${writer_predictions}" >&2
      exit 1
    }
    "${PYTHON_HOME}/bin/python" "${prepare_golden}" predictions \
      --predictions "${writer_predictions}" \
      --input-root "${input_root}" \
      --variant t2 \
      --expected-images 1000
    [[ ! -e "${output_dir}" ]] || { echo "Refusing to overwrite ${output_dir}" >&2; exit 1; }
    "${PYTHON_HOME}/bin/python" "${prepare_review}" reviewers \
      --writer-predictions "${writer_predictions}" \
      --output-dir "${output_dir}"
    for role in visual domain; do
      source_file="${output_dir}/${role}_reviewer_inputs.jsonl"
      shard_dir="${output_dir}/${role}_shards"
      mkdir -p "${shard_dir}"
      lines_per_shard=$(( (1000 + shards - 1) / shards ))
      split -d -a 2 -l "${lines_per_shard}" --additional-suffix=.jsonl \
        "${source_file}" "${shard_dir}/input-"
      test "$(find "${shard_dir}" -maxdepth 1 -type f -name 'input-*.jsonl' | wc -l)" -eq "${shards}"
      test "$(cat "${shard_dir}"/input-*.jsonl | wc -l)" -eq 1000
    done
    test "$(wc -l < "${output_dir}/review_manifest.jsonl")" -eq 1000
    ;;
  references)
    output="${OUTPUT:?Set OUTPUT to the post-inference Golden reference JSONL.}"
    "${PYTHON_HOME}/bin/python" "${project_root}/CV801/scripts/prepare_golden_vlm_references.py" \
      --output "${output}" \
      --expected-images 1000 \
      --expected-objects 1747
    ;;
  *)
    echo "Unsupported STAGE: ${stage}" >&2
    exit 1
    ;;
esac

echo "golden_test_stage_status=PASS stage=${stage} evaluation=NOT_RUN"
