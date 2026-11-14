#!/usr/bin/env bash
#SBATCH --job-name=cv801-ar-stage
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=wkdrs/cv801-agent-review/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }

python_bin="${project_root}/.venv-cv801/bin/python"
prepare_script="${project_root}/CV801/scripts/prepare_agent_review_inputs.py"
audit_script="${project_root}/CV801/scripts/audit_agent_outputs.py"
attach_script="${project_root}/CV801/scripts/attach_vlm_eval_labels.py"
evaluate_script="${project_root}/CV801/scripts/evaluate_vlm_predictions.py"
row_ops_script="${project_root}/CV801/scripts/jsonl_row_ops.py"
stage="${STAGE:?Set STAGE to judge, revision, revision_round2, recheck, evaluate, audit, select_rows, repair_reviewer, repair_judge, repair_recheck, or merge_reviewer.}"

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}

case "${stage}" in
  judge)
    require_file "${MANIFEST:?Set MANIFEST.}"
    require_file "${RC_PREDICTIONS:?Set RC_PREDICTIONS.}"
    require_file "${RD_PREDICTIONS:?Set RD_PREDICTIONS.}"
    output="${OUTPUT:?Set OUTPUT.}"
    "${python_bin}" "${audit_script}" --kind reviewer --predictions "${RC_PREDICTIONS}" --require-all-valid
    "${python_bin}" "${audit_script}" --kind reviewer --predictions "${RD_PREDICTIONS}" --require-all-valid
    "${python_bin}" "${prepare_script}" judge \
      --manifest "${MANIFEST}" \
      --visual-predictions "${RC_PREDICTIONS}" \
      --domain-predictions "${RD_PREDICTIONS}" \
      --output "${output}"
    ;;
  revision)
    require_file "${WRITER_PREDICTIONS:?Set WRITER_PREDICTIONS.}"
    feedback_source="${FEEDBACK_SOURCE:?Set FEEDBACK_SOURCE.}"
    output="${OUTPUT:?Set OUTPUT.}"
    args=(revision --writer-predictions "${WRITER_PREDICTIONS}" --feedback-source "${feedback_source}" --output "${output}")
    case "${feedback_source}" in
      rc)
        require_file "${RC_PREDICTIONS:?Set RC_PREDICTIONS.}"
        args+=(--visual-predictions "${RC_PREDICTIONS}")
        ;;
      rd)
        require_file "${RD_PREDICTIONS:?Set RD_PREDICTIONS.}"
        args+=(--domain-predictions "${RD_PREDICTIONS}")
        ;;
      rc_rd)
        require_file "${RC_PREDICTIONS:?Set RC_PREDICTIONS.}"
        require_file "${RD_PREDICTIONS:?Set RD_PREDICTIONS.}"
        args+=(--visual-predictions "${RC_PREDICTIONS}" --domain-predictions "${RD_PREDICTIONS}")
        ;;
      judge)
        require_file "${JUDGE_PREDICTIONS:?Set JUDGE_PREDICTIONS.}"
        "${python_bin}" "${audit_script}" --kind judge --predictions "${JUDGE_PREDICTIONS}" --require-all-valid
        args+=(--judge-predictions "${JUDGE_PREDICTIONS}")
        ;;
      *)
        echo "Unsupported FEEDBACK_SOURCE: ${feedback_source}" >&2
        exit 1
        ;;
    esac
    "${python_bin}" "${prepare_script}" "${args[@]}"
    ;;
  revision_round2)
    require_file "${REVISED_PREDICTIONS:?Set REVISED_PREDICTIONS.}"
    require_file "${RECHECK_PREDICTIONS:?Set RECHECK_PREDICTIONS.}"
    output="${OUTPUT:?Set OUTPUT.}"
    "${python_bin}" "${audit_script}" --kind recheck --predictions "${RECHECK_PREDICTIONS}" --require-all-valid
    "${python_bin}" "${prepare_script}" revision-round2 \
      --revised-predictions "${REVISED_PREDICTIONS}" \
      --recheck-predictions "${RECHECK_PREDICTIONS}" \
      --output "${output}"
    ;;
  recheck)
    require_file "${MANIFEST:?Set MANIFEST.}"
    require_file "${RC_PREDICTIONS:?Set RC_PREDICTIONS.}"
    require_file "${RD_PREDICTIONS:?Set RD_PREDICTIONS.}"
    require_file "${JUDGE_PREDICTIONS:?Set JUDGE_PREDICTIONS.}"
    require_file "${REVISED_PREDICTIONS:?Set REVISED_PREDICTIONS.}"
    output="${OUTPUT:?Set OUTPUT.}"
    previous_feedback_kind="${PREVIOUS_FEEDBACK_KIND:-judge}"
    [[ "${previous_feedback_kind}" == "judge" || "${previous_feedback_kind}" == "recheck" ]] || {
      echo "PREVIOUS_FEEDBACK_KIND must be judge or recheck." >&2
      exit 1
    }
    "${python_bin}" "${audit_script}" --kind "${previous_feedback_kind}" --predictions "${JUDGE_PREDICTIONS}" --require-all-valid
    "${python_bin}" "${prepare_script}" recheck \
      --manifest "${MANIFEST}" \
      --visual-predictions "${RC_PREDICTIONS}" \
      --domain-predictions "${RD_PREDICTIONS}" \
      --judge-predictions "${JUDGE_PREDICTIONS}" \
      --revised-predictions "${REVISED_PREDICTIONS}" \
      --output "${output}"
    ;;
  evaluate)
    require_file "${PREDICTIONS:?Set PREDICTIONS.}"
    require_file "${REFERENCE_PREDICTIONS:?Set REFERENCE_PREDICTIONS.}"
    labeled_output="${LABELED_OUTPUT:?Set LABELED_OUTPUT.}"
    metrics_output="${METRICS_OUTPUT:?Set METRICS_OUTPUT.}"
    "${python_bin}" "${attach_script}" \
      --predictions "${PREDICTIONS}" \
      --reference-predictions "${REFERENCE_PREDICTIONS}" \
      --output "${labeled_output}"
    "${python_bin}" "${evaluate_script}" "${labeled_output}" --output "${metrics_output}"
    ;;
  audit)
    require_file "${PREDICTIONS:?Set PREDICTIONS.}"
    audit_args=(
      --kind "${AUDIT_KIND:?Set AUDIT_KIND.}"
      --predictions "${PREDICTIONS}"
      --require-all-valid
    )
    if [[ -n "${AUDIT_OUTPUT:-}" ]]; then
      audit_args+=(--output "${AUDIT_OUTPUT}")
    fi
    "${python_bin}" "${audit_script}" "${audit_args[@]}"
    ;;
  select_rows)
    require_file "${SOURCE_INPUT:?Set SOURCE_INPUT.}"
    output="${OUTPUT:?Set OUTPUT.}"
    "${python_bin}" "${row_ops_script}" select \
      --source "${SOURCE_INPUT}" \
      --output "${output}" \
      --start "${START_LINE:?Set START_LINE.}" \
      --end "${END_LINE:?Set END_LINE.}"
    ;;
  repair_reviewer)
    require_file "${SOURCE_PREDICTIONS:?Set SOURCE_PREDICTIONS.}"
    require_file "${REPLACEMENT_PREDICTIONS:?Set REPLACEMENT_PREDICTIONS.}"
    output="${OUTPUT:?Set OUTPUT.}"
    IFS=: read -r -a line_numbers <<< "${LINE_NUMBERS:?Set colon-separated LINE_NUMBERS.}"
    "${python_bin}" "${row_ops_script}" replace \
      --source "${SOURCE_PREDICTIONS}" \
      --replacements "${REPLACEMENT_PREDICTIONS}" \
      --output "${output}" \
      --line-numbers "${line_numbers[@]}"
    "${python_bin}" "${audit_script}" --kind reviewer --predictions "${output}" --require-all-valid
    if [[ -n "${PROMOTE_TO:-}" ]]; then
      require_file "${PROMOTE_TO}"
      backup_output="${BACKUP_OUTPUT:?Set BACKUP_OUTPUT when PROMOTE_TO is used.}"
      [[ ! -e "${backup_output}" ]] || { echo "Refusing to overwrite ${backup_output}." >&2; exit 1; }
      mv "${PROMOTE_TO}" "${backup_output}"
      mv "${output}" "${PROMOTE_TO}"
      echo "agent_review_repair_promoted=${PROMOTE_TO} backup=${backup_output}"
    fi
    ;;
  repair_judge)
    require_file "${SOURCE_PREDICTIONS:?Set SOURCE_PREDICTIONS.}"
    require_file "${REPLACEMENT_PREDICTIONS:?Set REPLACEMENT_PREDICTIONS.}"
    output="${OUTPUT:?Set OUTPUT.}"
    IFS=: read -r -a line_numbers <<< "${LINE_NUMBERS:?Set colon-separated LINE_NUMBERS.}"
    "${python_bin}" "${row_ops_script}" replace-responses \
      --source "${SOURCE_PREDICTIONS}" \
      --replacements "${REPLACEMENT_PREDICTIONS}" \
      --output "${output}" \
      --line-numbers "${line_numbers[@]}"
    "${python_bin}" "${audit_script}" --kind judge --predictions "${output}" --require-all-valid
    if [[ -n "${PROMOTE_TO:-}" ]]; then
      require_file "${PROMOTE_TO}"
      backup_output="${BACKUP_OUTPUT:?Set BACKUP_OUTPUT when PROMOTE_TO is used.}"
      [[ ! -e "${backup_output}" ]] || { echo "Refusing to overwrite ${backup_output}." >&2; exit 1; }
      mv "${PROMOTE_TO}" "${backup_output}"
      mv "${output}" "${PROMOTE_TO}"
      echo "agent_review_repair_promoted=${PROMOTE_TO} backup=${backup_output}"
    fi
    ;;
  repair_recheck)
    require_file "${SOURCE_PREDICTIONS:?Set SOURCE_PREDICTIONS.}"
    require_file "${REPLACEMENT_PREDICTIONS:?Set REPLACEMENT_PREDICTIONS.}"
    output="${OUTPUT:?Set OUTPUT.}"
    IFS=: read -r -a line_numbers <<< "${LINE_NUMBERS:?Set colon-separated LINE_NUMBERS.}"
    "${python_bin}" "${row_ops_script}" replace-responses \
      --source "${SOURCE_PREDICTIONS}" \
      --replacements "${REPLACEMENT_PREDICTIONS}" \
      --output "${output}" \
      --line-numbers "${line_numbers[@]}"
    "${python_bin}" "${audit_script}" --kind recheck --predictions "${output}" --require-all-valid
    if [[ -n "${PROMOTE_TO:-}" ]]; then
      require_file "${PROMOTE_TO}"
      backup_output="${BACKUP_OUTPUT:?Set BACKUP_OUTPUT when PROMOTE_TO is used.}"
      [[ ! -e "${backup_output}" ]] || { echo "Refusing to overwrite ${backup_output}." >&2; exit 1; }
      mv "${PROMOTE_TO}" "${backup_output}"
      mv "${output}" "${PROMOTE_TO}"
      echo "agent_review_repair_promoted=${PROMOTE_TO} backup=${backup_output}"
    fi
    ;;
  merge_reviewer)
    output="${OUTPUT:?Set OUTPUT.}"
    require_file "${REFERENCE_INPUT:?Set REFERENCE_INPUT.}"
    IFS=: read -r -a sources <<< "${SOURCE_PREDICTIONS:?Set colon-separated SOURCE_PREDICTIONS.}"
    for source in "${sources[@]}"; do
      require_file "${source}"
    done
    "${python_bin}" "${row_ops_script}" merge \
      --sources "${sources[@]}" \
      --reference-input "${REFERENCE_INPUT}" \
      --output "${output}"
    "${python_bin}" "${audit_script}" --kind reviewer --predictions "${output}" --require-all-valid
    ;;
  *)
    echo "Unsupported STAGE: ${stage}" >&2
    exit 1
    ;;
esac

echo "agent_review_stage_status=PASS stage=${stage}"
