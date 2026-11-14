#!/usr/bin/env bash
#SBATCH --job-name=cv801-golden-final
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=wkdrs/cv801-golden-test/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_root}/.env"
output_json="${OUTPUT_JSON:-${project_root}/wkdrs/cv801-golden-test/final_summary.complete.json}"
output_markdown="${OUTPUT_MARKDOWN:-${project_root}/wkdrs/cv801-golden-test/final_summary.complete.md}"
[[ ! -e "${output_json}" && ! -e "${output_markdown}" ]] || {
  echo "Refusing to overwrite an existing Golden final summary." >&2
  exit 1
}

"${PYTHON_HOME}/bin/python" "${project_root}/CV801/scripts/aggregate_golden_results.py" \
  --detection yolo26x="${project_root}/wkdrs/cv801-golden-test/detection-yolo-403226/metrics.complete.json" \
  --detection grounding_dino="${project_root}/wkdrs/cv801-golden-test/detection-grounding_dino-403227/metrics.complete.json" \
  --detection locateanything="${project_root}/wkdrs/cv801-golden-test/detection-locateanything-403228/metrics.complete.json" \
  --vlm t0="${project_root}/wkdrs/cv801-golden-test/writer-t0-403229/metrics.golden.json" \
  --vlm t1="${project_root}/wkdrs/cv801-golden-test/writer-t1-403230/metrics.golden.json" \
  --vlm t2="${project_root}/wkdrs/cv801-golden-test/writer-t2-403232/metrics.golden.json" \
  --vlm qwen35_27b="${project_root}/wkdrs/cv801-golden-test/writer-qwen35-27b-baseline-403316/metrics.golden.json" \
  --vlm rd="${project_root}/wkdrs/cv801-agent-review/revision-rd-golden-403247/metrics.golden.json" \
  --vlm rc="${project_root}/wkdrs/cv801-agent-review/revision-rc-golden-403318/metrics.golden.json" \
  --vlm rc_rd="${project_root}/wkdrs/cv801-agent-review/revision-rc_rd-golden-403320/metrics.golden.json" \
  --vlm full_round1="${project_root}/wkdrs/cv801-agent-review/revision-full-golden-403322/metrics.golden.json" \
  --vlm full_round2="${project_root}/wkdrs/cv801-agent-review/revision-full-golden-round2-403332/metrics.golden.json" \
  --vlm pixtral_visual_reviewer="${project_root}/wkdrs/cv801-agent-review/revision-full-golden-pixtral-rc-403390/metrics.golden.json" \
  --independent t0="${project_root}/wkdrs/cv801-independent-eval/golden-t0/metrics.json" \
  --independent t1="${project_root}/wkdrs/cv801-independent-eval/golden-t1/metrics.json" \
  --independent t2="${project_root}/wkdrs/cv801-independent-eval/golden-t2/metrics.json" \
  --independent qwen35_27b="${project_root}/wkdrs/cv801-independent-eval/golden-qwen35-27b/metrics.json" \
  --independent rd="${project_root}/wkdrs/cv801-independent-eval/golden-rd/metrics.json" \
  --independent rc="${project_root}/wkdrs/cv801-independent-eval/golden-rc/metrics.json" \
  --independent rc_rd="${project_root}/wkdrs/cv801-independent-eval/golden-rc-rd/metrics.json" \
  --independent full_round1="${project_root}/wkdrs/cv801-independent-eval/golden-full-round1/metrics.json" \
  --independent full_round2="${project_root}/wkdrs/cv801-independent-eval/golden-full-round2/metrics.json" \
  --review rc="${project_root}/wkdrs/cv801-golden-test/review-metrics/rc.json" \
  --review rd="${project_root}/wkdrs/cv801-golden-test/review-metrics/rd.json" \
  --review judge="${project_root}/wkdrs/cv801-golden-test/review-metrics/judge.json" \
  --review recheck_round1="${project_root}/wkdrs/cv801-golden-test/review-metrics/recheck-round1.json" \
  --review recheck_round2="${project_root}/wkdrs/cv801-golden-test/review-metrics/recheck-round2.json" \
  --review pixtral_rc="${project_root}/wkdrs/cv801-golden-test/review-metrics/pixtral-rc.json" \
  --review pixtral_judge="${project_root}/wkdrs/cv801-golden-test/review-metrics/pixtral-judge.json" \
  --review pixtral_recheck="${project_root}/wkdrs/cv801-golden-test/review-metrics/pixtral-recheck.json" \
  --efficiency "${project_root}/wkdrs/cv801-golden-test/efficiency.json" \
  --output-json "${output_json}" \
  --output-markdown "${output_markdown}"
echo "golden_finalize_status=PASS json=${output_json} markdown=${output_markdown}"
