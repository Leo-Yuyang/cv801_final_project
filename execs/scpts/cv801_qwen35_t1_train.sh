#!/usr/bin/env bash
#SBATCH --job-name=cv801-qwen-t1
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=wkdrs/cv801-qwen35/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
swift_bin="${project_root}/inits/envs/cv801-qwen35/bin/swift"
dataset_root="${project_root}/datas/cv801_vlm_v1"
smoke="${SMOKE:-0}"
stress="${STRESS:-0}"
gpus="${GPUS:-8}"

[[ -f "${container}" && -f "${project_root}/inits/envs/cv801-qwen35/pyvenv.cfg" && -f "${swift_bin}" \
   && -f "${dataset_root}/t1_train.jsonl" && -f "${dataset_root}/t1_val.jsonl" ]] || {
  echo "Qwen runtime or T1 dataset is missing." >&2
  exit 1
}
mkdir -p "${project_root}/wkdrs/cv801-qwen35"

if [[ "${smoke}" == "1" ]]; then
  if [[ "${stress}" == "1" ]]; then
    train_data="${dataset_root}/t1_train_stress.jsonl"
    val_data="${dataset_root}/t1_val_stress.jsonl"
    run_name="t1-stress-${SLURM_JOB_ID}"
  else
    train_data="${dataset_root}/t1_train_smoke.jsonl"
    val_data="${dataset_root}/t1_val_smoke.jsonl"
    run_name="t1-smoke-${SLURM_JOB_ID}"
  fi
  max_steps=2
  eval_steps=2
  save_steps=2
  grad_accum=1
  dataset_workers=1
  loader_workers=1
else
  train_data="${dataset_root}/t1_train.jsonl"
  val_data="${dataset_root}/t1_val.jsonl"
  max_steps=-1
  eval_steps=200
  save_steps=200
  grad_accum=2
  dataset_workers=8
  loader_workers=4
  run_name="t1-train-${SLURM_JOB_ID}"
fi
output_dir="${project_root}/wkdrs/cv801-qwen35/${run_name}"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=true
export APPTAINERENV_NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
export APPTAINERENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${project_root}" \
  "${container}" env NPROC_PER_NODE="${gpus}" "${swift_bin}" sft \
  --model "${MODEL_PATH:-Qwen/Qwen3.5-9B}" \
  --use_hf true \
  --dataset "${train_data}" \
  --val_dataset "${val_data}" \
  --load_from_cache_file true \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EPOCHS:-3}" \
  --max_steps "${MAX_STEPS:-${max_steps}}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lora_rank "${LORA_RANK:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_llm false \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION:-${grad_accum}}" \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --ddp_find_unused_parameters true \
  --attn_impl sdpa \
  --add_non_thinking_prefix true \
  --eval_steps "${eval_steps}" \
  --save_steps "${save_steps}" \
  --save_total_limit 3 \
  --logging_steps 1 \
  --max_length "${MAX_LENGTH:-4096}" \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --dataset_num_proc "${dataset_workers}" \
  --dataloader_num_workers "${loader_workers}" \
  --save_only_model true \
  --report_to tensorboard \
  --seed 20261104 \
  --output_dir "${output_dir}"

echo "qwen35_t1_status=PASS mode=$([[ "${smoke}" == "1" ]] && echo smoke || echo full)"
