#!/usr/bin/env bash
#SBATCH --job-name=cv801-locany
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=wkdrs/cv801-locateanything/slurm-%x-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${project_root}/.env" ]] || { echo "Submit this job from the STAR_demo root." >&2; exit 1; }
container="${project_root}/inits/containers/rocm7.0-pytorch2.8.sif"
python_bin="${project_root}/inits/envs/cv801-locateanything/bin/python"
eagle_root="${project_root}/inits/Eagle/Embodied"
dataset_root="${project_root}/datas/cv801_detection_v1"
smoke="${SMOKE:-0}"
gpus="${GPUS:-8}"

[[ -f "${container}" && -f "${project_root}/inits/envs/cv801-locateanything/pyvenv.cfg" && -f "${dataset_root}/locateanything/recipe_train.json" ]] || {
  echo "Training runtime or dataset is missing; run cv801_prepare_training_runtime.sh first." >&2
  exit 1
}
mkdir -p "${project_root}/wkdrs/cv801-locateanything"

export APPTAINERENV_HF_HOME="${project_root}/inits/huggingface"
export APPTAINERENV_TOKENIZERS_PARALLELISM=true
export APPTAINERENV_NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_LAUNCHER=pytorch

if [[ "${smoke}" == "1" ]]; then
  max_steps=2
  save_steps=2
  workers=1
  packing_buffer=4
  recipe="${dataset_root}/locateanything/recipe_smoke.json"
  run_name="smoke-${SLURM_JOB_ID}"
else
  max_steps="${MAX_STEPS:-3000}"
  save_steps="${SAVE_STEPS:-250}"
  workers="${WORKERS:-4}"
  packing_buffer="${PACKING_BUFFER_SIZE:-32}"
  recipe="${dataset_root}/locateanything/recipe_train.json"
  run_name="train-${SLURM_JOB_ID}"
fi
output_dir="${project_root}/wkdrs/cv801-locateanything/${run_name}"

apptainer exec --rocm --bind "${project_root}:${project_root}" --pwd "${eagle_root}" \
  "${container}" "${python_bin}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${gpus}" \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "${MODEL_PATH:-nvidia/LocateAnything-3B}" \
  --output_dir "${output_dir}" \
  --meta_path "${recipe}" \
  --max_steps "${max_steps}" \
  --overwrite_output_dir False \
  --block_size 6 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm True \
  --freeze_backbone True \
  --freeze_mlp False \
  --use_llm_lora "${LORA_RANK:-64}" \
  --use_backbone_lora 0 \
  --vision_select_layer -1 \
  --dataloader_num_workers "${workers}" \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION:-1}" \
  --save_strategy steps \
  --save_steps "${save_steps}" \
  --save_total_limit 3 \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay 0.01 \
  --warmup_steps "${WARMUP_STEPS:-100}" \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --sample_log_interval 10 \
  --packing_buffer_size "${packing_buffer}" \
  --max_seq_length 4096 \
  --max_num_tokens_per_sample 4096 \
  --max_num_tokens 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --report_to tensorboard \
  --run_name "${run_name}" \
  --mlp_connector_layers 2

echo "locateanything_status=PASS mode=$([[ "${smoke}" == "1" ]] && echo smoke || echo full)"
