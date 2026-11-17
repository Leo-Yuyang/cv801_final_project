# CV801: Agricultural Object Detection, Vision-Language Generation, and Agent Review

CV801 is a reproducible experimental project for fruit and leaf imagery. It covers three object detectors, structured agricultural description with Qwen3.5, and an Agent Review pipeline composed of visual and domain reviewers, a text-only Judge, Writer revision, Judge recheck, and an independent visual evaluator.

This repository contains the executable training, inference, evaluation, and audit code. Processed data, project-trained weights, per-sample outputs, and aggregate results are distributed through two public Hugging Face repositories. Public upstream checkpoints are not mirrored.

- GitHub: <https://github.com/Leo-Yuyang/cv801_final_project>
- Experimental report: [`results/report/main.pdf`](results/report/main.pdf)
- Machine-readable summary: [`results/final_summary.complete.json`](results/final_summary.complete.json)
- Complete result tables: [`results/final_summary.complete.md`](results/final_summary.complete.md)
- Additive Writer-scale results: [`results/writer-scale/README.md`](results/writer-scale/README.md)

> The Hugging Face repositories are publicly readable and do not require authentication for download. Publication does not replace the artifact-specific license terms described below.

## 1. Release repositories

Only two Hugging Face repositories are used:

| Repository | Type | Contents |
|---|---|---|
| [`leo20000306/cv801_models`](https://huggingface.co/leo20000306/cv801_models) | Model | Nine released project-trained checkpoints, configurations, and checkpoint-level metrics |
| [`leo20000306/cv801_data`](https://huggingface.co/datasets/leo20000306/cv801_data) | Dataset | Processed datasets, Golden inputs/references, per-sample predictions, audit evidence, aggregate metrics, and the report |

`cv801_models` is organized by model family:

```text
yolo26x/
grounding-dino-tiny/
locateanything-3b/
qwen35-t1-lora/
qwen35-t2-lora/
qwen35-2b-t1-lora/
qwen35-2b-t2-lora/
qwen35-4b-t1-lora/
qwen35-4b-t2-lora/
```

`cv801_data` separates inputs from outputs:

```text
dataset/                 # processed detection, T1, T2, and Golden Test data
results/                 # predictions, metrics, audits, efficiency data, and report
```

Completed additive Writer-by-reviewer releases place the paired 3 x 3 analysis
under `results/writer-reviewer-grid/`. Treat the grid as complete only when the
directory contains the nine cells' structured and independent-evaluator
outputs, paired-analysis JSON, report tables, per-image sufficient statistics,
and repair-lineage files together. This atomic boundary prevents partial cells
from being presented as final results.

## 2. Quick start

Run these commands from the repository root:

```bash
git clone https://github.com/Leo-Yuyang/cv801_final_project.git
cd cv801_final_project

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "huggingface_hub>=0.34"

cp .env.example .env
# Edit .env and set PYTHON_HOME to the absolute path of .venv.

PYTHONPATH=CV801 python -m pytest -q CV801/tests
# Optional for public CV801 downloads; useful for higher Hub rate limits or gated upstream models:
# hf auth login
# hf auth whoami
```

The CPU environment requires Python 3.10+ and supports data audits, COCO/VLM metrics, and tests. Formal training and large-model inference use Slurm, Apptainer, ROCm 7.0, and AMD PyTorch 2.8.

## 3. Download and prepare the data

Download the consolidated data repository:

```bash
mkdir -p datas/cv801_data
hf download leo20000306/cv801_data \
  --repo-type dataset \
  --local-dir datas/cv801_data

(cd datas/cv801_data && sha256sum -c MANIFEST.sha256)
```

Map the released directories to the paths expected by the scripts:

```bash
[[ -e datas/cv801_detection_v1 || -L datas/cv801_detection_v1 ]] || \
  ln -s cv801_data/dataset/detection datas/cv801_detection_v1
[[ -e datas/cv801_vlm_v1 || -L datas/cv801_vlm_v1 ]] || \
  ln -s cv801_data/dataset/vlm-t1 datas/cv801_vlm_v1
[[ -e datas/cv801_vlm_t2_yolo26x_v1 || -L datas/cv801_vlm_t2_yolo26x_v1 ]] || \
  ln -s cv801_data/dataset/vlm-t2-yolo26x datas/cv801_vlm_t2_yolo26x_v1
[[ -e datas/cv801_golden_test_inputs_v1 || -L datas/cv801_golden_test_inputs_v1 ]] || \
  ln -s cv801_data/dataset/golden-inputs datas/cv801_golden_test_inputs_v1
```

Do not create a training-time link to `dataset/golden-references/`. Golden references must remain isolated until every model, prompt, threshold, checkpoint, and review configuration has been frozen.

To download only experiment results:

```bash
hf download leo20000306/cv801_data \
  --repo-type dataset \
  --include "results/*" \
  --local-dir artifacts/cv801_data
```

The Hugging Face repositories do not currently provide DOIs. For an exact
reproduction, record the Hub commit hash and verify `MANIFEST.sha256` rather
than citing only the mutable `main` branch.

### Dataset definition

The frozen dataset contains 8,097 images, 14,601 annotated objects, and 11 crop classes:

| Split | Images | Objects | Purpose |
|---|---:|---:|---|
| Train | 6,387 | 11,555 | Parameter learning |
| Validation | 710 | 1,299 | Model selection and configuration freezing |
| Golden Test | 1,000 | 1,747 | Final comparison after freezing |

Near-duplicate groups do not cross splits. Images were normalized to orientation-corrected RGB JPEG so that pixels and box coordinates remain aligned. VLM supervision uses dataset-level visually reviewed GPT captions; it is not an independent agricultural-expert reference.

Using the frozen processed release is recommended. The GitHub repository does not redistribute the original OneDrive/SharePoint source files. Authorized users who need to rebuild the processed dataset should use the `prepare_*`, `validate_*`, and `audit_*` tools under `CV801/scripts/` and follow:

- [`docs/protocols/detection/README.md`](docs/protocols/detection/README.md)
- [`docs/protocols/vlm/README.md`](docs/protocols/vlm/README.md)
- [`docs/protocols/golden-test/README.md`](docs/protocols/golden-test/README.md)
- [`docs/protocols/agent-review/README.md`](docs/protocols/agent-review/README.md)

T2 proposal hints are generated by a frozen YOLO26x checkpoint. They are predictions, not reference boxes.

## 4. GPU training and evaluation environment

### Validated stack

| Layer | Configuration |
|---|---|
| Scheduler | Slurm, one node, 1--8 GPUs |
| GPU | AMD MI210 64 GiB; formal training uses 8 GPUs |
| Container | ROCm 7.0 + AMD PyTorch 2.8 Apptainer SIF |
| Detector environment | Ultralytics 8.4.202, Transformers 4.57.1, Accelerate 1.5.2, pycocotools |
| LocateAnything environment | Transformers 4.57.1, PEFT 0.12.0, pinned Eagle checkout, MI210 SDPA patch |
| Qwen environment | ms-swift 4.6.0, Transformers 5.19.0, qwen-vl-utils 0.0.14 |
| Precision/attention | BF16 or AMP; PyTorch SDPA on MI210; no FlashAttention dependency |

The internal cluster image is not distributed. Obtain a compatible ROCm 7.0 / PyTorch 2.8 SIF and place it at the expected path:

```bash
mkdir -p inits/containers inits/envs inits/huggingface inits/base-models wkdrs
cp /absolute/path/to/compatible-rocm7-pytorch2.8.sif \
  inits/containers/rocm7.0-pytorch2.8.sif

git clone https://github.com/NVlabs/Eagle.git inits/Eagle
git -C inits/Eagle checkout 783f656d127ee498137b5ff52603ce36c292d317
```

The detector runtime script creates both the detector and LocateAnything environments and applies [`CV801/patches/eagle_mi210_sdpa.patch`](CV801/patches/eagle_mi210_sdpa.patch). Qwen uses a separate environment:

```bash
sbatch execs/scpts/cv801_prepare_training_runtime.sh
sbatch execs/scpts/cv801_prepare_qwen35_runtime.sh
```

Before submission, edit the `#SBATCH` headers under `execs/scpts/` for the target cluster: partition, account, QoS, GPU GRES name, memory, and time limit. Submit from the repository root after `.env`, the SIF, data links, and runtime environments are ready.

Runtime dependency snapshots are written to:

```text
wkdrs/cv801-runtime/detector-requirements.freeze.txt
wkdrs/cv801-runtime/locateanything-requirements.freeze.txt
wkdrs/cv801-qwen35/requirements.freeze.txt
```

## 5. Download upstream base models

Project training uses the following immutable upstream revisions. `MODEL_PATH` may point to a local directory or, where supported, the default Hugging Face repository ID. Pre-download to shared storage to avoid network access from compute nodes.

| Role | Upstream model | Pinned revision |
|---|---|---|
| Grounding DINO | [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| LocateAnything | [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) | `c32291ca5e996f5a7a485845b4f57a233936bba0` |
| Small Writer | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Small Writer | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| T0/T1/T2 Writer and Judge | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| 27B Writer baseline | [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) | `fc05daec18b0a78c049392ed2e771dde82bdf654` |
| R-low Rc reviewer | [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) | `093f9f388b31de276ce2de164bdc2081324b9767` |
| R-mid Rc reviewer | [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| R-high Rc reviewer | [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) | `005ad3404e59d6023443cb575daa05336842228a` |
| R-low Rd reviewer / original Rc | [`OpenGVLab/InternVL3_5-8B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) | `741a7d03020411e666c6109218ab71e08151ef86` |
| R-mid Rd reviewer | [`OpenGVLab/InternVL3_5-14B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF) | `226b96d5912e69159abc0384cefcbd51487fdce0` |
| R-high Rd reviewer / original Rd | [`OpenGVLab/InternVL3_5-38B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-38B-HF) | `7c830fc25e874f9a861087e3c70587f2b9b555f3` |
| Reviewer swap / evaluator | [`mistral-community/pixtral-12b`](https://huggingface.co/mistral-community/pixtral-12b) | `c2756cbbb9422eba9f6c5c439a214b0392dfc998` |

Core training downloads:

```bash
hf download IDEA-Research/grounding-dino-tiny \
  --revision a2bb814dd30d776dcf7e30523b00659f4f141c71 \
  --local-dir inits/base-models/grounding-dino-tiny

hf download nvidia/LocateAnything-3B \
  --revision c32291ca5e996f5a7a485845b4f57a233936bba0 \
  --local-dir inits/base-models/locateanything-3b

hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir inits/base-models/qwen3.5-9b
```

The controlled Writer-scale extension uses the same command with the pinned
2B or 4B repository and revision from the table. It is orchestrated by
`execs/scpts/cv801_writer_scale_train_submit.sh`, followed by
`cv801_writer_scale_posttrain.sh` and `cv801_writer_eval_submit.sh`.

The controlled reviewer-capacity extension downloads all three Gemma Rc tiers
and all three InternVL Rd tiers at the pinned revisions above. Gemma repositories
may require acceptance of the publisher's Hugging Face terms and an
authenticated token even though the resulting CV801 artifacts are public.

Ultralytics 8.4.202 downloads the public `yolo26x.pt` checkpoint on the first YOLO training run and stores it under `inits/yolo/`. You may instead prepare the file in advance and pass it through `MODEL_PATH`. All upstream licenses and access terms remain in force.

If downloads must run through a compute job after the Qwen runtime is ready:

```bash
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3.5-9B,REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  execs/scpts/cv801_download_hf_model.sh
```

## 6. Fine-tune from upstream checkpoints

### One-GPU smoke tests

```bash
sbatch --export=ALL,SMOKE=1,GPUS=1 \
  --gres=gpu:mi210:1 execs/scpts/cv801_yolo26x_train.sh

sbatch --export=ALL,SMOKE=1,GPUS=1,MODEL_PATH="$PWD/inits/base-models/grounding-dino-tiny" \
  --gres=gpu:mi210:1 execs/scpts/cv801_grounding_dino_train.sh

sbatch --export=ALL,SMOKE=1,GPUS=1,MODEL_PATH="$PWD/inits/base-models/locateanything-3b" \
  --gres=gpu:mi210:1 execs/scpts/cv801_locateanything_lora_train.sh

sbatch --export=ALL,SMOKE=1,GPUS=1,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b" \
  --gres=gpu:mi210:1 execs/scpts/cv801_qwen35_t1_train.sh

sbatch --export=ALL,SMOKE=1,GPUS=1,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b" \
  --gres=gpu:mi210:1 execs/scpts/cv801_qwen35_t2_train.sh
```

### Formal eight-GPU training

```bash
sbatch execs/scpts/cv801_yolo26x_train.sh

sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/grounding-dino-tiny" \
  execs/scpts/cv801_grounding_dino_train.sh

sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/locateanything-3b" \
  execs/scpts/cv801_locateanything_lora_train.sh

sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b" \
  execs/scpts/cv801_qwen35_t1_train.sh

sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",MAX_LENGTH=6144 \
  execs/scpts/cv801_qwen35_t2_train.sh
```

Frozen formal settings:

| Model | Training configuration |
|---|---|
| YOLO26x | 100 epochs; 960 px; global batch 64; 8 GPUs; AMP; cosine LR; seed 20261104 |
| Grounding DINO Tiny | 12 epochs / 4,800 steps; batch 1/GPU; accumulation 2; effective batch 16; LR 2e-5; BF16; cosine |
| LocateAnything-3B | 3,000 steps; batch 1/GPU; effective batch 8; LR 2e-5; warmup 100; BF16; SDPA; 4,096 tokens; LoRA rank 64 |
| Qwen3.5-9B T1 | 3 epochs; batch 1/GPU; accumulation 2; effective batch 16; LR 1e-4; LoRA 64/128/0.05; 4,096 tokens |
| Qwen3.5-9B T2 | 3 epochs / 1,197 steps; otherwise identical to T1; formal `MAX_LENGTH=6144`; frozen YOLO proposals added to the input |

T1 and T2 freeze the vision encoder and aligner and train LoRA modules on language-model linear layers. The released dataset already includes the formal T2 JSONL, so proposal regeneration is not required for a direct fine-tuning reproduction.

## 7. Use the consolidated project checkpoints

Download every trained checkpoint:

```bash
mkdir -p inits/project-models
hf download leo20000306/cv801_models \
  --local-dir inits/project-models
(cd inits/project-models && sha256sum -c MANIFEST.sha256)
```

To download only one checkpoint, use an include pattern:

```bash
hf download leo20000306/cv801_models \
  --include "qwen35-t2-lora/*" \
  --local-dir inits/project-models
```

For example, the released 4B-T2 scaling adapter is selected with
`--include "qwen35-4b-t2-lora/*"`. It requires the pinned
`Qwen/Qwen3.5-4B` base checkpoint; it is not a standalone full model.

### Detection inference on reference-free Golden inputs

```bash
sbatch --export=ALL,MODEL_FAMILY=yolo,RUN_LABEL=hf-yolo,MODEL_PATH="$PWD/inits/project-models/yolo26x/best.pt" \
  execs/scpts/cv801_golden_detection_infer.sh

sbatch --export=ALL,MODEL_FAMILY=grounding_dino,RUN_LABEL=hf-gdino,MODEL_PATH="$PWD/inits/project-models/grounding-dino-tiny" \
  execs/scpts/cv801_golden_detection_infer.sh

sbatch --export=ALL,MODEL_FAMILY=locateanything,RUN_LABEL=hf-locate,MODEL_PATH="$PWD/inits/project-models/locateanything-3b" \
  execs/scpts/cv801_golden_detection_infer.sh
```

Outputs are written to `wkdrs/cv801-golden-test/detection-<family>-<label>/predictions.coco.json`. The inference script reads `golden-inputs` and does not access test annotations.

### Qwen T1/T2 LoRA inference

The adapters require the compatible `Qwen/Qwen3.5-9B` base model:

```bash
sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t1-lora",MAX_NEW_TOKENS=4096 \
  execs/scpts/cv801_qwen35_t1_infer.sh

sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t2-lora",MAX_NEW_TOKENS=4096 \
  execs/scpts/cv801_qwen35_t2_infer.sh
```

Eight independent T1 inference shards can run on one eight-GPU node:

```bash
sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t1-lora",GPUS=8,MAX_NEW_TOKENS=4096 \
  execs/scpts/cv801_qwen35_t1_infer_8gpu.sh
```

Reference-free 1,000-image Golden Writer runs:

```bash
sbatch --export=ALL,VARIANT=t1,RUN_LABEL=hf-t1,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t1-lora" \
  execs/scpts/cv801_golden_writer_infer.sh

sbatch --export=ALL,VARIANT=t2,RUN_LABEL=hf-t2,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t2-lora" \
  execs/scpts/cv801_golden_writer_infer.sh
```

For the multi-stage Rc/Rd/Judge/revision/recheck/evaluator pipeline, follow [`docs/protocols/agent-review/README.md`](docs/protocols/agent-review/README.md). Do not bypass configuration freezing or reference-isolation audits.

## 8. Unified evaluation

### COCO detection metrics

Use annotations only after prediction is complete. Operating thresholds were selected on validation and must not be re-selected on Golden Test:

| Model | Validation-frozen score threshold |
|---|---:|
| YOLO26x | 0.404 |
| Grounding DINO Tiny | 0.25091278553009033 |
| LocateAnything-3B | 0.2; the model has no native confidence and all decoded boxes receive 0.2 |

Example Golden Test evaluation:

```bash
.venv/bin/python CV801/scripts/evaluate_coco_predictions.py \
  --ground-truth datas/cv801_detection_v1/coco/instances_test.json \
  --predictions wkdrs/cv801-golden-test/detection-yolo-hf-yolo/predictions.coco.json \
  --output wkdrs/cv801-golden-test/detection-yolo-hf-yolo/metrics.coco.json \
  --model-name "YOLO26x fine-tuned" \
  --score-policy "native_model_score; AP conf=0.001; nms_iou=0.7" \
  --operating-score-threshold 0.404
```

Validation prediction plus unified COCO evaluation for downloaded Grounding DINO and LocateAnything checkpoints:

```bash
sbatch --export=ALL,MODEL_PATH="$PWD/inits/project-models/grounding-dino-tiny" \
  execs/scpts/cv801_grounding_dino_coco_eval.sh

sbatch --export=ALL,MODEL_PATH="$PWD/inits/project-models/locateanything-3b" \
  execs/scpts/cv801_locateanything_coco_eval.sh
```

### VLM structure and text metrics

ms-swift result JSONL includes predictions and frozen labels and can be evaluated on CPU:

```bash
.venv/bin/python CV801/scripts/evaluate_vlm_predictions.py \
  /absolute/path/to/predictions.jsonl \
  --output /absolute/path/to/metrics.json
```

The evaluator reports schema validity, field accuracy, object linkage, caption token F1, BLEU, ROUGE, and related metrics. Caption references are reviewed GPT supervision, not independent human ratings. `extra_object_hallucination_proxy_rate` is only an extra-object count proxy and must not be presented as a visual claim-level hallucination rate.

## 9. Main results

Unified pycocotools Golden Test detection results:

| Model | AP50--95 | AP50 | Operating-point F1 |
|---|---:|---:|---:|
| YOLO26x | 0.7288 | 0.8221 | 0.7822 |
| LocateAnything-3B | 0.5333 | 0.6180 | 0.7788 |
| Grounding DINO Tiny | 0.2647 | 0.2815 | 0.3269 |

The vision-language matrix includes:

- T0: zero-shot `Qwen/Qwen3.5-9B`, whole image, no box hints.
- T1: Qwen3.5-9B LoRA, whole image, no box hints.
- T2: the same images and assistant targets as T1, with frozen YOLO26x proposal hints added.
- Agent Review: visual reviewer Rc, domain reviewer Rd, text-only Judge, Writer revision, Judge recheck, and an independent Pixtral evaluator.
- Extensions: a Qwen3.5-27B Writer baseline, a Pixtral reviewer swap, and an additive Gemma 3 reviewer branch.

The frozen Writer-only scaling extension is reported separately in
[`results/writer-scale/README.md`](results/writer-scale/README.md). Under the
fixed protocol, 2B-T2, 4B-T2, and 9B-T2 achieved schema-valid rates of 0.1%,
5.2%, and 99.5%, respectively. The 4B-T2 independent visual score was high
despite its low schema validity, so model-based visual score is not treated as
a substitute for interface validity. The completed paired `3 x 3`
Writer-by-reviewer table and its confidence intervals are included in the PDF
and under [`results/writer-reviewer-grid/`](results/writer-reviewer-grid/).
All nine cells contain the same 1,000 Golden Test images; the analysis uses
10,000 paired image-cluster bootstrap resamples with seed `20261104`.

Human rating and the Direct GPT-6 Astra baseline were explicitly cancelled and are not missing experiments. The bundled PDF freezes the completed InternVL/Pixtral matrix. The Gemma branch is additive and does not remove or overwrite existing results. See the report and `cv801_data/results/` for complete values and evidence boundaries.

## 10. Repository layout and integrity checks

```text
CV801/scripts/       # data preparation, inference adapters, evaluation, and audits
execs/scpts/         # MI210/ROCm runtime, training, inference, and evaluation launchers
CV801/tests/         # regression tests for isolation, metrics, and I/O contracts
CV801/patches/       # MI210 SDPA compatibility patch for LocateAnything/Eagle
docs/protocols/      # frozen detection, VLM, review, and Golden Test protocols
results/             # summaries, experimental PDF, and LaTeX source
release/             # release-building and publishing tools
MANIFEST.sha256      # checksums for the GitHub release
```

After cloning:

```bash
sha256sum -c MANIFEST.sha256
PYTHONPATH=CV801 .venv/bin/python -m pytest -q CV801/tests
```

The legacy `agroannotate` UI in the original delivery had empty `prompts/` and `static/` directories. This release does not publish a non-runnable partial UI or recreate missing files and present them as original experimental source.

## 11. License and access limitations

This repository currently has no unified project license; project code is all-rights-reserved by default. Upstream models, Ultralytics, LocateAnything/Eagle, data sources, and third-party code retain their own terms. LocateAnything and its derivative checkpoint are restricted to non-commercial research or evaluation. Public availability does not waive or replace those terms.

A formal project `LICENSE` and citation file should be added only after the authors finalize code licensing, data permissions, authorship, and citation details. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the third-party dependency scope.

## 12. Data and code availability

The executable code and frozen protocols are available in this GitHub
repository. Processed data, Golden inputs and references, predictions, metrics,
and report material are mapped explicitly to
[`leo20000306/cv801_data`](https://huggingface.co/datasets/leo20000306/cv801_data).
Released fine-tuned checkpoints and LoRA adapters are mapped to
[`leo20000306/cv801_models`](https://huggingface.co/leo20000306/cv801_models).
Public upstream base weights are obtained from their original repositories and
are not mirrored. None of these records currently has a DOI; cite the Git or
Hub commit hash used and verify the associated `MANIFEST.sha256`. Public access
does not create a blanket licence or supersede source-data and upstream-model
terms.
