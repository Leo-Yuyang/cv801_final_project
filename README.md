# CV801：农业目标检测、视觉语言生成与 Agent Review 实验

CV801 是一套面向水果与叶片图像的可复现实验工程，覆盖三类目标检测器、基于 Qwen3.5 的结构化视觉语言生成，以及由视觉 reviewer、领域 reviewer、Judge、revision 和独立视觉 evaluator 组成的 Agent Review 流程。

本仓库提供训练、推理、评估、审计脚本以及实验说明 PDF；处理后数据、项目微调权重和逐样本结果发布在 Hugging Face。公开基础模型必须从原作者仓库下载，本项目不重复分发。

- GitHub：<https://github.com/Leo-Yuyang/cv801_final_project>
- 实验说明：[`results/report/main.pdf`](results/report/main.pdf)
- 机器可读总表：[`results/final_summary.complete.json`](results/final_summary.complete.json)
- 完整结果表：[`results/final_summary.complete.md`](results/final_summary.complete.md)

> Hugging Face 项目仓库目前为 private。下载前需要项目授权并执行 `hf auth login`；公开 GitHub 仓库本身不自动授予数据或权重访问权限。

## 1. 快速开始

以下命令均在仓库根目录执行。

```bash
git clone https://github.com/Leo-Yuyang/cv801_final_project.git
cd cv801_final_project

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "huggingface_hub>=0.34"

cp .env.example .env
# 编辑 .env，将 PYTHON_HOME 设为当前 .venv 的绝对路径。

PYTHONPATH=CV801 python -m pytest -q CV801/tests
hf auth login
hf auth whoami
```

本地 CPU 环境使用 Python 3.10+，负责数据审计、COCO/VLM 指标和单元测试。正式训练与大模型推理使用 Slurm、Apptainer、ROCm 7.0 和 AMD PyTorch 2.8；完整 GPU 环境见第 4 节。

## 2. Hugging Face 物料

| 用途 | Hugging Face 仓库 | 内容 |
|---|---|---|
| 处理后数据 | [`leo20000306/cv801-data`](https://huggingface.co/datasets/leo20000306/cv801-data) | detection、T1、T2、Golden inputs/references |
| 完整实验结果 | [`leo20000306/cv801-experiment-results`](https://huggingface.co/datasets/leo20000306/cv801-experiment-results) | 逐样本预测、汇总指标、审计证据、PDF |
| YOLO26x | [`leo20000306/cv801-yolo26x`](https://huggingface.co/leo20000306/cv801-yolo26x) | 项目微调后的 `best.pt` |
| Grounding DINO Tiny | [`leo20000306/cv801-grounding-dino-tiny`](https://huggingface.co/leo20000306/cv801-grounding-dino-tiny) | 完整 Transformers checkpoint |
| LocateAnything-3B | [`leo20000306/cv801-locateanything-3b`](https://huggingface.co/leo20000306/cv801-locateanything-3b) | 完整微调 checkpoint；非商业研究/评估限制 |
| Qwen3.5-9B T1 | [`leo20000306/cv801-qwen3.5-9b-t1-lora`](https://huggingface.co/leo20000306/cv801-qwen3.5-9b-t1-lora) | T1 LoRA adapter，不含基础模型 |
| Qwen3.5-9B T2 | [`leo20000306/cv801-qwen3.5-9b-t2-lora`](https://huggingface.co/leo20000306/cv801-qwen3.5-9b-t2-lora) | T2 LoRA adapter，不含基础模型 |

### 下载数据并映射到代码期望目录

```bash
mkdir -p datas/cv801-hf
hf download leo20000306/cv801-data \
  --repo-type dataset \
  --local-dir datas/cv801-hf

(cd datas/cv801-hf && sha256sum -c MANIFEST.sha256)

[[ -e datas/cv801_detection_v1 || -L datas/cv801_detection_v1 ]] || \
  ln -s cv801-hf/detection datas/cv801_detection_v1
[[ -e datas/cv801_vlm_v1 || -L datas/cv801_vlm_v1 ]] || \
  ln -s cv801-hf/vlm-t1 datas/cv801_vlm_v1
[[ -e datas/cv801_vlm_t2_yolo26x_v1 || -L datas/cv801_vlm_t2_yolo26x_v1 ]] || \
  ln -s cv801-hf/vlm-t2-yolo26x datas/cv801_vlm_t2_yolo26x_v1
[[ -e datas/cv801_golden_test_inputs_v1 || -L datas/cv801_golden_test_inputs_v1 ]] || \
  ln -s cv801-hf/golden-inputs datas/cv801_golden_test_inputs_v1
```

数据目录含义如下：

| HF 目录 | 本地运行目录 | 用途 |
|---|---|---|
| `detection/` | `datas/cv801_detection_v1` | 图片、YOLO labels、COCO annotations、LocateAnything recipes |
| `vlm-t1/` | `datas/cv801_vlm_v1` | T1 train/validation JSONL |
| `vlm-t2-yolo26x/` | `datas/cv801_vlm_t2_yolo26x_v1` | 加入冻结 YOLO26x proposal hints 的 T2 JSONL |
| `golden-inputs/` | `datas/cv801_golden_test_inputs_v1` | 不含 reference 的最终推理输入 |
| `golden-references/` | 不建立训练软链接 | 最终评分 reference，避免训练或调参阶段误读 |

如果只需要查看已有结果，不需要配置 GPU：

```bash
hf download leo20000306/cv801-experiment-results \
  --repo-type dataset \
  --local-dir results/hf-release
(cd results/hf-release && sha256sum -c MANIFEST.sha256)
```

## 3. 数据口径与准备原则

最终口径为 8,097 张图、14,601 个目标和 11 个作物类别：

| split | 图像 | 目标 | 用途 |
|---|---:|---:|---|
| train | 6,387 | 11,555 | 参数学习 |
| validation | 710 | 1,299 | 模型选择、阈值和 review 配置冻结 |
| Golden Test | 1,000 | 1,747 | 冻结后的最终比较 |

同一近重复组不会跨 split。处理后图像已统一为校正 EXIF 方向的 RGB JPEG，框坐标与像素方向一致。监督 caption 来自经过数据集级视觉复核的 GPT caption，不应表述为独立人工 reference。

推荐直接使用 `cv801-data` 中已经冻结的处理后数据。GitHub 不包含原始 OneDrive/SharePoint 源文件，也不重新分发未确认授权的原始素材。如果需要从有权访问的原始源重新构建，使用 `CV801/scripts/prepare_*`、`validate_*` 和 `audit_*` 工具，并遵守以下协议：

- [`docs/protocols/detection/README.md`](docs/protocols/detection/README.md)
- [`docs/protocols/vlm/README.md`](docs/protocols/vlm/README.md)
- [`docs/protocols/golden-test/README.md`](docs/protocols/golden-test/README.md)
- [`docs/protocols/agent-review/README.md`](docs/protocols/agent-review/README.md)

Golden Test 必须在模型、prompt、阈值、checkpoint 和 review 配置冻结后才解封 reference。T2 的框提示来自冻结 YOLO26x 预测，不是 reference boxes。

## 4. GPU 训练与评估环境

### 已验证配置

| 层级 | 配置 |
|---|---|
| 调度器 | Slurm，单节点 1–8 GPU |
| GPU | AMD MI210 64 GiB；正式训练使用 8 卡 |
| 容器 | ROCm 7.0 + AMD PyTorch 2.8 Apptainer SIF |
| detector env | Ultralytics 8.4.202、Transformers 4.57.1、Accelerate 1.5.2、pycocotools |
| LocateAnything env | Transformers 4.57.1、PEFT 0.12.0、Eagle pinned checkout + MI210 SDPA patch |
| Qwen env | ms-swift 4.6.0、Transformers 5.19.0、qwen-vl-utils 0.0.14 |
| 精度/attention | BF16 或 AMP；MI210 使用 PyTorch SDPA，不依赖 FlashAttention |

先准备仓库期望的目录和容器。项目不分发内部集群镜像，需自行取得兼容的 ROCm 7.0 / PyTorch 2.8 SIF，并放到指定位置：

```bash
mkdir -p inits/containers inits/envs inits/huggingface inits/base-models wkdrs
cp /absolute/path/to/compatible-rocm7-pytorch2.8.sif \
  inits/containers/rocm7.0-pytorch2.8.sif

git clone https://github.com/NVlabs/Eagle.git inits/Eagle
git -C inits/Eagle checkout 783f656d127ee498137b5ff52603ce36c292d317
```

`cv801_prepare_training_runtime.sh` 会安装 detector/LocateAnything 两套环境，并对上述 Eagle commit 应用 [`CV801/patches/eagle_mi210_sdpa.patch`](CV801/patches/eagle_mi210_sdpa.patch)。Qwen 使用独立环境：

```bash
sbatch execs/scpts/cv801_prepare_training_runtime.sh
sbatch execs/scpts/cv801_prepare_qwen35_runtime.sh
```

运行前必须检查所有 `execs/scpts/*.sh` 的 `#SBATCH` 配置，并按目标集群修改 `--partition`、`--account`、`--qos`、GPU GRES 名称、内存和时限。脚本默认从仓库根目录提交，并要求 `.env`、SIF、数据软链接和运行环境均已就绪。

运行时会保存实际依赖快照：

```text
wkdrs/cv801-runtime/detector-requirements.freeze.txt
wkdrs/cv801-runtime/locateanything-requirements.freeze.txt
wkdrs/cv801-qwen35/requirements.freeze.txt
```

## 5. 从官方 Hugging Face 下载基础模型

训练使用以下上游模型和不可变 revision。`MODEL_PATH` 可以指向本地下载目录，也可以使用默认 HF repo ID；为避免训练节点临时联网，推荐提前下载到共享存储。

| 任务 | 上游模型 | 固定 revision |
|---|---|---|
| Grounding DINO | [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| LocateAnything | [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) | `c32291ca5e996f5a7a485845b4f57a233936bba0` |
| T0/T1/T2 Writer | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| 27B Writer baseline | [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) | `fc05daec18b0a78c049392ed2e771dde82bdf654` |
| 主线视觉 reviewer | [`OpenGVLab/InternVL3_5-8B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) | `741a7d03020411e666c6109218ab71e08151ef86` |
| 领域 reviewer | [`OpenGVLab/InternVL3_5-38B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-38B-HF) | `7c830fc25e874f9a861087e3c70587f2b9b555f3` |
| reviewer swap / 独立 evaluator | [`mistral-community/pixtral-12b`](https://huggingface.co/mistral-community/pixtral-12b) | `c2756cbbb9422eba9f6c5c439a214b0392dfc998` |
| Gemma 增量 reviewer | [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |

核心训练基础模型下载示例：

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

YOLO26x 的公开基础权重由固定版本 Ultralytics 在首次训练时下载为 `inits/yolo/yolo26x.pt`；也可以提前准备该文件并通过 `MODEL_PATH` 显式传入。所有上游模型继续受其原始许可和访问条款约束。

集群必须通过计算节点下载时，也可在 Qwen runtime 创建后使用：

```bash
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3.5-9B,REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  execs/scpts/cv801_download_hf_model.sh
```

## 6. 从基础模型进行 finetune

### 先跑单卡 smoke

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

### 正式 8 卡训练

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

正式配置如下；脚本中的环境变量可以覆盖这些默认值，但要复现实验应保持冻结设置：

| 模型 | 正式训练设置 |
|---|---|
| YOLO26x | 100 epochs；960 px；全局 batch 64；8 GPU；AMP；cosine LR；seed 20261104 |
| Grounding DINO Tiny | 12 epochs / 4,800 steps；batch 1/GPU；accumulation 2；有效 batch 16；LR 2e-5；BF16；cosine |
| LocateAnything-3B | 3,000 steps；batch 1/GPU；有效 batch 8；LR 2e-5；warmup 100；BF16；SDPA；4,096 tokens；LoRA rank 64 |
| Qwen3.5-9B T1 | 3 epochs；batch 1/GPU；accumulation 2；有效 batch 16；LR 1e-4；LoRA 64/128/0.05；4,096 tokens |
| Qwen3.5-9B T2 | 3 epochs / 1,197 steps；其余同 T1；正式训练 `MAX_LENGTH=6144`；输入增加冻结 YOLO26x proposals |

T1/T2 冻结 vision encoder 和 aligner，只在语言模型线性层训练 LoRA。HF 数据仓库已提供正式 T2 JSONL，因此复现 T2 finetune 不需要重新生成 proposals；只有重建数据时才需先完成 YOLO train/validation 推理。

## 7. 直接使用本项目 HF 权重推理

### 下载项目权重

```bash
mkdir -p inits/project-models

hf download leo20000306/cv801-yolo26x \
  --local-dir inits/project-models/yolo26x
hf download leo20000306/cv801-grounding-dino-tiny \
  --local-dir inits/project-models/grounding-dino-tiny
hf download leo20000306/cv801-locateanything-3b \
  --local-dir inits/project-models/locateanything-3b
hf download leo20000306/cv801-qwen3.5-9b-t1-lora \
  --local-dir inits/project-models/qwen35-t1-lora
hf download leo20000306/cv801-qwen3.5-9b-t2-lora \
  --local-dir inits/project-models/qwen35-t2-lora

for directory in inits/project-models/*; do
  (cd "$directory" && sha256sum -c MANIFEST.sha256)
done
```

### 检测模型：reference-free Golden inputs 推理

`cv801_golden_detection_infer.sh` 只读取 `golden-inputs`，不会在推理阶段看到 test annotations。三个模型通过 `MODEL_FAMILY` 和 `MODEL_PATH` 选择：

```bash
sbatch --export=ALL,MODEL_FAMILY=yolo,RUN_LABEL=hf-yolo,MODEL_PATH="$PWD/inits/project-models/yolo26x/best.pt" \
  execs/scpts/cv801_golden_detection_infer.sh

sbatch --export=ALL,MODEL_FAMILY=grounding_dino,RUN_LABEL=hf-gdino,MODEL_PATH="$PWD/inits/project-models/grounding-dino-tiny" \
  execs/scpts/cv801_golden_detection_infer.sh

sbatch --export=ALL,MODEL_FAMILY=locateanything,RUN_LABEL=hf-locate,MODEL_PATH="$PWD/inits/project-models/locateanything-3b" \
  execs/scpts/cv801_golden_detection_infer.sh
```

输出位于 `wkdrs/cv801-golden-test/detection-<family>-<label>/predictions.coco.json`。

### Qwen T1/T2 LoRA：validation 推理

LoRA 不能脱离 `Qwen/Qwen3.5-9B` 单独运行。以下命令分别加载同一基础模型与两个项目 adapter；脚本也继续兼容旧的 `TRAIN_JOB_ID=<job_id>` 自动定位方式。

```bash
sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t1-lora",MAX_NEW_TOKENS=4096 \
  execs/scpts/cv801_qwen35_t1_infer.sh

sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t2-lora",MAX_NEW_TOKENS=4096 \
  execs/scpts/cv801_qwen35_t2_infer.sh
```

T1 可用 8 卡独立分片加速：

```bash
sbatch --export=ALL,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t1-lora",GPUS=8,MAX_NEW_TOKENS=4096 \
  execs/scpts/cv801_qwen35_t1_infer_8gpu.sh
```

如果要运行不含 reference 的 1,000-image Golden Writer：

```bash
sbatch --export=ALL,VARIANT=t1,RUN_LABEL=hf-t1,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t1-lora" \
  execs/scpts/cv801_golden_writer_infer.sh

sbatch --export=ALL,VARIANT=t2,RUN_LABEL=hf-t2,MODEL_PATH="$PWD/inits/base-models/qwen3.5-9b",ADAPTER_PATH="$PWD/inits/project-models/qwen35-t2-lora" \
  execs/scpts/cv801_golden_writer_infer.sh
```

Agent Review 的输入构造、Rc/Rd、Judge、revision、recheck 和独立 evaluator 是多阶段流程，请按 [`docs/protocols/agent-review/README.md`](docs/protocols/agent-review/README.md) 执行，不要跳过冻结与 reference 隔离审计。

## 8. 统一评估

### 检测 COCO mAP 与冻结 operating point

预测完成后才允许使用对应 split 的 annotations。下面以 Golden Test 为例；阈值必须使用 validation 已冻结值，不能在 test 上重新选择：

| 模型 | validation 冻结 score threshold |
|---|---:|
| YOLO26x | 0.404 |
| Grounding DINO Tiny | 0.25091278553009033 |
| LocateAnything-3B | 0.2（模型无原生 confidence，所有解码框使用常数 0.2） |

```bash
.venv/bin/python CV801/scripts/evaluate_coco_predictions.py \
  --ground-truth datas/cv801_detection_v1/coco/instances_test.json \
  --predictions wkdrs/cv801-golden-test/detection-yolo-hf-yolo/predictions.coco.json \
  --output wkdrs/cv801-golden-test/detection-yolo-hf-yolo/metrics.coco.json \
  --model-name "YOLO26x fine-tuned" \
  --score-policy "native_model_score; AP conf=0.001; nms_iou=0.7" \
  --operating-score-threshold 0.404
```

Grounding DINO 和 LocateAnything 的 validation 端到端预测与统一 COCO 评估可直接加载 HF checkpoint：

```bash
sbatch --export=ALL,MODEL_PATH="$PWD/inits/project-models/grounding-dino-tiny" \
  execs/scpts/cv801_grounding_dino_coco_eval.sh

sbatch --export=ALL,MODEL_PATH="$PWD/inits/project-models/locateanything-3b" \
  execs/scpts/cv801_locateanything_coco_eval.sh
```

### VLM 结构与文本指标

ms-swift 推理 JSONL 内包含预测与冻结 labels，可在 CPU 环境评估：

```bash
.venv/bin/python CV801/scripts/evaluate_vlm_predictions.py \
  /absolute/path/to/predictions.jsonl \
  --output /absolute/path/to/metrics.json
```

该评估报告结构合法率、字段准确率、对象 linkage、caption token F1、BLEU/ROUGE 等。caption reference 是复核后的 GPT supervision，不是人工农业专家评分。`extra_object_hallucination_proxy_rate` 仅是额外对象数量代理量，也不能表述为视觉 claim-level hallucination rate。

## 9. 主要结果

统一 pycocotools Golden Test 检测结果：

| 模型 | AP50–95 | AP50 | operating-point F1 |
|---|---:|---:|---:|
| YOLO26x | 0.7288 | 0.8221 | 0.7822 |
| LocateAnything-3B | 0.5333 | 0.6180 | 0.7788 |
| Grounding DINO Tiny | 0.2647 | 0.2815 | 0.3269 |

视觉语言实验包含：

- T0：`Qwen/Qwen3.5-9B` 零样本整图基线。
- T1：整图输入、无框提示的 Qwen3.5-9B LoRA。
- T2：与 T1 使用相同图像和 assistant target，只增加冻结 YOLO26x 预测框提示。
- Agent Review：Rc 视觉 reviewer、Rd 农业领域 reviewer、纯文本 Judge、Writer revision、Judge recheck 和独立 Pixtral visual evaluator。
- 扩展：Qwen3.5-27B Writer baseline、Pixtral visual reviewer 替换，以及 Gemma 3 visual reviewer 增量分支。

人工评分和 Direct GPT-6 Astra baseline 已按实验决策取消，不是缺失实验。当前 PDF 冻结已完成的 InternVL/Pixtral 主实验矩阵；Gemma 3 分支仍按增量方式追加，既有结果不会被删除或覆盖。完整数字、指标定义和证据边界以报告和 Hugging Face results 仓库为准。

## 10. 仓库结构与完整性检查

```text
CV801/scripts/       # 数据构建、推理适配、评估和审计工具
execs/scpts/         # MI210/ROCm 环境、训练、推理与评估启动器
CV801/tests/         # 数据隔离、指标和输入输出协议回归测试
CV801/patches/       # LocateAnything/Eagle 的 MI210 SDPA 兼容补丁
docs/protocols/      # detection、VLM、review 与 Golden Test 冻结协议
results/             # 汇总 JSON/Markdown、实验说明 PDF 与 LaTeX 源码
release/             # GitHub/Hugging Face 发布包生成与上传脚本
MANIFEST.sha256      # GitHub release 文件校验和
```

克隆后可执行：

```bash
sha256sum -c MANIFEST.sha256
PYTHONPATH=CV801 .venv/bin/python -m pytest -q CV801/tests
```

原始交付 ZIP 中旧 `agroannotate` 标注 UI 的 `prompts/` 和 `static/` 目录为空，因此本仓库不发布一个无法运行的残缺 UI，也不重建文件并冒充原实验源码。冻结后的数据、训练、推理、评估与审计脚本均在发布白名单中。

## 11. 许可与限制

本仓库当前未授予统一项目许可证，默认保留项目代码的全部权利。各基础模型、Ultralytics、LocateAnything/Eagle、数据来源和第三方代码继续受各自条款约束；尤其 LocateAnything 及其衍生权重仅限非商业研究或评估用途。不要将“能够访问 private HF repo”等同于获得公开再分发许可。

项目作者确认代码许可、数据许可、作者信息和引用格式后，再补充正式 `LICENSE` 与 citation 文件。第三方依赖范围见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
