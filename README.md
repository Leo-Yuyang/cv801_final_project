# CV801：农业目标检测、视觉语言生成与 Agent Review 实验

CV801 是一套面向水果与叶片图像的可复现实验代码，覆盖三类目标检测器、基于 Qwen3.5 的结构化视觉语言生成，以及视觉 reviewer、领域 reviewer、Judge、revision 和独立视觉 evaluator 组成的 Agent Review 流程。

本仓库发布代码、作业脚本、测试、实验协议和汇总结果。数据、最终微调权重和完整逐样本输出计划按用途单独发布到 Hugging Face；公开基础模型权重不会重新分发。本次 GitHub 快照不包含数据集、训练权重或逐样本推理输出。

## 数据与切分

最终实验口径为 8,097 张图、14,601 个目标和 11 个作物类别：

| split | 图像 | 目标 | 用途 |
|---|---:|---:|---|
| train | 6,387 | 11,555 | 参数学习 |
| validation | 710 | 1,299 | 模型选择、阈值和 review 配置冻结 |
| Golden Test | 1,000 | 1,747 | 冻结后的最终比较 |

同一近重复组不会跨 split。Golden Test 在模型、prompt、阈值、checkpoint 和 review 配置冻结前保持隔离。监督 caption 来自经过复核的 GPT caption，不应表述为独立人工参考答案。

## 实验矩阵

### 目标检测

- YOLO26x：100 epochs，960 px，全局 batch 64，8×AMD MI210。
- Grounding DINO Tiny：12 epochs，有效 batch 16，BF16。
- LocateAnything-3B：LoRA rank 64，3,000 steps，SDPA 4K。

统一 pycocotools Golden Test 结果：

| 模型 | AP50–95 | AP50 | operating-point F1 |
|---|---:|---:|---:|
| YOLO26x | 0.7288 | 0.8221 | 0.7822 |
| LocateAnything-3B | 0.5333 | 0.6180 | 0.7788 |
| Grounding DINO Tiny | 0.2647 | 0.2815 | 0.3269 |

### 视觉语言模型

- T0：`Qwen/Qwen3.5-9B` 零样本基线。
- T1：整图输入，不提供框；LoRA rank 64、3 epochs、有效 batch 16。
- T2：与 T1 使用相同图像和目标文本，只增加冻结 YOLO26x 的预测框提示。
- Agent Review：Rc 视觉 reviewer、Rd 农业领域 reviewer、纯文本 Judge、Writer revision、Judge recheck 和独立 Pixtral 视觉 evaluator。
- 扩展项：27B Writer baseline、Pixtral visual reviewer 替换，以及 Gemma 3 visual reviewer 替换脚本。

人工评分和 Direct GPT-6 Astra baseline 已按实验决策取消，不是缺失实验。

完整表格、指标定义和证据边界见 [`results/report/main.pdf`](results/report/main.pdf) 与 [`results/final_summary.complete.json`](results/final_summary.complete.json)。本版报告冻结已完成的 InternVL/Pixtral 主实验矩阵；Gemma 3 替换结果完成后将作为新增提交补入，不覆盖本版已有结果。

## 仓库结构

```text
CV801/scripts/      # 数据构建、推理适配、评估和审计工具
execs/scpts/         # MI210/ROCm 训练与推理启动器
CV801/tests/        # 数据隔离、指标、输入输出协议回归测试
CV801/patches/      # LocateAnything 在 MI210 上的兼容补丁
docs/protocols/     # detection、VLM、review 与 Golden Test 冻结协议
results/            # 汇总 JSON、实验说明 PDF 与 LaTeX 源码
release/            # GitHub/Hugging Face 发布包生成与上传脚本
```

## 本地安装与测试

数据处理、评估和审计工具需要 Python 3.10+。训练所需的 PyTorch、ROCm、Ultralytics、ms-swift 与 Eagle 由 `execs/scpts/cv801_prepare_*_runtime.sh` 在集群环境中单独准备：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=CV801 python -m pytest -q CV801/tests
```

训练环境使用 ROCm 7.0、AMD PyTorch 2.8 容器和 8×MI210。集群作业示例：

```bash
sbatch execs/scpts/cv801_prepare_training_runtime.sh
sbatch execs/scpts/cv801_yolo26x_train.sh
sbatch execs/scpts/cv801_grounding_dino_train.sh
sbatch execs/scpts/cv801_locateanything_lora_train.sh

sbatch execs/scpts/cv801_prepare_qwen35_runtime.sh
sbatch execs/scpts/cv801_qwen35_t1_train.sh
sbatch execs/scpts/cv801_qwen35_t2_train.sh
```

脚本中的 Slurm partition/account、容器位置和数据根目录属于部署参数，应按目标集群修改。复现实验必须固定 README 和协议中记录的基础模型 revision、随机种子与 split manifest。

原始交付 ZIP 中旧 `agroannotate` 标注 UI 的 `prompts/` 和 `static/` 目录为空，因此本仓库不发布一个无法运行的残缺 UI，也不重建文件并冒充原实验源码。冻结后的数据、训练、推理、评估与审计脚本均在上述白名单中。

## Hugging Face 物料

Hugging Face 发布计划按用途拆为 `cv801-data`、`cv801-experiment-results`、三个检测模型仓库和两个 Qwen LoRA 仓库。默认保持私有；数据授权和 Golden Test 公开策略确认后再切换可见性。仓库映射写在 `release/repo-map.env.example`。

## 许可与限制

本仓库当前未授予统一项目许可证，默认保留项目代码的全部权利。各基础模型、LocateAnything/Eagle、数据来源和第三方代码继续受各自条款约束；尤其 LocateAnything 权重仅按非商业学术研究边界准备。项目作者确认代码许可、数据许可、作者信息和引用格式后，再补充正式 `LICENSE` 与 citation 文件。
