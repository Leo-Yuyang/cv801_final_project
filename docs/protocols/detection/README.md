# CV801 检测训练准备

## 固定实验口径

- 检测类别为 11 个作物类别；`fruit/leaf` 仅用于分层，不扩成 22 类。
- 原有 `golden` 的 1,000 张图保持不变，只作为最终测试集。
- 原有开发池 7,097 张按图像分为 6,387 张训练和 710 张验证。
- 同一 `duplicate_group` 必须在同一个集合；没有分组的图像各自成组。
- 验证集按 `fruit/leaf × crop` 分层，固定随机种子为 `20261104`。
- 三个模型必须使用同一份 train/val/test，禁止根据测试结果调参。
- 93 张仅有拒绝框的图像不进入 v1；它们保存在排除清单中，后续人工确认确为无目标图后才能作为 hard negatives 使用。

| 集合 | 图像 | 目标框 | 用途 |
|---|---:|---:|---|
| train | 6,387 | 11,555 | 参数学习 |
| val | 710 | 1,299 | 超参数和早停 |
| test | 1,000 | 1,747 | 冻结方案后的最终比较 |

完整校验结果：8,097 张图、14,601 个框、7,862 个分组，跨集合分组数为 0。

## 数据产物

数据根目录为 `datas/cv801_detection_v1/`：

- `dataset.yaml` 与 `labels/`：Ultralytics YOLO 格式。
- `coco/instances_{train,val,test}.json`：Grounding DINO 使用的 COCO 格式。
- `locateanything/{train,val,test}.jsonl`：LocateAnything 官方 ShareGPT 对话格式，坐标归一化为 `[0,1000]`。
- `locateanything/recipe_{train,smoke}.json`：LocateAnything 数据配方。
- `manifests/images.jsonl`：每张图的来源、尺寸、集合、近重复组和对象数。
- `manifests/excluded_images.jsonl`：93 张暂不纳入训练的图片及原因。
- `manifests/split_summary.json`：各层的图像数和固定验证目标。

原 HEIC/JPEG 没有改动。训练副本统一转成校正 EXIF 方向后的 RGB JPEG，使训练框坐标与像素方向一致。

重新生成和验证：

```bash
source .env
"$PYTHON_HOME/bin/python" CV801/scripts/prepare_detection_dataset.py --workers 16
"$PYTHON_HOME/bin/python" CV801/scripts/validate_detection_dataset.py
```

## 模型和运行环境

- YOLO：Ultralytics `8.4.202` 的 `yolo26x.pt`，当前已发布 YOLO26 系列中容量最大的检测模型。
- Grounding DINO：`IDEA-Research/grounding-dino-tiny`，沿用现有项目的基础模型声明。
- LocateAnything：`nvidia/LocateAnything-3B`，使用 LoRA 微调。
- LocateAnything 训练代码固定为 NVlabs/Eagle 提交 `783f656d127ee498137b5ff52603ce36c292d317`。
- 计算环境：8×AMD Instinct MI210（每卡约 64 GiB），ROCm 7.0，AMD 官方 PyTorch 2.8 容器。
- LocateAnything 在 MI210 上使用 `sdpa` 路径和 4K 上下文；Magi Attention 仅适用于 Hopper/Blackwell。`CV801/patches/eagle_mi210_sdpa.patch` 还包含 MI210 所需的标准 PyTorch 交叉熵、AMD 显存日志回退，以及相机大图的 1,792 px 长边上限，避免 FlashAttention、NVML、Liger Triton 和图像 token 截断问题。
- LocateAnything 模型许可限非商业学术研究；发布权重或商业使用前需重新核对许可。

运行环境由以下作业一次性创建并冻结依赖版本：

```bash
sbatch execs/scpts/cv801_prepare_training_runtime.sh
```

## 启动作业

单卡 smoke 命令如下；完整训练仍默认申请 8×MI210：

```bash
sbatch --export=ALL,SMOKE=1,GPUS=1 --gres=gpu:mi210:1 --cpus-per-task=8 --mem=64G --time=00:30:00 execs/scpts/cv801_yolo26x_train.sh
sbatch --export=ALL,SMOKE=1,GPUS=1 --gres=gpu:mi210:1 --cpus-per-task=8 --mem=64G --time=00:30:00 execs/scpts/cv801_grounding_dino_train.sh
sbatch --export=ALL,SMOKE=1,GPUS=1 --gres=gpu:mi210:1 --cpus-per-task=16 --mem=128G --time=01:00:00 execs/scpts/cv801_locateanything_lora_train.sh

sbatch execs/scpts/cv801_yolo26x_train.sh
sbatch execs/scpts/cv801_grounding_dino_train.sh
sbatch execs/scpts/cv801_locateanything_lora_train.sh
```

默认完整训练参数分别为 YOLO26x 100 epochs（960 px、全局 batch 64）、Grounding DINO Tiny 12 epochs（有效 batch 16，按验证损失恢复最佳权重），以及 LocateAnything-3B LoRA 3,000 steps（8 卡每步约 8 个样本，约 3.8 次数据遍历）。所有输出写入各自的 `wkdrs/cv801-*` 目录。

## 已完成验证

| 模型 | Slurm 作业 | 结果 | 覆盖范围 |
|---|---:|---|---|
| YOLO26x | 357614 | PASS | 1 epoch/64 张，训练、完整 val、权重保存 |
| Grounding DINO Tiny | 357657 | PASS | 32 张训练、16 张验证、权重保存 |
| LocateAnything-3B LoRA | 357678 | PASS | 2 个反向步骤、checkpoint 与最终权重保存 |

此前 8×MI210 硬件/RCCL 探针作业 357445 已通过；每卡约 64 GiB，设备架构为 `gfx90a`。上述三个模型 smoke 均读取了本项目的真实图片和标签，不是仅做 import 或随机张量测试。

## 正式微调（2026-11-04）

三个作业均在单节点 8×MI210 上完成：

| 模型 | 作业 | 冻结参数 | 最终状态 |
|---|---:|---|---|
| YOLO26x | 357714 | 100 epochs；960 px；全局 batch 64；AMP；cosine；30-epoch patience；每 10 epochs 留档 | COMPLETED；1:26:49 |
| Grounding DINO Tiny | 357715 | 12 epochs；batch 1/卡；gradient accumulation 2；有效 batch 16；LR 2e-5；BF16；cosine；按 eval loss 恢复最佳模型 | COMPLETED；1:06:11 |
| LocateAnything-3B | 357716 | LoRA rank 64；MLP projector 可训练；LLM/ViT 主权重冻结；3,000 steps；LR 2e-5；100-step warmup；BF16；SDPA 4K；stream packing buffer 32 | COMPLETED；5:20:14 |

冻结后的 Golden Test 统一 COCO 指标见仓库根目录 `README.md`、`results/final_summary.complete.json` 和实验报告 PDF。公开仓库不包含训练权重或集群输出目录。
