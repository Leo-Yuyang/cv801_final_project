# CV801 视觉语言训练

## 实验边界

- T0：`Qwen/Qwen3.5-9B` 零样本基线，不训练。
- T1：整图输入，不暴露任何框；同一张图的全部合格对象聚合成一个结构化输出。
- T2：保持与 T1 相同的图像和 caption 目标，只增加冻结 YOLO26x 定位器产生的预测框；全量 train/val 定位结果、训练和冻结评估均已完成。
- `test` 的 1,000 张 golden 图不写入 T1 训练或验证文件。
- 监督文本来自经过数据集级视觉复核和修订的 GPT caption，但不是独立的工程师 reference；训练产物和论文中必须保留这个来源说明。

## 数据

```bash
source .env
"$PYTHON_HOME/bin/python" CV801/scripts/prepare_vlm_dataset.py
```

产物写入 `datas/cv801_vlm_v1/`：

- `t1_train.jsonl`、`t1_val.jsonl`：ms-swift 官方多模态 messages/images 格式。
- `t1_train_smoke.jsonl`、`t1_val_smoke.jsonl`：运行链路验证子集。
- `t1_train_stress.jsonl`、`t1_val_stress.jsonl`：最长多目标序列压力测试；当前分别为 35 和 22 个对象。
- `manifests/`：逐图审计信息；reference box 只在这里保存，不进入 T1 prompt 或 target。
- `summary.json`：数量、来源、排除原因和 golden 封存检查。

多目标图片只出现一次，目标按中心点从上到下、从左到右稳定编号。这样不会把同一张无框图片复制成多个输入相同但输出冲突的训练样本。

## 模型与参数

Qwen3.5-9B 是原生多模态模型。训练使用 ms-swift 4.6.0、Transformers 5.19.0 和 BF16 LoRA：rank 64、alpha 128、dropout 0.05、LR 1e-4、3 epochs、有效 batch 16。ViT 和 aligner 冻结，只在语言模型线性层添加 LoRA。序列上限为 4096，图像 token 上限为 768，以覆盖审计发现的 2800-token 长尾标注。MI210 使用 PyTorch SDPA，不依赖 FlashAttention。

```bash
sbatch execs/scpts/cv801_prepare_qwen35_runtime.sh
sbatch --export=ALL,SMOKE=1,GPUS=1 --gres=gpu:mi210:1 --cpus-per-task=16 --mem=96G --time=02:00:00 execs/scpts/cv801_qwen35_t1_train.sh
sbatch execs/scpts/cv801_qwen35_t1_train.sh
```

正式作业只能在数据验证、运行环境和单卡 smoke 均通过后提交。

## 推理与评估

T0 直接使用基础模型在 validation 上推理；T1 使用训练日志记录的最佳 LoRA checkpoint。两者均使用 temperature 0、整图输入、无框提示、4096 个最大生成 token 和同一个 `cv801_vlm_structured_v3_all_samples` 评估协议。对象 precision 的分母为全部预测对象，recall 的分母为全部 reference objects。原有 JSON/schema 合法率、对象计数、crop/plant part、分类字段、caption/action token F1 保持不变。新增 caption BLEU-4 和 ROUGE-L 时，只比较唯一 `object_id` 匹配的对象：BLEU-4 是小写字母数字分词、单参考、无平滑的 corpus BLEU-4；ROUGE-L 是对象级 F1 的均值。新增的 object-link precision/recall/F1 只衡量预测对象 ID 与 reference ID 的关联，不代表图像中的视觉关联已经得到确认。

`extra_object_hallucination_proxy_rate` 只统计“预测对象数超过 reference 对象数”的结构化代理量，分母为全部预测对象。它不能发现对象属性、病害或 action 中的无证据文本，也不能把 reference 的遗漏与模型幻觉区分开，因此报告时不得简称为视觉幻觉率。这些指标的文本 reference 仍是复核后的 GPT caption，不应写成独立人工真值。

```bash
sbatch --export=ALL,SMOKE=1 execs/scpts/cv801_qwen35_t0_infer.sh
sbatch --export=ALL,TRAIN_JOB_ID=<job_id>,SMOKE=1 execs/scpts/cv801_qwen35_t1_infer.sh
sbatch --export=ALL,TRAIN_JOB_ID=<job_id>,MAX_NEW_TOKENS=4096,GPUS=8 execs/scpts/cv801_qwen35_t1_infer_8gpu.sh
sbatch --export=ALL,PREDICTIONS=<predictions.jsonl> execs/scpts/cv801_vlm_eval.sh
```

Golden Test 使用单独的无 reference 输入和冻结推理 DAG，见
[`docs/protocols/golden-test/README.md`](../golden-test/README.md)。协议要求在 validation 选型完成前不得运行
Golden 推理；冻结推理 DAG 本身不包含测试指标评估。
