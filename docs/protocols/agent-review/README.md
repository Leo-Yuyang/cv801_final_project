# CV801 T2 与 Agent Review 协议

## 当前顺序

1. 在 validation 上冻结定位器：显式小批次推理后的统一 pycocotools COCO AP50–95 为 YOLO26x 0.614、LocateAnything 0.444、Grounding DINO 0.250。YOLO26x 的 Ultralytics 原生口径 mAP50–95 为 0.624；两种口径不混用。
2. 用冻结的 YOLO26x 对 train/val 生成预测框，构建 T2；golden test 保持封存。
3. T2 与 T1 使用完全相同的图像和 assistant target。唯一处理差异是 T2 user prompt 增加预测框、类别和原生置信度。
4. T2 在结构合法率、对象计数、precision/recall 和 caption F1 上优于 T1，冻结 `checkpoint-1197` 作为第一轮 Agent Review Writer；T1 在 action、stage 和 confidence MAE 上更好，这些回退继续单独报告。
5. 冻结 Writer 后运行 Rc、Rd、J 和 revision ablation。

## T2 输入约束

- 预测框来自 `wkdrs/cv801-yolo26x/train-357714/weights/best.pt`，不是 reference box。
- 输入坐标为相对整图归一化到 0–1000 的 `xyxy`。
- 默认只暴露 score ≥ 0.25 的候选，每图最多 20 个。
- 候选可能漏检或误检；prompt 明确要求模型把它们当作定位提示，而不是事实标签。
- 数据构建器逐样本检查 T2 assistant target 与 T1 完全一致，并单独审计预测框对 reference objects 的 IoU≥0.5 覆盖率。

## Review loop 隔离

- 第一轮 Rc 使用 `OpenGVLab/InternVL3_5-8B-HF`，作为冻结主线结果保留。Gemma 3 的 Hugging Face 访问权限后来已解决，其 visual-reviewer 替换实验作为独立增量运行，不覆盖 InternVL/Pixtral 结果。
- 第一轮 Rd 使用 `OpenGVLab/InternVL3_5-38B-HF`；两个 InternVL checkpoint 均固定到已下载的 commit。
- Judge J 使用未微调的 `Qwen/Qwen3.5-9B`，Writer revision 继续使用冻结的 T2 LoRA。
- Rc：看图像与冻结 Writer 草稿，只检查可见证据、目标覆盖和 object linkage。
- Rd：看图像与草稿，保守检查农业术语、成熟度、病害、严重度、置信度和 action。
- J：只看 Writer 草稿及两份 review，不看图像，不得自行补充视觉事实。
- Rc、Rd、J 均不接收 validation/test reference 文本。
- 独立 evaluator 在 revision loop 冻结后评分，不能由 J 兼任。

`CV801/scripts/prepare_agent_review_inputs.py` 会删除 Writer 的 `<think>` 前缀，只导出 draft 和 image path；原始 prediction 中的 `labels` 不进入 review 输入。第一阶段运行冻结的 instruction reviewer baseline，不做缺少人工 review supervision 的 verifier 微调。

## 最终评估边界

- 取消人工或工程师评分步骤；它不再是实验完成条件。
- 取消 Direct GPT 6 Astra 外部基线；它不再进入实验矩阵或成本比较。
- 最终质量评分由与 revision loop 隔离的自动 evaluator 完成，Judge J 不得兼任。
- 0–6 justification rubric 只能报告为模型评估分数，不声称经过人工一致性验证。
- golden test reference 继续对 Writer、Rc、Rd 和 J 隐藏，仅供冻结后的自动指标计算使用。

## 独立自动 evaluator 协议

`CV801/scripts/prepare_independent_vlm_evaluator_inputs.py` 从冻结的 Writer 或 revision prediction 生成视觉 evaluator 输入。生成器只保留图像、去除 `<think>` 后的候选答案和稳定 sample ID，不复制 `labels` 或其他 reference 字段。协议 `cv801_independent_visual_rubric_v1` 要求与 Writer、Rc、Rd、J 均不同的视觉模型独立查看图像，并输出三项 0–2 整数分：视觉证据、结论支撑、置信度适当性；`total` 必须是三项之和，范围为 0–6。输出同时逐对象列出 supported/unsupported claims 和 `correct|incorrect|uncertain` object linkage。

```bash
source .env
"$PYTHON_HOME/bin/python" CV801/scripts/prepare_independent_vlm_evaluator_inputs.py \
  --predictions <frozen-validation-predictions.jsonl> \
  --output <independent-evaluator-inputs.jsonl>
```

这个生成器只定义无泄漏的输入/输出协议，不产生模型分数。必须另选独立视觉模型运行后，严格校验 JSON、0–2 分值和 total 求和关系，才能汇总 0–6 分。该分数属于模型评估，不替代已经取消的人工一致性验证。结构化评估器中的 `extra_object_hallucination_proxy_rate` 也只是额外对象代理率，不能代替这里基于图像的 unsupported-claim 审核。

## 作业链

- T2 数据 smoke：`381810`（通过）
- 修正后 T2 数据 smoke：`382172`（通过）
- 全量 YOLO train/val 预测与 T2 构建：`382173`（通过）
- 常规 T2 训练 smoke：`382482`（4096 tokens，通过）
- 最长序列 T2 训练 smoke：`382483`（6144 tokens，通过）
- 正式 8×MI210 T2 训练：`382484`（6144 tokens，3 epochs、1197 steps，通过）
- 正式 T2 validation 推理：`383368`（710/710，通过）
- T2 自动评估：`383369`（通过）
- Rc 8 样本运行链路 smoke：`391043`（通过；原始输出带 Markdown fence）
- Rc 严格 JSON smoke：`391051`（8/8 schema 合法）
- Rd 38B 双卡运行链路 smoke：`391044`（通过；原始输出 3/8 schema 合法）
- Rc 全量 validation：`391077`（完成；710 样本；初次严格审计 708/710，行 107–108 因 2048-token 上限截断）
- Rd 严格 JSON smoke：`391078`（完成；8/8 JSON 与 schema 合法）
- Rd 单任务全量 validation：`391095`（运行至 205/710 后主动取消；8 小时 walltime 无法延长且尾段超时风险过高，未产生结果文件）
- Judge 严格 JSON smoke：`391098`（完成；8/8 JSON 与 schema 合法，纯文本输入）
- Writer revision 强制前缀 smoke：Rc-only `391100`、Rd-only `391101`、Rc+Rd `391102`（均完成，但发现 `response_prefix` 会吞掉首键引号，结果不可用于评估）
- Writer revision 原生解码 smoke：Rc-only `391108`、Rd-only `391109`、Rc+Rd `391110`、完整 Judge 反馈 `391111`（均完成；四组均 8/8 JSON 与 schema 合法）
- Judge recheck smoke：`391118`（完成；8/8 JSON 与 schema 合法，8/8 判定 `revise_again`）

原始全量依赖链 `391134–391150` 已在上游重构后取消，未执行任何推理或评估。当前修复版 `afterok` 自动依赖链如下：

- Rc 定点修复：`391402` 的 4096-token 输出修复了行 108，但行 107 因重复循环仍截断；`391479` 的 8192-token 诊断确认是 object ID 循环而非真实长度需求。最终 compact-mode 修复 `391526 → 391527` 完成，合并后严格审计 710/710 JSON 与 schema 合法。原始及两次无效输出均以独立文件保留。
- Rd 四分片：`391403`、`391404`、`391405`、`391406`（均完成，共 710 样本）。首次合并审计 `391409` 检出 26 条数组内伪键值对语法，例如无效的 `["stage": "ripe"]`；有效率为 684/710。
- Rd 26 条 strict-JSON 定点修复与重新合并审计：`394287 → 394288`（完成；710/710 JSON 与 schema 合法）。原始合并输出以 `predictions.invalid-schema.jsonl` 保留。
- `391410–391428` 因首次 Rd 审计失败而取消，未执行推理或评估；最终链如下。
- reviewer 汇总与 Judge 输入：`394291`；Judge 全量：`394292`
- revision 输入与推理：Rc-only `394294 → 394295`、Rd-only `394296 → 394297`、Rc+Rd `394298 → 394299`、完整 Judge 反馈 `394300 → 394301`
- revision 自动评估：Rc-only `394302`、Rd-only `394303`、Rc+Rd `394304`、完整 Judge 反馈 `394305`
- recheck 输入、全量推理与严格审计：`394306 → 394307 → 394308`

Judge `394292` 全量输出有 709/710 条严格合法；第 478 条仅缺少最外层结束右花括号，使 `394300` 的强制审计失败。定点重跑 `396981` 后，`397026` 只替换该行的 response，并审计确认 710/710 JSON 与 schema 合法；`397027` 已重建 Full revision 输入。Full revision `397028` 与评估 `397029` 已完成。首次 recheck 输入任务 `397030` 因误指向不存在的 `t2-validation-final/review_manifest.jsonl` 而失败，依赖任务 `397031–397032` 已取消；修正后使用 `t2-validation/review_manifest.jsonl` 的链为输入准备 `401645`、recheck 推理 `401646`和严格审计 `401647`。无法再满足依赖的旧作业 `394301` 与 `394305–394308` 已取消；三组已完成的 revision 与评估没有重跑。

依赖链中的准备、评估和审计任务因 `gtqos` 要求最小 GRES，申请 1×MI210、10 分钟上限；它们只运行短 CPU 脚本，通常几十秒内释放资源。

8 样本 revision smoke 只用于协议验收，不据此选择最佳配置。Rc、Rd、Rc+Rd、完整 Judge 反馈的 caption F1 分别为 0.5702、0.5708、0.5758、0.5657；confidence MAE 分别为 0.1400、0.1400、0.1400、0.1050。最终结论以 710 样本 validation 为准。

Golden Test 的无 reference 输入、泄漏审计和冻结后并行提交 DAG 见
[`docs/protocols/golden-test/README.md`](../golden-test/README.md)。在 710 张 validation 上选定
review 配置前，提交脚本只执行 dry-run；正式 DAG 不包含测试 reference attach 或指标评估。

数据生成器必须把每个 rank 的路径显式切成小批次后再调用 Ultralytics；直接传入完整路径列表时，Ultralytics 会把整个列表作为一个内存批次，导致 warm-up 超过 MI210 的 64 GiB。该修正只改变数据装载和吞吐，不改变模型、输入尺寸、置信度阈值、NMS 或输出协议。
