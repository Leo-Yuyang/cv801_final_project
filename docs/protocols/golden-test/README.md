# CV801 Golden Test 冻结推理流程

## 边界

- 最终数据口径固定为 8,097 张图和 14,601 个框：train 6,387、validation 710、Golden Test 1,000。
- Golden Test 只能在 validation 选型、prompt、checkpoint、阈值和 review 配置全部冻结后运行。
- Writer、Rc、Rd 和 Judge J 均不得接收 Golden caption、reference box、object count 或其他 reference label。
- 本流程只准备和运行冻结推理，不提交测试指标评估。测试 reference 只允许在独立的最终评估门之后读取。

## 已准备的无 reference 输入

`datas/cv801_golden_test_inputs_v1/` 当前包含：

- `writer_t0_t1.jsonl`：1,000 个 T0/T1 输入；每行只有 `system`、`user` 和一张图片，没有 assistant target。
- `manifest.jsonl`：仅保存 sample/image ID、图片路径、作物、plant part、尺寸和文件哈希，不保存框或对象数。
- `detector_input.coco.json`：只含 1,000 个 `images` 和 11 个 `categories`，没有 `annotations`。
- `summary.json`：输入版本和 reference 隔离声明。

安全检测清单的 image/category ID 与冻结 `instances_test.json` 的对应部分逐字节结构一致，因此三个检测器的预测可在最终解封后直接进行 COCO 评估；检测推理本身不读取测试 annotations。

重新检查输入，不会运行模型或评估：

```bash
source .env
"$PYTHON_HOME/bin/python" CV801/scripts/prepare_golden_test_inputs.py audit
```

预期结果必须为 1,000 张、`t0_t1_reference_free=true`、`detector_annotations_present=false`。

## 两阶段 T2 构建

T2 不提前读取 reference box。冻结 YOLO26x 首先在无 annotations 的 `detector_input.coco.json` 上生成预测；随后只把这些预测转换为 normalized predicted hints：

```bash
source .env
"$PYTHON_HOME/bin/python" CV801/scripts/prepare_golden_test_inputs.py t2 \
  --predictions wkdrs/cv801-golden-test/detection-yolo-<job>/predictions.coco.json \
  --locator-model wkdrs/cv801-yolo26x/train-357714/weights/best.pt
"$PYTHON_HOME/bin/python" CV801/scripts/prepare_golden_test_inputs.py audit --require-t2
```

生成的 `writer_t2.jsonl` 仍只有 `system + user`，与 T0/T1 的差异仅为冻结 YOLO prediction hints。`manifest_t2.jsonl` 保存预测框以供审计，但不含 reference box 或覆盖率标签。

## 检测 operating-point 指标

COCO AP 使用低阈值预测集合；Precision、Recall、F1 和 TP mean IoU 另按 validation 冻结的 score threshold 计算，禁止在 Golden Test 上选阈值。统一匹配口径为 class-aware、score-descending、one-to-one greedy matching，IoU 阈值为 0.5。当前冻结值为 YOLO26x `0.404`、Grounding DINO Tiny `0.25091278553009033`、LocateAnything `0.2`；后者为无原生置信度时的固定分数，包含全部解码框。

## 可复现推理 DAG

以下脚本先执行 reference 泄漏检查并打印 DAG，默认不提交任何作业：

```bash
execs/scpts/cv801_golden_test_submit.sh
```

并行关系为：

```text
YOLO26x ─> T2 输入 ─> T2 Writer ─> Rc/Rd 四分片 ─> Judge ─> 选定 revision ─> recheck
   ├──── Grounding DINO test inference
   └──── LocateAnything test inference

T0 Writer ─┐
T1 Writer ─┴─ 与检测/T2 分支并行
```

validation 冻结后，明确填写选中的配置再提交：

```bash
PROTOCOL_FROZEN=YES \
REVIEW_CONFIGURATION=<rc|rd|rc_rd|full> \
SUBMIT=1 \
execs/scpts/cv801_golden_test_submit.sh
```

这个提交脚本：

- 并行启动 YOLO26x、Grounding DINO、LocateAnything 以及 T0/T1。
- YOLO 完成后构建无 reference 的 T2 输入，再运行 T2 Writer。
- 对 Rc 8B 和 Rd 38B 各切四个稳定连续分片，严格按原顺序合并并审计；Rd 每个分片显式申请 2×MI210 和 320 GB 内存，与 validation 已验证的 38B 运行配置一致。
- 只运行 validation 选定的 revision 配置；`full` 配置额外运行 Judge recheck。
- 不创建任何测试 metrics，不调用 `evaluate_coco_predictions.py`、`evaluate_vlm_predictions.py` 或 reference label attach。

主要启动器：

- `execs/scpts/cv801_golden_detection_infer.sh`：三种检测模型只推理，不评估。
- `execs/scpts/cv801_golden_writer_infer.sh`：T0/T1/T2 八卡稳定分片推理；输出后检查图片顺序、source messages 和空 `labels`。
- `execs/scpts/cv801_golden_test_stage.sh`：T2 input 和 reviewer input 的无 reference 数据准备。
- 现有 Agent Review 启动器继续负责 Rc、Rd、Judge、revision 和 recheck。

## 泄漏审计证据

`CV801/scripts/prepare_golden_test_inputs.py` 对以下条件强制失败：

- 样本数不是 1,000，或 image ID/path 不唯一、不对齐。
- Writer input 出现 assistant 消息或 `labels`、`annotations`、`reference_boxes` 等字段。
- detector input 出现 `annotations`。
- T2 除 predicted hints 以外改变基础 prompt、图片顺序或安全 manifest identity。
- Writer inference output 出现非空 `labels`、图片顺序变化或 source messages 变化。

现有 `prepare_agent_review_inputs.py` 只读取 Writer 的 `response` 和 image path；即使 ms-swift prediction record 存在其他字段，也不会把 `labels` 传入 Rc/Rd/J。相关回归测试同时覆盖 reviewer、Judge、revision 和 recheck 的 reference 隔离。

## 当前状态

- 基础 Golden Test 输入：已生成并通过 1,000/1,000 泄漏审计。
- validation 已冻结主线 `Rd-only` review 配置；三个检测器、T0/T1/T2 Writer 和 Golden Test 推理均已完成。
- Agent Review 的 Rc-only、Rd-only、Rc+Rd、Full、第二轮 revision/recheck、27B Writer baseline 与 Pixtral-12B 替代视觉 reviewer 均已完成。
- 独立 0–6 视觉评估、reference-based 最终指标、Reviewer/Judge/recheck 决策率审计和端到端效率统计均已汇总。
- 冻结结果发布在 `results/final_summary.complete.json`、`results/final_summary.complete.md` 和 `results/report/main.pdf`。
- Gemma 3 visual-reviewer 替换是后续增量实验；完成后追加结果，不覆盖上述冻结快照。
- 人工评估与 GPT-6 Astra 实验按项目决定取消，不属于未完成项。
