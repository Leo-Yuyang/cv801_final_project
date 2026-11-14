# 第三方模型与代码说明

本项目的训练和评估依赖下列公开模型或项目，但发布包不复制其未修改基础权重：

- Ultralytics YOLO26x
- `IDEA-Research/grounding-dino-tiny`
- `nvidia/LocateAnything-3B` 与 NVLabs Eagle
- `Qwen/Qwen3.5-9B` 和 27B baseline
- `OpenGVLab/InternVL3_5-8B-HF`、`OpenGVLab/InternVL3_5-38B-HF`
- `mistral-community/pixtral-12b`
- `google/gemma-3-12b-it`

微调权重、adapter 和兼容补丁不改变上述项目的原许可证。发布者必须在公开模型仓库前重新核对每个上游模型的许可、署名、可再分发范围和使用限制。LocateAnything 物料默认受非商业学术研究边界约束。
