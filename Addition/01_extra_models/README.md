# Extra Models

新增模型实验目录，包含：

- `configs/resnet50.yaml`
- `configs/efficientnet_b0.yaml`
- `scripts/run_extra_models.py`

脚本复用 `Reproduction` 的训练主干，对两个新增模型跑四种策略。

支持通过 `--models` 和 `--strategies` 选择子集实验。
