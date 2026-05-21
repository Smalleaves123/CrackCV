# Hyperparameter Study

关键参数敏感性实验目录，当前提供：

- `configs/mobilenetv2_lr_sweep.yaml`
- `scripts/run_hyperparameter_study.py`

默认对 `mobilenetv2` 扫描学习率和 batch size。

产物约定：

- 每组实验写入 `results/lr_*_bs_*/`
- 汇总写入 `results/study_summary.json`
