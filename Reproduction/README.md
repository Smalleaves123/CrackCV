# Reproduction

完整运行顺序和全部实验说明见 [RUN_ALL_EXPERIMENTS.md](../RUN_ALL_EXPERIMENTS.md)。

常用命令：

```bash
python3 Reproduction/scripts/00_backup_dataset.py
python3 Reproduction/scripts/01_prepare_dataset.py
python3 Reproduction/scripts/02_train_single.py --config Reproduction/configs/mobilenetv2.yaml --epochs 1 --offline
python3 Reproduction/scripts/03_eval_single.py --config Reproduction/results/mobilenetv2/full_finetune_aug/logs/config.yaml --checkpoint Reproduction/results/mobilenetv2/full_finetune_aug/checkpoints/best.pt --split test
python3 Reproduction/scripts/06_predict_image.py --config Reproduction/results/mobilenetv2/full_finetune_aug/logs/config.yaml --checkpoint Reproduction/results/mobilenetv2/full_finetune_aug/checkpoints/best.pt --image data_work/processed/test/crack/dataset_Positive_00001.jpg
python3 Reproduction/scripts/04_train_all.py --epochs 100 --offline
python3 Reproduction/scripts/05_eval_all.py --split test
python3 Reproduction/scripts/08_collect_results.py
```

环境变量建议：

```bash
export TORCH_HOME=./pretrained_weights/torch
export HF_HOME=./pretrained_weights/huggingface
export TIMM_HOME=./pretrained_weights/timm
```

权重策略说明：

- `weights.offline_mode=true` 时不尝试联网下载预训练权重。
- `weights.local_weights_path` 可指定本地权重文件。
- `weights.allow_random_init_when_download_fails=true` 时，下载失败会退回随机初始化。
- `06_predict_image.py` 和 `07_gradcam.py` 的示例图片路径以当前数据准备脚本输出命名为准；如果你用旧版本处理过数据，请先查看 `data_work/processed/test/crack/` 下的实际文件名。

全量主实验：

- `04_train_all.py` 负责六个论文模型乘四种策略的 24 组训练
- `05_eval_all.py` 负责批量测试集评估
- `08_collect_results.py` 负责把 `Reproduction/results/` 汇总到 `report_assets/`
