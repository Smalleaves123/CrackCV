# Run All Experiments

## 1. 环境准备

建议使用 Python 3.9 或以上，并先安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

如果你使用 conda：

```bash
conda env create -f environment.yml
conda activate crackcv
```

建议提前设置权重缓存目录：

```bash
export TORCH_HOME=./pretrained_weights/torch
export HF_HOME=./pretrained_weights/huggingface
export TIMM_HOME=./pretrained_weights/timm
mkdir -p pretrained_weights/torch pretrained_weights/timm pretrained_weights/huggingface pretrained_weights/manual
```

## 2. 数据准备

原始数据目录是只读输入：

```text
dataset/
```

先做备份，再做划分：

```bash
python3 Reproduction/scripts/00_backup_dataset.py --src dataset --dst data_work/raw_backup
python3 Reproduction/scripts/01_prepare_dataset.py --src data_work/raw_backup --out data_work/processed --splits-out data_work/splits --image-size 227 --seed 42
```

运行结果应包含：

```text
data_work/raw_backup/
data_work/processed/train|val|test/
data_work/splits/train.csv
data_work/splits/val.csv
data_work/splits/test.csv
```

## 3. 权重模式

项目支持三种模式：

- 联网自动下载：默认模式，首次加载模型时自动下载官方权重
- 本地权重：在配置中填写 `weights.local_weights_path`
- 离线模式：命令行追加 `--offline`，只使用本地权重或随机初始化

如果你需要完全复现论文设定，优先使用预训练权重，不建议全程 `--offline`。

## 4. 先跑单个实验检查链路

建议先跑一组最小实验确认环境无误，例如：

```bash
python3 Reproduction/scripts/02_train_single.py --config Reproduction/configs/mobilenetv2.yaml --strategy full_finetune_aug --epochs 5
python3 Reproduction/scripts/03_eval_single.py --config Reproduction/results/mobilenetv2/full_finetune_aug/logs/config.yaml --checkpoint Reproduction/results/mobilenetv2/full_finetune_aug/checkpoints/best.pt --split test
python3 Reproduction/scripts/06_predict_image.py --config Reproduction/results/mobilenetv2/full_finetune_aug/logs/config.yaml --checkpoint Reproduction/results/mobilenetv2/full_finetune_aug/checkpoints/best.pt --image data_work/processed/test/crack/raw_backup_Positive_00001.jpg
python3 Reproduction/scripts/07_gradcam.py --config Reproduction/results/mobilenetv2/full_finetune_aug/logs/config.yaml --checkpoint Reproduction/results/mobilenetv2/full_finetune_aug/checkpoints/best.pt --image data_work/processed/test/crack/raw_backup_Positive_00001.jpg
```

如果这里没问题，再进入全量实验。

说明：

- 当前 `01_prepare_dataset.py` 会把处理后的文件名改成带来源目录前缀的稳定名字，例如 `raw_backup_Positive_00001.jpg`
- 如果你的 `data_work/processed/` 是用更早版本脚本生成的，文件名可能不同，先用 `ls data_work/processed/test/crack/` 看实际名字

`02_train_single.py` 现在会真实应用四种策略，不只是改输出目录名。可选值只有这四个：

- `full_finetune_aug`
- `full_finetune_no_aug`
- `linear_probe_aug`
- `linear_probe_no_aug`

例如你想手动只跑 `mobilenetv2 + linear_probe_no_aug`：

```bash
python3 Reproduction/scripts/02_train_single.py \
  --config Reproduction/configs/mobilenetv2.yaml \
  --strategy linear_probe_no_aug \
  --epochs 5
```

这条命令会同时把：

- `model.freeze_mode` 设为 `linear_probe`
- `augmentation.enabled` 设为 `false`
- 输出目录设为 `Reproduction/results/mobilenetv2/linear_probe_no_aug/`

## 5. 运行 Reproduction 全部主实验

论文复现部分共有：

```text
6 models x 4 strategies = 24 experiments
```

完整命令：

```bash
python3 Reproduction/scripts/04_train_all.py --epochs 100
python3 Reproduction/scripts/05_eval_all.py --split test
python3 Reproduction/scripts/08_collect_results.py
```

如果你想先分批跑：

```bash
python3 Reproduction/scripts/04_train_all.py --models mobilenetv2 vgg16 --strategies full_finetune_aug linear_probe_aug --epochs 100
```

如果你不想一次性跑 24 组，推荐把 `04_train_all.py` 当成“批量跑子集”的入口来用。例如：

只跑一个模型的四种策略：

```bash
python3 Reproduction/scripts/04_train_all.py --models mobilenetv2 --epochs 100
```

只跑两个模型里的两种策略：

```bash
python3 Reproduction/scripts/04_train_all.py \
  --models mobilenetv2 vgg16 \
  --strategies full_finetune_aug linear_probe_no_aug \
  --epochs 100
```

只想先限制前几组做 smoke test：

```bash
python3 Reproduction/scripts/04_train_all.py \
  --models mobilenetv2 vgg16 \
  --strategies full_finetune_aug full_finetune_no_aug linear_probe_aug linear_probe_no_aug \
  --epochs 3 \
  --limit 2
```

主要输出目录：

```text
Reproduction/results/{model}/{strategy}/
report_assets/tables/reproduction_summary.csv
report_assets/tables/reproduction_best_by_model.csv
report_assets/figures/model_accuracy_comparison.png
report_assets/figures/model_f1_comparison.png
report_assets/figures/parameter_vs_accuracy.png
report_assets/figures/best_strategy_per_model.png
```

## 6. 运行 Addition 1：Extra Models

新增模型实验：

```bash
python3 Addition/01_extra_models/scripts/run_extra_models.py --epochs 100
```

如果只跑一个模型或部分策略：

```bash
python3 Addition/01_extra_models/scripts/run_extra_models.py --models resnet50 --strategies full_finetune_aug linear_probe_aug --epochs 100
```

输出目录：

```text
Addition/01_extra_models/results/{model}/{strategy}/
```

## 7. 运行 Addition 2：Hyperparameter Study

默认扫描 `mobilenetv2` 的学习率和 batch size：

```bash
python3 Addition/02_hyperparameter_study/scripts/run_hyperparameter_study.py --epochs 100
```

配置文件在：

```text
Addition/02_hyperparameter_study/configs/mobilenetv2_lr_sweep.yaml
```

你可以先改里面的：

- `learning_rates`
- `batch_sizes`
- `freeze_mode`
- `augmentation_enabled`

输出目录：

```text
Addition/02_hyperparameter_study/results/lr_*_bs_*/
Addition/02_hyperparameter_study/results/study_summary.json
```

## 8. 运行 Addition 3：Train From Scratch

从零训练和迁移学习对比：

```bash
python3 Addition/03_train_from_scratch/scripts/run_scratch_comparison.py --epochs 100
```

输出目录：

```text
Addition/03_train_from_scratch/results/small_cnn_from_scratch/
Addition/03_train_from_scratch/results/mobilenetv2_pretrained/
```

## 9. 推荐执行顺序

推荐按这个顺序跑：

1. `00_backup_dataset.py`
2. `01_prepare_dataset.py`
3. `02_train_single.py` 跑 `mobilenetv2/full_finetune_aug`
4. `03_eval_single.py`
5. `07_gradcam.py`
6. `04_train_all.py`
7. `05_eval_all.py`
8. `08_collect_results.py`
9. `Addition/01_extra_models/scripts/run_extra_models.py`
10. `Addition/02_hyperparameter_study/scripts/run_hyperparameter_study.py`
11. `Addition/03_train_from_scratch/scripts/run_scratch_comparison.py`

如果你平时不会一次性全跑，建议按下面这个更实用的节奏：

1. `02_train_single.py` 先确认单组策略正确
2. `04_train_all.py --models <one_model>` 跑单个模型的四组
3. `04_train_all.py --models <two_or_three_models>` 逐步扩展
4. 最后再补齐剩余模型，凑满 24 组

## 10. 常见注意事项

- `xception` 和 `inception_resnet_v2` 依赖 `timm`
- 没有外网时，`timm` 和 `torchvision` 预训练权重不会自动下载
- 使用 `--offline` 时，除非你已经准备好本地权重，否则模型会随机初始化
- 如果你要完整保留论文对比意义，主实验、额外模型和迁移学习对比都建议保留预训练模式
- 当前主汇总脚本只汇总 `Reproduction/`，`Addition/` 结果需要分别查看各自 `results/`
- AMP 现在同时兼容新旧 PyTorch；如果你仍看到 `torch.amp` 或 `torch.cuda.amp` 相关 warning，说明远端机器跑的不是当前代码版本
