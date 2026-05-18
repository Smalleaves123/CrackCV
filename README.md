# CrackCV

基于论文 *Deep learning for crack detection on masonry façades using limited data and transfer learning* 的 PyTorch 复现工程。

当前仓库已有原始裁剪数据集：

- `dataset/Positive`: 裂缝图像，350 张
- `dataset/Negative`: 非裂缝图像，350 张

项目按论文设定完成二分类任务：

- `crack`
- `non-crack`

并且全部改为 `PyTorch + OpenCV` 实现，不使用 TensorFlow。

## 当前实现内容

- 自动把 `dataset/Positive|Negative` 分层划分为 `train/val/test = 500/100/100`
- 先完整备份原始 `dataset/` 到 `data/raw_backup/`，再在备份副本上做划分
- 如果备份数据本身已经带有 `train/val/test`，则直接复用该 split，不再二次重切分
- 支持 6 个 backbone：
  - `vgg16`
  - `vgg19`
  - `mobilenetv2`
  - `inceptionresnetv2`
  - `inceptionv3`
  - `xception`
- 支持 4 种训练策略：
  - `e2e_aug`
  - `e2e_no_aug`
  - `frozen_aug`
  - `frozen_no_aug`
- 输出：
  - 最佳模型
  - 训练历史
  - 训练曲线
  - 测试指标
  - 混淆矩阵
  - Grad-CAM 结果

## 目录结构

```text
CrackCV/
  dataset/
    Positive/
    Negative/
  data/
    raw_backup/
    processed/
      train/
      val/
      test/
  src/
    dataset.py
    models.py
    train.py
    evaluate.py
    gradcam.py
    utils.py
  scripts/
    run_all_experiments.py
  outputs/
  requirements.txt
```

## 环境

建议 Python `3.9+`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据映射

代码内部固定把原始目录映射为：

- `Positive -> crack`
- `Negative -> non-crack`

标签顺序固定为：

- `non-crack = 0`
- `crack = 1`

## 训练单个推荐模型

论文最推荐的是 `MobileNetV2 + end-to-end + augmentation`：

```bash
python -m src.train \
  --data-dir data/processed \
  --raw-data-dir dataset \
  --raw-backup-dir data/raw_backup \
  --backbone mobilenetv2 \
  --train-backbone true \
  --augmentation true \
  --batch-size 32 \
  --num-workers 0 \
  --rotation-mode positive \
  --lr 1e-5 \
  --max-epochs 200 \
  --patience 20 \
  --seed 42 \
  --refresh-raw-backup false \
  --output-dir outputs/mobilenetv2_e2e_aug
```

运行时数据处理顺序是：

```text
dataset/                  # 原始数据，只读，不参与后续写入
  -> data/raw_backup/     # 完整备份
  -> data/processed/      # 从备份数据生成 train / val / test
```

如果 `data/raw_backup` 或 `data/processed` 还不存在，脚本会自动生成：

```text
data/raw_backup/
  Positive
  Negative

data/processed/
  train/crack
  train/non-crack
  val/crack
  val/non-crack
  test/crack
  test/non-crack
```

## 评估

```bash
python -m src.evaluate \
  --model-path outputs/mobilenetv2_e2e_aug/best_model.pt \
  --data-dir data/processed \
  --batch-size 32 \
  --num-workers 0 \
  --output-dir outputs/mobilenetv2_e2e_aug
```

输出文件：

- `best_model.pt`
- `history.csv`
- `training_curves.png`
- `train_config.json`
- `metrics.json`
- `predictions.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`

## Grad-CAM

```bash
python -m src.gradcam \
  --model-path outputs/mobilenetv2_e2e_aug/best_model.pt \
  --image-dir data/processed/test/crack \
  --class-index 1 \
  --output-dir outputs/mobilenetv2_e2e_aug/gradcam
```

这里 `OpenCV` 会用于：

- 读取原图
- resize
- 生成热力图
- 叠加保存 Grad-CAM 可视化
- 自动选择 backbone 中最后一个真实的 4D 特征层，而不是简单取最后一个卷积层

## 跑全部 24 组实验

```bash
python scripts/run_all_experiments.py \
  --data-dir data/processed \
  --raw-data-dir dataset \
  --raw-backup-dir data/raw_backup \
  --output-root outputs \
  --batch-size 32 \
  --num-workers 0 \
  --rotation-mode positive \
  --lr 1e-5 \
  --max-epochs 200 \
  --patience 20 \
  --seed 42
```

每组实验输出到：

```text
outputs/{backbone}_{strategy}/
```

例如：

- `outputs/mobilenetv2_e2e_aug/`
- `outputs/vgg16_frozen_no_aug/`

## 与论文对齐的关键设置

- 输入尺寸：`227x227`
- 归一化：`0-255 -> 0-1`
- 输出层：2 类 logits
- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam(lr=1e-5)`
- 训练集增强：
  - 水平翻转
  - 垂直翻转
  - 亮度扰动
  - 旋转 `0` 到 `45` 度，默认 `--rotation-mode positive`
- 验证集和测试集不做随机增强
- 模型选择标准：`val_accuracy`
- 评估指标：
  - `accuracy`
  - `macro f1`
  - `macro precision`
  - `macro recall`
  - `macro jaccard`

## 主要代码入口

- [src/dataset.py](/Users/aiziqi/Desktop/CrackCV/src/dataset.py:1)
- [src/models.py](/Users/aiziqi/Desktop/CrackCV/src/models.py:1)
- [src/train.py](/Users/aiziqi/Desktop/CrackCV/src/train.py:1)
- [src/evaluate.py](/Users/aiziqi/Desktop/CrackCV/src/evaluate.py:1)
- [src/gradcam.py](/Users/aiziqi/Desktop/CrackCV/src/gradcam.py:1)
- [scripts/run_all_experiments.py](/Users/aiziqi/Desktop/CrackCV/scripts/run_all_experiments.py:1)

## 原始数据保护

- 原始 `dataset/` 现在视为只读输入目录。
- 训练脚本不会在 `dataset/` 里创建、删除或改写任何文件。
- 默认只在 `data/raw_backup/` 和 `data/processed/` 下写文件。
- 如果你想强制用当前最新的 `dataset/` 重新生成备份，可以在训练时加：

```bash
--refresh-raw-backup true
```

- 当你刷新原始备份时，脚本也会同步重建 `data/processed/`，避免继续使用旧 split。
