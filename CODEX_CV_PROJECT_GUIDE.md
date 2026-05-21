# CODEX 项目指引：PyTorch 复现与扩展 Brickwork Crack Detection 论文

> 本文档用于指导 Codex 完成一个完整的 CV Final Project。项目目标是使用 PyTorch / torchvision / timm 复现论文 **Deep learning for crack detection on masonry façades using limited data and transfer learning**，并在复现基础上完成三个扩展实验。
>
> 严格要求：不使用 TensorFlow / Keras；不污染原始 `dataset/` 数据集；所有处理都基于备份数据；完整保存训练过程、评估指标、图表、模型权重和可复现实验配置。

---

## 0. 项目背景与总体目标

### 0.1 论文任务

论文任务是砖砌体外立面裂缝二分类：

```text
输入：一张砖墙局部图像
输出：crack 或 non_crack
```

论文数据集包含：

```text
crack: 350 images
non_crack: 350 images
total: 700 images
train: 500
val: 100
test: 100
```

论文主要使用 ImageNet 预训练 CNN 进行 transfer learning，比较了六个模型：

```text
VGG16
VGG19
MobileNetV2
InceptionV3
InceptionResNetV2
Xception
```

论文比较了四种训练策略：

```text
1. end-to-end training + data augmentation
2. end-to-end training + no data augmentation
3. frozen convolutional base + data augmentation
4. frozen convolutional base + no data augmentation
```

### 0.2 本项目目标

本项目不是简单跑一个分类模型，而是完整 CV 工程：

```text
数据备份
-> 数据预处理
-> 数据划分
-> 模型构建
-> 训练
-> 验证
-> 测试
-> 可视化
-> Grad-CAM
-> Demo 推理
-> 扩展实验
-> 报告素材输出
```

本项目分为四部分：

```text
1. Reproduction：完整复现论文六个模型和四种策略
2. Addition/01_extra_models：增加两个模型并跑相同实验
3. Addition/02_hyperparameter_study：对同一模型做关键参数敏感性分析
4. Addition/03_train_from_scratch：对小模型从零训练，并与预训练模型对比
```

---

## 1. 强制技术约束

### 1.1 禁止使用

Codex 不允许使用：

```text
TensorFlow
Keras
tf.keras
任何 TensorFlow 相关库
```

### 1.2 允许使用

允许使用：

```text
Python >= 3.9
PyTorch
torchvision
timm
numpy
pandas
scikit-learn
matplotlib
Pillow
opencv-python，可选
tqdm
PyYAML
seaborn，可选；如果画图最好优先 matplotlib
```

### 1.3 数据保护要求

原始数据目录：

```text
dataset/
```

必须只读，不允许修改、移动、删除里面的任何文件。

所有处理必须基于：

```text
data_work/raw_backup/
```

---

## 2. 项目最终目录结构

Codex 必须创建如下结构：

```text
BrickworkCrackProject/
├── dataset/                         # 用户已有原始数据，禁止修改
│   ├── crack/
│   └── non_crack/
│
├── data_work/                       # 自动生成
│   ├── raw_backup/                  # dataset 的完整备份
│   ├── processed/                   # 处理和划分后的数据
│   │   ├── train/
│   │   │   ├── crack/
│   │   │   └── non_crack/
│   │   ├── val/
│   │   │   ├── crack/
│   │   │   └── non_crack/
│   │   └── test/
│   │       ├── crack/
│   │       └── non_crack/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── Reproduction/
│   ├── configs/
│   │   ├── base.yaml
│   │   ├── vgg16.yaml
│   │   ├── vgg19.yaml
│   │   ├── mobilenetv2.yaml
│   │   ├── inceptionv3.yaml
│   │   ├── inception_resnet_v2.yaml
│   │   ├── xception.yaml
│   │   └── all_models.yaml
│   │
│   ├── src/
│   │   ├── data/
│   │   │   ├── backup_dataset.py
│   │   │   ├── prepare_dataset.py
│   │   │   ├── dataset.py
│   │   │   └── transforms.py
│   │   │
│   │   ├── models/
│   │   │   ├── model_factory.py
│   │   │   └── model_utils.py
│   │   │
│   │   ├── engine/
│   │   │   ├── trainer.py
│   │   │   ├── evaluator.py
│   │   │   └── predictor.py
│   │   │
│   │   ├── utils/
│   │   │   ├── config.py
│   │   │   ├── seed.py
│   │   │   ├── metrics.py
│   │   │   ├── logger.py
│   │   │   ├── checkpoints.py
│   │   │   ├── plotting.py
│   │   │   └── weights.py
│   │   │
│   │   └── explain/
│   │       └── grad_cam.py
│   │
│   ├── scripts/
│   │   ├── 00_backup_dataset.py
│   │   ├── 01_prepare_dataset.py
│   │   ├── 02_train_single.py
│   │   ├── 03_eval_single.py
│   │   ├── 04_train_all.py
│   │   ├── 05_eval_all.py
│   │   ├── 06_predict_image.py
│   │   ├── 07_gradcam.py
│   │   └── 08_collect_results.py
│   │
│   ├── results/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── metrics/
│   │   ├── figures/
│   │   ├── predictions/
│   │   └── gradcam/
│   │
│   └── README.md
│
├── Addition/
│   ├── 01_extra_models/
│   │   ├── configs/
│   │   ├── scripts/
│   │   ├── results/
│   │   └── README.md
│   │
│   ├── 02_hyperparameter_study/
│   │   ├── configs/
│   │   ├── scripts/
│   │   ├── results/
│   │   └── README.md
│   │
│   └── 03_train_from_scratch/
│       ├── configs/
│       ├── scripts/
│       ├── results/
│       └── README.md
│
├── pretrained_weights/
│   ├── torch/
│   ├── timm/
│   └── manual/
│
├── report_assets/
│   ├── tables/
│   ├── figures/
│   └── demo_images/
│
├── requirements.txt
├── environment.yml
├── README.md
├── PROJECT_GUIDE.md
└── CODEX_CV_PROJECT_GUIDE.md
```

---

## 3. 数据处理流程

### 3.1 原始数据格式假设

用户已经下载好数据集，位于：

```text
dataset/
```

里面至少包含两类图片，常见结构可能是：

```text
dataset/
├── crack/
└── non_crack/
```

也可能是：

```text
dataset/
├── Crack/
└── Non-crack/
```

或者：

```text
dataset/
├── Positive/
└── Negative/
```

Codex 需要写代码自动兼容类别目录名，并统一映射为：

```text
crack -> label 1
non_crack -> label 0
```

### 3.2 数据备份脚本

创建：

```text
Reproduction/src/data/backup_dataset.py
Reproduction/scripts/00_backup_dataset.py
```

功能：

```text
复制 dataset/ 到 data_work/raw_backup/
如果 raw_backup 已存在，检查文件数量，不重复覆盖，除非用户传 --force
保留原始文件名
输出备份日志
```

命令：

```bash
python Reproduction/scripts/00_backup_dataset.py \
  --src dataset \
  --dst data_work/raw_backup
```

### 3.3 数据划分脚本

创建：

```text
Reproduction/src/data/prepare_dataset.py
Reproduction/scripts/01_prepare_dataset.py
```

功能：

```text
1. 从 data_work/raw_backup 读取图片
2. 自动识别 crack / non_crack 类别
3. 分层随机划分 train / val / test
4. 默认按论文数量划分：500 / 100 / 100
5. 如果数据总量不是 700，则按比例 71.4% / 14.3% / 14.3%
6. 每个 split 保持类别尽可能平衡
7. 将图片复制到 data_work/processed/train|val|test/class_name/
8. 生成 data_work/splits/train.csv、val.csv、test.csv
9. 所有图片统一转 RGB，并 resize 到默认 227x227 后保存
```

命令：

```bash
python Reproduction/scripts/01_prepare_dataset.py \
  --src data_work/raw_backup \
  --out data_work/processed \
  --splits-out data_work/splits \
  --image-size 227 \
  --seed 42
```

### 3.4 数据增强

训练集增强：

```text
RandomHorizontalFlip
RandomVerticalFlip
RandomRotation(degrees=(0, 45))
ColorJitter brightness range approximately 0.3 to 1.0
ToTensor
Normalize
```

验证集和测试集：

```text
Resize 或已 resize
ToTensor
Normalize
```

注意：增强只在 DataLoader 运行时对 train split 做，不要把增强图片写回硬盘。

---

## 4. 模型设计

### 4.1 模型来源

使用 `torchvision.models` 和 `timm.create_model`。

优先规则：

```text
VGG16: torchvision.models.vgg16
VGG19: torchvision.models.vgg19
MobileNetV2: torchvision.models.mobilenet_v2
InceptionV3: torchvision.models.inception_v3 或 timm
InceptionResNetV2: timm
Xception: timm
```

新增模型：

```text
ResNet50: torchvision.models.resnet50
EfficientNet-B0: torchvision.models.efficientnet_b0 或 timm
```

### 4.2 输出层

所有模型最终必须输出：

```text
num_classes = 2
```

标签：

```text
non_crack = 0
crack = 1
```

### 4.3 model_factory.py 要求

实现函数：

```python
def create_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    local_weights_path: str | None = None,
    freeze_backbone: bool = False,
    partial_unfreeze: str | None = None,
) -> torch.nn.Module:
    ...
```

支持模型名：

```text
vgg16
vgg19
mobilenetv2
inceptionv3
inception_resnet_v2
xception
resnet50
efficientnet_b0
small_cnn
```

### 4.4 冻结策略

必须支持三种训练方式：

```text
full_finetune：所有参数 requires_grad=True
linear_probe：冻结 backbone，只训练 classifier
partial_finetune：冻结大部分层，只解冻最后若干 block 和 classifier
```

论文四策略对应：

```text
end-to-end + augmentation       -> full_finetune_aug
end-to-end + no augmentation    -> full_finetune_no_aug
frozen base + augmentation      -> linear_probe_aug
frozen base + no augmentation   -> linear_probe_no_aug
```

---

## 5. 预训练权重下载、缓存与超时处理

### 5.1 必须支持的加载模式

代码必须支持：

```text
1. 自动下载 torchvision / timm 官方权重
2. 从 pretrained_weights/manual/ 加载本地权重
3. 离线模式，不下载，只使用本地权重或随机初始化
4. 下载失败时给清晰错误提示，不要让程序无说明崩溃
```

### 5.2 环境变量

在 README 中说明可设置：

```bash
export TORCH_HOME=./pretrained_weights/torch
export HF_HOME=./pretrained_weights/huggingface
export TIMM_HOME=./pretrained_weights/timm
```

Windows PowerShell：

```powershell
$env:TORCH_HOME="./pretrained_weights/torch"
$env:HF_HOME="./pretrained_weights/huggingface"
$env:TIMM_HOME="./pretrained_weights/timm"
```

### 5.3 下载失败处理

实现：

```text
try auto download
except timeout / connection error:
    print manual download instruction
    print expected local path
    if allow_random_init is true:
        continue with random initialization
    else:
        raise a clear RuntimeError
```

### 5.4 配置项

配置中加入：

```yaml
weights:
  pretrained: true
  offline_mode: false
  local_weights_path: null
  allow_random_init_when_download_fails: false
  cache_dir: ./pretrained_weights
```

---

## 6. 训练配置

### 6.1 base.yaml

创建：

```yaml
project:
  name: brickwork_crack_detection
  seed: 42
  device: cuda

data:
  processed_dir: ../../data_work/processed
  train_dir: ../../data_work/processed/train
  val_dir: ../../data_work/processed/val
  test_dir: ../../data_work/processed/test
  image_size: 227
  num_workers: 4
  class_names: [non_crack, crack]

model:
  name: mobilenetv2
  num_classes: 2
  pretrained: true
  freeze_mode: full_finetune

training:
  epochs: 100
  batch_size: 32
  optimizer: adam
  learning_rate: 1.0e-5
  weight_decay: 0.0
  loss: cross_entropy
  early_stopping:
    monitor: val_accuracy
    patience: 20
    mode: max

augmentation:
  enabled: true
  horizontal_flip: true
  vertical_flip: true
  rotation_degrees: 45
  brightness_min: 0.3
  brightness_max: 1.0

outputs:
  result_dir: results
  save_best: true
  save_last: true
  save_curves: true
  save_confusion_matrix: true
  save_predictions: true
```

### 6.2 单模型配置

每个模型一个 yaml，覆盖 `model.name`。

例如：

```yaml
model:
  name: vgg16
  pretrained: true
  freeze_mode: full_finetune
```

---

## 7. Reproduction 主实验设计

### 7.1 六个论文模型

必须跑：

```text
vgg16
vgg19
mobilenetv2
inceptionv3
inception_resnet_v2
xception
```

### 7.2 四种策略

每个模型跑：

```text
full_finetune_aug
full_finetune_no_aug
linear_probe_aug
linear_probe_no_aug
```

共：

```text
6 models x 4 strategies = 24 experiments
```

### 7.3 每个实验输出目录

格式：

```text
Reproduction/results/{model_name}/{strategy}/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── logs/
│   ├── train_log.csv
│   └── config.yaml
├── metrics/
│   ├── val_metrics.json
│   ├── test_metrics.json
│   ├── classification_report.txt
│   └── predictions.csv
└── figures/
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── f1_curve.png
    └── confusion_matrix.png
```

---

## 8. 训练代码要求

### 8.1 trainer.py

必须实现：

```python
class Trainer:
    def __init__(self, config): ...
    def fit(self): ...
    def train_one_epoch(self, epoch): ...
    def validate(self, epoch): ...
    def save_checkpoint(self, is_best): ...
```

每个 epoch 记录：

```text
epoch
train_loss
train_accuracy
train_precision
train_recall
train_f1
val_loss
val_accuracy
val_precision
val_recall
val_f1
learning_rate
epoch_time
```

### 8.2 mixed precision

可以支持但不强制：

```yaml
training:
  amp: true
```

如果 GPU 显存较小，AMP 有利于降低显存。

### 8.3 early stopping

监控：

```text
val_accuracy
```

如果 20 个 epoch 没有提升，则停止。

### 8.4 checkpoint

保存：

```text
best.pt
last.pt
```

checkpoint 内容：

```python
{
  "epoch": epoch,
  "model_state_dict": model.state_dict(),
  "optimizer_state_dict": optimizer.state_dict(),
  "best_metric": best_val_accuracy,
  "config": config,
  "class_names": ["non_crack", "crack"]
}
```

---

## 9. 评估代码要求

### 9.1 evaluator.py

必须计算：

```text
accuracy
precision
recall
f1_score
jaccard_index
confusion_matrix
classification_report
```

使用 sklearn：

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix, classification_report
```

二分类平均方式：

```python
average="macro"
```

同时保存每张图预测：

```text
image_path
true_label
pred_label
prob_non_crack
prob_crack
correct
```

### 9.2 测试命令

```bash
python Reproduction/scripts/03_eval_single.py \
  --config Reproduction/results/mobilenetv2/full_finetune_aug/logs/config.yaml \
  --checkpoint Reproduction/results/mobilenetv2/full_finetune_aug/checkpoints/best.pt \
  --split test
```

---

## 10. 可视化要求

### 10.1 训练曲线

每个实验保存：

```text
loss_curve.png
accuracy_curve.png
f1_curve.png
```

### 10.2 混淆矩阵

保存：

```text
confusion_matrix.png
```

坐标标签：

```text
non_crack
crack
```

### 10.3 汇总图

`08_collect_results.py` 生成：

```text
report_assets/tables/reproduction_summary.csv
report_assets/figures/model_accuracy_comparison.png
report_assets/figures/model_f1_comparison.png
report_assets/figures/parameter_vs_accuracy.png
```

---

## 11. Grad-CAM 要求

### 11.1 文件

创建：

```text
Reproduction/src/explain/grad_cam.py
Reproduction/scripts/07_gradcam.py
```

### 11.2 功能

输入：

```text
checkpoint
model_name
image_path 或 image_dir
class target，可选
```

输出：

```text
原图
heatmap
overlay 图
```

保存到：

```text
Reproduction/results/{model}/{strategy}/gradcam/
```

### 11.3 默认模型

优先对：

```text
MobileNetV2 full_finetune_aug
```

做 Grad-CAM，因为它参数小、论文推荐、效果好。

---

## 12. Demo 推理要求

### 12.1 predict.py

命令：

```bash
python Reproduction/scripts/06_predict_image.py \
  --model mobilenetv2 \
  --checkpoint Reproduction/results/mobilenetv2/full_finetune_aug/checkpoints/best.pt \
  --image demo.jpg
```

输出：

```text
Prediction: crack
Confidence: 0.9732
Probabilities:
  non_crack: 0.0268
  crack: 0.9732
```

### 12.2 可选 Gradio

如果时间允许，加：

```text
Reproduction/scripts/09_gradio_demo.py
```

功能：上传图片，输出类别、概率、Grad-CAM。

---

## 13. Addition 1：新增两个模型

### 13.1 目标

在论文六个模型基础上增加：

```text
ResNet50
EfficientNet-B0
```

并跑同样四种策略：

```text
full_finetune_aug
full_finetune_no_aug
linear_probe_aug
linear_probe_no_aug
```

总计：

```text
2 x 4 = 8 experiments
```

### 13.2 输出

保存到：

```text
Addition/01_extra_models/results/{model}/{strategy}/
```

同时把结果汇总进：

```text
report_assets/tables/extra_models_summary.csv
report_assets/figures/extra_models_comparison.png
```

### 13.3 需要在报告中回答

```text
ResNet50 是否优于 VGG 和 MobileNetV2？
EfficientNet-B0 是否能在较小参数量下获得高性能？
新增模型和论文六个模型相比有什么优势和不足？
```

---

## 14. Addition 2：超参数敏感性分析

### 14.1 主模型

默认选择：

```text
MobileNetV2
```

原因：论文表现好、参数量小、适合部署。

### 14.2 不要做全排列

不要把所有参数组合全排列，否则实验太多。

采用 one-factor-at-a-time：一次只改一个参数，其他保持论文默认。

论文默认：

```text
model = MobileNetV2
image_size = 227
batch_size = 32
learning_rate = 1e-5
optimizer = Adam
augmentation = paper augmentation
freeze_mode = full_finetune
```

### 14.3 必做实验

#### A. Learning rate study

```text
1e-3
1e-4
1e-5
1e-6
```

#### B. Batch size study

```text
16
32
64
```

#### C. Image size study

```text
160
224
227
299
```

#### D. Augmentation study

```text
none
weak
paper
strong
```

定义：

```text
none: no random augmentation
weak: horizontal flip only
paper: horizontal flip + vertical flip + rotation 45 + brightness 0.3-1.0
strong: paper + random crop/resize + color jitter stronger，可选
```

#### E. Fine-tuning strategy study

```text
linear_probe
partial_finetune
full_finetune
```

### 14.4 输出

保存到：

```text
Addition/02_hyperparameter_study/results/{study_name}/{setting}/
```

汇总：

```text
report_assets/tables/hyperparameter_study_summary.csv
report_assets/figures/lr_study.png
report_assets/figures/batch_size_study.png
report_assets/figures/image_size_study.png
report_assets/figures/augmentation_study.png
report_assets/figures/finetune_strategy_study.png
```

### 14.5 报告分析重点

必须分析：

```text
学习率过大是否不稳定？
学习率过小是否收敛慢？
batch size 是否影响泛化？
图像尺寸增加是否带来收益？
数据增强是否减轻过拟合？
full fine-tune 是否优于 frozen backbone？
```

---

## 15. Addition 3：from-scratch 对比

### 15.1 目标

比较：

```text
pretrained transfer learning
vs
from-scratch training
```

证明小数据集下迁移学习的必要性。

### 15.2 模型

建议跑：

```text
SmallCustomCNN
MobileNetV2
EfficientNet-B0
```

其中：

```text
SmallCustomCNN 从零训练
MobileNetV2 跑 pretrained 和 scratch 两种
EfficientNet-B0 跑 pretrained 和 scratch 两种
```

不要从零训练 VGG19、InceptionResNetV2，太大、太慢、过拟合明显。

### 15.3 SmallCustomCNN 结构

实现一个轻量 CNN：

```text
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-MaxPool
GlobalAveragePooling
Dropout
Linear(2)
```

### 15.4 输出

```text
Addition/03_train_from_scratch/results/{model}/{pretrained_or_scratch}/
```

汇总：

```text
report_assets/tables/from_scratch_summary.csv
report_assets/figures/pretrained_vs_scratch.png
```

### 15.5 报告分析重点

必须分析：

```text
从零训练是否过拟合？
预训练模型是否更快收敛？
预训练模型是否测试集 F1 更高？
小模型是否更适合小数据集？
```

---

## 16. requirements.txt

生成：

```text
torch
torchvision
timm
numpy
pandas
scikit-learn
matplotlib
Pillow
tqdm
PyYAML
opencv-python
```

可选：

```text
gradio
```

---

## 17. README.md 必须包含

主 README 至少包含：

```text
1. Project overview
2. Dataset structure
3. Environment setup
4. Data backup and preprocessing
5. Reproduction experiments
6. Addition experiments
7. Evaluation and visualization
8. Prediction demo
9. Pretrained weight troubleshooting
10. Expected outputs
```

### 17.1 README 运行流程

必须写清楚：

```bash
# 1. install
pip install -r requirements.txt

# 2. backup dataset
python Reproduction/scripts/00_backup_dataset.py --src dataset --dst data_work/raw_backup

# 3. prepare dataset
python Reproduction/scripts/01_prepare_dataset.py --src data_work/raw_backup --out data_work/processed --splits-out data_work/splits --image-size 227

# 4. train one model
python Reproduction/scripts/02_train_single.py --config Reproduction/configs/mobilenetv2.yaml --strategy full_finetune_aug

# 5. evaluate
python Reproduction/scripts/03_eval_single.py --checkpoint Reproduction/results/mobilenetv2/full_finetune_aug/checkpoints/best.pt --split test

# 6. train all reproduction experiments
python Reproduction/scripts/04_train_all.py --config Reproduction/configs/all_models.yaml

# 7. collect results
python Reproduction/scripts/08_collect_results.py

# 8. predict image
python Reproduction/scripts/06_predict_image.py --model mobilenetv2 --checkpoint path/to/best.pt --image path/to/image.jpg
```

---

## 18. 代码质量要求

### 18.1 通用要求

```text
所有脚本必须支持 --help
所有路径必须可配置，不能硬编码绝对路径
所有随机过程必须设置 seed
所有结果必须保存
所有实验必须保存 config.yaml
训练中断后最好能从 checkpoint 恢复，可选
```

### 18.2 异常处理

必须处理：

```text
dataset 不存在
类别文件夹不存在
图片损坏
GPU 不可用
预训练权重下载失败
checkpoint 不存在
配置字段缺失
```

### 18.3 日志

每个实验必须保存：

```text
train_log.csv
console.log，可选
config.yaml
test_metrics.json
```

---

## 19. 最终报告素材输出

Codex 最后需要保证 `report_assets/` 里有：

```text
report_assets/
├── tables/
│   ├── reproduction_summary.csv
│   ├── extra_models_summary.csv
│   ├── hyperparameter_study_summary.csv
│   └── from_scratch_summary.csv
│
├── figures/
│   ├── model_accuracy_comparison.png
│   ├── model_f1_comparison.png
│   ├── confusion_matrix_best_model.png
│   ├── loss_curve_best_model.png
│   ├── accuracy_curve_best_model.png
│   ├── lr_study.png
│   ├── batch_size_study.png
│   ├── image_size_study.png
│   ├── augmentation_study.png
│   ├── finetune_strategy_study.png
│   ├── pretrained_vs_scratch.png
│   └── gradcam_examples.png
│
└── demo_images/
    ├── example_crack_prediction.png
    └── example_non_crack_prediction.png
```

---

## 20. 推荐实现顺序

Codex 必须按以下顺序开发，不要一开始就写所有实验。

### Stage 1：项目骨架

```text
创建目录结构
写 requirements.txt
写 README 初版
写 config 读取工具
写 seed 工具
```

### Stage 2：数据流程

```text
backup_dataset.py
prepare_dataset.py
dataset.py
transforms.py
确认 DataLoader 可正常读取
```

### Stage 3：单模型跑通

```text
实现 model_factory.py
先跑 MobileNetV2 full_finetune_aug
保存 checkpoint、log、曲线、metrics
```

### Stage 4：复现论文主实验

```text
跑六个模型 x 四种策略
生成 reproduction_summary.csv
```

### Stage 5：新增模型

```text
跑 ResNet50 和 EfficientNet-B0
生成 extra_models_summary.csv
```

### Stage 6：超参数实验

```text
以 MobileNetV2 为主模型
按 study 分组跑
生成对应图表
```

### Stage 7：from-scratch 对比

```text
SmallCustomCNN
MobileNetV2 scratch vs pretrained
EfficientNet-B0 scratch vs pretrained
```

### Stage 8：解释性与 Demo

```text
Grad-CAM
predict.py
可选 Gradio
```

### Stage 9：整理报告素材

```text
collect_results.py
复制关键图表到 report_assets
更新 README
```

---

## 21. 验收标准

项目完成时，至少应该满足：

```text
1. dataset/ 未被修改
2. data_work/raw_backup 存在完整备份
3. data_work/processed 有 train/val/test 三个 split
4. MobileNetV2 单模型可以完整训练、评估、预测
5. Reproduction 至少完成六模型四策略中的主要实验
6. Addition 1 完成 ResNet50 和 EfficientNet-B0 对比
7. Addition 2 至少完成 learning rate、batch size、augmentation 三组 study
8. Addition 3 至少完成 SmallCustomCNN 和 MobileNetV2 scratch/pretrained 对比
9. 每个实验有 checkpoint、log、metrics、figures
10. report_assets 下有可直接用于报告和 PPT 的表格与图片
11. README 能让别人从零跑通项目
```

---

## 22. 最小可交付版本

如果时间不够，优先保证：

```text
1. 数据备份和预处理
2. MobileNetV2 论文参数复现
3. 六个模型至少跑 full_finetune_aug
4. ResNet50 / EfficientNet-B0 full_finetune_aug
5. MobileNetV2 learning rate study
6. MobileNetV2 scratch vs pretrained
7. 训练曲线 + 混淆矩阵 + Grad-CAM + predict.py
```

这样也能形成完整项目。

---

## 23. Codex 开发时的重要提醒

```text
不要污染 dataset/
不要使用 TensorFlow/Keras
不要只写 notebook
不要把所有结果只打印在终端
不要把路径写死
不要跳过评估指标
不要忘记保存每个实验配置
不要让预训练权重下载失败导致整个项目不可运行
不要把 validation/test 做随机增强
不要用 test set 调参
```

---

## 24. 最终项目定位

最终报告和展示可以把项目写成：

```text
A PyTorch Reproduction and Extension of Transfer Learning Based Brickwork Crack Detection
```

创新点：

```text
1. TensorFlow/Keras 到 PyTorch/timm 的工程化复现
2. 完整复现论文多模型、多策略实验
3. 增加 ResNet50 和 EfficientNet-B0 模型对比
4. 系统性超参数敏感性分析
5. 预训练迁移学习与从零训练对比
6. Grad-CAM 可解释性验证
7. 支持本地权重、离线模式和 demo 推理
```

