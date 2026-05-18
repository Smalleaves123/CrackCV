# 论文复现代码说明：砖砌体外立面裂缝检测 - 数据处理、训练与评估

> 目标：把论文 **Deep learning for crack detection on masonry façades using limited data and transfer learning** 的数据集处理、模型训练、评估流程整理成可直接交给 Claude/代码生成模型实现的工程规格说明。  
> 任务类型：二分类图像分类，输入为砖砌体图像块，输出为 `crack` 或 `non-crack`。

---

## 1. 论文任务与核心结论

论文要解决的是：在标注数据有限的情况下，使用迁移学习检测砖砌体外立面图像中是否存在裂缝。作者不是做像素级分割，也不是检测框输出，而是把预先裁剪出的砖砌体图像块作为样本，训练 CNN 二分类模型：

- `crack`：图像块中存在裂缝；
- `non-crack`：图像块中不存在裂缝。

核心方法是：

1. 构建并清洗一个砖砌体裂缝数据集；
2. 把原始图像中的裂缝区域和无损区域用 bounding box 标注出来；
3. 将每个 bounding box 对应的图像区域单独裁剪成样本图像；
4. 所有样本统一缩放到 `227 x 227` 像素；
5. 使用 ImageNet 预训练 CNN 作为 backbone；
6. 比较 6 种 CNN 架构与 4 种训练策略；
7. 使用验证集选择最佳 epoch 的模型，再在完全未见过的测试集上报告指标。

最佳推荐模型是：**MobileNetV2 + 端到端微调 + 训练集数据增强**。该模型测试集 Accuracy/F1-score 达到 `100%`，且参数量仅约 `2.26M`，适合部署到手机、平板、无人机等算力有限设备。

---

## 2. 数据集构建流程

### 2.1 数据来源

作者构建了一个名为 **Brickwork Cracks Dataset** 的数据集。数据来源分为两类：

#### 来源 A：University of Strathclyde 建筑学院外立面图像

这些图像来自 University of Strathclyde, Glasgow 的 Architecture School 外立面砖砌体墙面。图像包含：

- 有裂缝的砖砌体墙面；
- 无裂缝的砖砌体墙面；
- 带窗户、门等结构元素的墙面；
- 不同砖块颜色；
- 不同光照条件；
- 不同拍摄角度。

这些图像最初来源于一个用于建筑外立面缺陷检测的智能移动 App 项目，项目结合了虚拟现实、数字摄影测量和移动 App 技术，用于实时采集外立面缺陷检测数据。

#### 来源 B：在线来源

作者还从多个在线来源收集图像，包括：

- 在线图片数据库；
- 砖材制造商网站；
- 学术出版物中的图像。

收集到的在线图像经过质量筛选，只有适合砖砌体裂缝分类任务的图像才被纳入数据集。

---

## 3. 标注与样本生成

### 3.1 标注方式

作者没有直接把整张原始外立面图像作为分类样本，而是先在原始图像上进行 bounding box 标注。标注使用自建 annotation tool 完成。

标注对象包括两类：

1. **裂缝区域 bounding box**
   - 框住砖砌体中的裂缝；
   - 标注裂缝方向：`horizontal`、`vertical` 或 `diagonal`；
   - 但最终分类任务只使用 `crack` vs `non-crack` 二分类标签，方向标签不作为模型输出。

2. **无损砖砌体区域 bounding box**
   - 专家额外标注没有损伤的砖砌体区域；
   - 这些区域裁剪后作为 `non-crack` 类样本。

### 3.2 裁剪与缩放

对每一个标注框执行：

1. 从原始图像中裁剪 bounding box 对应的图像区域；
2. 将裁剪区域保存为单独图像；
3. 将图像缩放到固定尺寸：

```text
227 x 227 pixels
```

论文中所有模型均使用这些 `227 x 227` 的图像块作为输入。

### 3.3 数据规模与类别平衡

最终数据集包含：

| 类别 | 数量 |
|---|---:|
| `crack` | 350 |
| `non-crack` | 350 |
| 总计 | 700 |

数据集是严格平衡的：

```text
50% crack, 50% non-crack
```

---

## 4. 数据集划分

作者使用 **stratified sampling** 将 700 张图像随机划分为训练集、验证集和测试集。每个子集都保持类别平衡。

| 子集 | 总数量 | `crack` | `non-crack` | 占比 |
|---|---:|---:|---:|---:|
| Train | 500 | 250 | 250 | 约 71.4% |
| Validation | 100 | 50 | 50 | 约 14.3% |
| Test | 100 | 50 | 50 | 约 14.3% |

实现时必须注意：

- 训练集只用于训练；
- 验证集只用于选择最佳模型/最佳 epoch；
- 测试集只用于最终一次性评估；
- 数据增强只能作用在训练集；
- 验证集和测试集必须保持原图，不做随机增强；
- 由于样本来自原始大图的 bounding boxes，如果可获得原始图像 ID，最好按原始图像分组划分，避免同一原图的相似 crop 同时出现在训练和测试中。论文只说明使用 stratified sampling，未说明是否按原图分组。

推荐目录结构：

```text
dataset/
  train/
    crack/
    non-crack/
  val/
    crack/
    non-crack/
  test/
    crack/
    non-crack/
```

如果拿到的是官方 Zenodo 数据集而非原始大图，优先检查数据集中是否已经提供 train/val/test split。如果没有，则用固定随机种子进行分层划分。

---

## 5. 图像预处理

### 5.1 尺寸

所有输入图像统一为：

```python
IMG_SIZE = (227, 227)
```

### 5.2 像素归一化

论文明确说明：所有图像像素值在输入模型前按以下方式缩放：

```python
x = x / 255.0
```

注意：论文使用的是 `1/255` rescale，而不是各 backbone 对应的 `preprocess_input`。为了复现论文，应统一使用 `rescale=1./255`。

### 5.3 标签编码

二分类标签建议编码为：

```text
non-crack -> 0
crack     -> 1
```

但训练时作者使用 2 个输出神经元加 softmax，因此可使用 one-hot 标签：

```python
non-crack -> [1, 0]
crack     -> [0, 1]
```

损失函数使用 categorical cross-entropy。

---

## 6. 数据增强

作者为解决小数据集问题，对训练集使用 Keras `ImageDataGenerator` 做数据增强。

### 6.1 增强只用于训练集

增强图像只在训练阶段从训练集动态生成。验证集和测试集不做增强。测试集结果全部基于原始图像。

### 6.2 增强操作

每个训练样本在训练迭代中可随机执行以下操作：

| 操作 | 参数 |
|---|---|
| 水平翻转 | random horizontal flip |
| 垂直翻转 | random vertical flip |
| 亮度变化 | brightness randomly shifted between `0.3` and `1.0` |
| 旋转 | random rotation between `0°` and `45°` |
| 像素缩放 | rescale by `1/255` |

论文说明所有随机值来自 **uniform probability distribution**。

Keras 近似实现：

```python
train_datagen_aug = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.3, 1.0),
    rotation_range=45
)

train_datagen_no_aug = ImageDataGenerator(
    rescale=1.0 / 255.0
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)
```

说明：Keras 的 `rotation_range=45` 表示随机旋转范围通常为 `[-45°, +45°]`。论文文字写的是 `0°` 到 `45°`。为了严格复现，可自定义 augmentation 只采样 `[0, 45]`；为了贴近常规 Keras 用法，可使用 `rotation_range=45`。建议在代码配置中明确记录这一点。

---

## 7. 模型架构

作者比较了 6 个 ImageNet 预训练 CNN backbone：

| Backbone | 参数量，论文表 1 |
|---|---:|
| VGG16 | 134.27M |
| VGG19 | 139.58M |
| MobileNetV2 | 2.26M |
| InceptionResNetV2 | 54.34M |
| InceptionV3 | 21.81M |
| Xception | 20.87M |

所有模型都使用 ImageNet 预训练权重。

---

## 8. 分类头设计

### 8.1 非 VGG 模型的分类头

适用于：

- MobileNetV2；
- InceptionResNetV2；
- InceptionV3；
- Xception。

结构：

```text
Pretrained convolutional base
-> GlobalAveragePooling2D
-> Dense(2, activation='softmax')
```

Keras 示例：

```python
base_model = keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(227, 227, 3)
)

x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=base_model.input, outputs=outputs)
```

### 8.2 VGG16/VGG19 的分类头

作者为了公平遵循 VGG 原设计，没有使用 GAP + Dense(2)，而是使用：

```text
Pretrained convolutional base
-> Flatten
-> Dense(4096, activation='relu')
-> Dense(4096, activation='relu')
-> Dense(2, activation='softmax')
```

Keras 示例：

```python
base_model = keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(227, 227, 3)
)

x = base_model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(4096, activation="relu")(x)
x = keras.layers.Dense(4096, activation="relu")(x)
outputs = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=base_model.input, outputs=outputs)
```

---

## 9. 训练策略

作者对每个 backbone 都比较 4 种训练策略。

| 策略编号 | 训练方式 | 是否数据增强 |
|---:|---|---|
| 1 | 端到端训练，全模型可训练 | 是 |
| 2 | 端到端训练，全模型可训练 | 否 |
| 3 | 冻结 convolutional base，只训练新增分类层 | 是 |
| 4 | 冻结 convolutional base，只训练新增分类层 | 否 |

### 9.1 端到端训练

端到端训练时：

```python
base_model.trainable = True
```

这意味着 ImageNet 预训练 backbone 也会被微调。论文结果显示，端到端训练显著优于冻结 backbone。

### 9.2 冻结 convolutional base

冻结 backbone 时：

```python
base_model.trainable = False
```

此时只训练后面的分类层。该策略训练成本较低，但性能明显低于端到端微调。

---

## 10. 训练超参数

论文训练设置如下：

| 项目 | 设置 |
|---|---|
| 框架 | Python + Keras + TensorFlow |
| 损失函数 | Cross-Entropy / Categorical Crossentropy |
| 优化器 | Adam |
| 学习率 | `1e-5` |
| Batch size | `32` |
| Early stopping 监控指标 | validation accuracy |
| Early stopping patience | `20` epochs without validation accuracy improvement |
| 最佳模型选择 | 选择 validation accuracy 最好的 epoch |
| 测试方式 | 使用最佳验证集模型在 unseen test set 上评估 |

Keras 编译示例：

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

Callbacks 示例：

```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=20,
        restore_best_weights=True
    )
]
```

论文没有明确给出最大 epoch。实现时可以设置一个较大的 `max_epochs`，例如 `100` 或 `200`，依赖 early stopping 结束训练。

---

## 11. 损失函数

论文使用 Cross-Entropy：

```text
L_CE = - sum_{c=1}^{M} y_{o,c} log(p_{o,c})
```

其中：

- `M = 2`，即两个类别；
- `y_{o,c}` 是样本 `o` 属于类别 `c` 的 one-hot 标签；
- `p_{o,c}` 是模型预测样本 `o` 属于类别 `c` 的概率。

代码中直接使用：

```python
loss="categorical_crossentropy"
```

如果改用单输出 sigmoid，则可用 `binary_crossentropy`，但为复现论文，应使用 `Dense(2, softmax)` + `categorical_crossentropy`。

---

## 12. 评估流程

### 12.1 模型选择

每个实验配置执行：

1. 在训练集上训练；
2. 每个 epoch 在验证集上计算 validation accuracy；
3. 保存 validation accuracy 最高的模型；
4. 使用该最佳模型在测试集上评估；
5. 报告测试集指标。

### 12.2 指标

作者报告以下指标：

- Accuracy；
- F1-score；
- Precision；
- Recall；
- Jaccard index；
- Confusion matrix。

由于 F1、Precision、Recall 依赖正类定义，论文报告的是两个类别 `crack` 与 `non-crack` 的平均值。代码中建议使用 macro average：

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, confusion_matrix

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
jaccard = jaccard_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
```

推荐 confusion matrix 行列顺序：

```text
Rows:    actual    [non-crack, crack]
Columns: predicted [non-crack, crack]
```

---

## 13. 论文结果汇总

### 13.1 端到端训练 + 数据增强

| Base model | Accuracy | F1-score | Precision | Recall | Jaccard |
|---|---:|---:|---:|---:|---:|
| VGG16 | 0.9700 | 0.9700 | 0.9702 | 0.9700 | 0.9417 |
| VGG19 | 0.9900 | 0.9900 | 0.9902 | 0.9900 | 0.9802 |
| MobileNetV2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| InceptionResNetV2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| InceptionV3 | 0.9900 | 0.9900 | 0.9902 | 0.9900 | 0.9802 |
| Xception | 0.9700 | 0.9700 | 0.9717 | 0.9700 | 0.9417 |

对应 confusion matrices：

| Model | Matrix `[[TN, FP], [FN, TP]]` |
|---|---|
| VGG16 | `[[49, 1], [2, 48]]` |
| VGG19 | `[[49, 1], [0, 50]]` |
| MobileNetV2 | `[[50, 0], [0, 50]]` |
| InceptionResNetV2 | `[[50, 0], [0, 50]]` |
| InceptionV3 | `[[49, 1], [0, 50]]` |
| Xception | `[[47, 3], [0, 50]]` |

### 13.2 端到端训练 + 无数据增强

| Base model | Accuracy | F1-score | Precision | Recall | Jaccard |
|---|---:|---:|---:|---:|---:|
| VGG16 | 0.9900 | 0.9900 | 0.9902 | 0.9900 | 0.9802 |
| VGG19 | 0.9900 | 0.9900 | 0.9902 | 0.9900 | 0.9802 |
| MobileNetV2 | 0.9600 | 0.9600 | 0.9607 | 0.9600 | 0.9230 |
| InceptionResNetV2 | 0.9800 | 0.9800 | 0.9808 | 0.9800 | 0.9608 |
| InceptionV3 | 0.9900 | 0.9900 | 0.9902 | 0.9900 | 0.9802 |
| Xception | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

对应 confusion matrices：

| Model | Matrix `[[TN, FP], [FN, TP]]` |
|---|---|
| VGG16 | `[[50, 0], [1, 49]]` |
| VGG19 | `[[50, 0], [1, 49]]` |
| MobileNetV2 | `[[49, 1], [3, 47]]` |
| InceptionResNetV2 | `[[48, 2], [0, 50]]` |
| InceptionV3 | `[[49, 1], [0, 50]]` |
| Xception | `[[50, 0], [0, 50]]` |

### 13.3 冻结 convolutional base + 数据增强

| Base model | Accuracy | F1-score | Precision | Recall | Jaccard |
|---|---:|---:|---:|---:|---:|
| VGG16 | 0.9200 | 0.9199 | 0.9227 | 0.9200 | 0.8516 |
| VGG19 | 0.8500 | 0.8493 | 0.8570 | 0.8500 | 0.7382 |
| MobileNetV2 | 0.6800 | 0.6716 | 0.7005 | 0.6800 | 0.5079 |
| InceptionResNetV2 | 0.8300 | 0.8298 | 0.8312 | 0.8300 | 0.7092 |
| InceptionV3 | 0.7600 | 0.7478 | 0.8224 | 0.7600 | 0.6003 |
| Xception | 0.7400 | 0.7292 | 0.7857 | 0.7400 | 0.5766 |

对应 confusion matrices：

| Model | Matrix `[[TN, FP], [FN, TP]]` |
|---|---|
| VGG16 | `[[44, 6], [2, 48]]` |
| VGG19 | `[[39, 11], [4, 46]]` |
| MobileNetV2 | `[[26, 24], [8, 42]]` |
| InceptionResNetV2 | `[[43, 7], [10, 40]]` |
| InceptionV3 | `[[27, 23], [1, 49]]` |
| Xception | `[[27, 23], [3, 47]]` |

### 13.4 冻结 convolutional base + 无数据增强

| Base model | Accuracy | F1-score | Precision | Recall | Jaccard |
|---|---:|---:|---:|---:|---:|
| VGG16 | 0.9200 | 0.9200 | 0.9200 | 0.9200 | 0.8519 |
| VGG19 | 0.9300 | 0.9299 | 0.9316 | 0.9300 | 0.8691 |
| MobileNetV2 | 0.8100 | 0.8098 | 0.8111 | 0.8100 | 0.6805 |
| InceptionResNetV2 | 0.8100 | 0.8095 | 0.8131 | 0.8100 | 0.6801 |
| InceptionV3 | 0.8400 | 0.8394 | 0.8450 | 0.8400 | 0.7234 |
| Xception | 0.7900 | 0.7852 | 0.8187 | 0.7900 | 0.6475 |

对应 confusion matrices：

| Model | Matrix `[[TN, FP], [FN, TP]]` |
|---|---|
| VGG16 | `[[46, 4], [4, 46]]` |
| VGG19 | `[[48, 2], [5, 45]]` |
| MobileNetV2 | `[[39, 11], [8, 42]]` |
| InceptionResNetV2 | `[[38, 12], [7, 43]]` |
| InceptionV3 | `[[39, 11], [5, 45]]` |
| Xception | `[[32, 18], [3, 47]]` |

---

## 14. 论文对结果的解释

### 14.1 端到端微调明显优于冻结 backbone

冻结 convolutional base 时，模型只训练最后的全连接分类层。这样做训练成本低，但 backbone 仍然是 ImageNet 通用图像特征提取器，未适配砖砌体图像。

论文认为砖砌体图像与 ImageNet 自然图像在纹理、结构、颜色上差异明显，因此需要端到端微调，使 convolutional base 学会更适合区分砖墙裂缝的特征。

### 14.2 MobileNetV2 是推荐部署模型

虽然 MobileNetV2、InceptionResNetV2、Xception 在某些配置下都达到 100% 测试性能，但参数量差异很大：

- MobileNetV2：约 2.26M；
- Xception：约 20.87M；
- InceptionResNetV2：约 54.34M。

论文推荐 MobileNetV2，因为它在 `end-to-end + augmentation` 下达到最佳性能，同时模型更小，适合部署到：

- 手机；
- 平板；
- 手持检测设备；
- 商用 UAV/无人机。

### 14.3 Grad-CAM 可用于解释模型输出

作者使用 Grad-CAM 检查 MobileNetV2 模型是否真正关注裂缝区域。做法是：

1. 选择训练好的 MobileNetV2；
2. 使用最后一个 convolutional layer；
3. 生成 `crack` 类的 Grad-CAM heatmap；
4. 将 heatmap 叠加到原图；
5. 观察模型关注区域是否覆盖裂缝。

论文图 10 显示，模型激活区域主要落在裂缝处。这说明模型不是简单依赖颜色或背景，而是在使用裂缝相关视觉特征。实际部署时可以把 Grad-CAM heatmap 作为辅助输出，让工程师核验模型预测。

---

## 15. 推荐实现工程结构

建议 Claude 生成如下项目结构：

```text
brickwork-crack-detection/
  README.md
  requirements.txt
  configs/
    mobilenetv2_e2e_aug.yaml
    all_experiments.yaml
  data/
    raw/
    processed/
      train/
        crack/
        non-crack/
      val/
        crack/
        non-crack/
      test/
        crack/
        non-crack/
  src/
    dataset.py
    augment.py
    models.py
    train.py
    evaluate.py
    gradcam.py
    utils.py
  outputs/
    checkpoints/
    logs/
    metrics/
    confusion_matrices/
    gradcam/
```

---

## 16. Claude 需要实现的核心模块

### 16.1 `dataset.py`

功能：

- 从目录读取图像；
- 强制 resize 到 `227 x 227`；
- 执行 `1/255` rescale；
- 支持有增强和无增强两种训练 dataloader；
- 验证/测试 dataloader 只 rescale，不做增强；
- 输出 one-hot 标签。

Keras 可用 `flow_from_directory`：

```python
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(227, 227),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(227, 227),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(227, 227),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)
```

### 16.2 `models.py`

实现函数：

```python
def build_model(
    backbone_name: str,
    input_shape=(227, 227, 3),
    num_classes=2,
    train_backbone=True,
) -> keras.Model:
    ...
```

支持：

```text
vgg16
vgg19
mobilenetv2
inceptionresnetv2
inceptionv3
xception
```

逻辑：

- 加载 ImageNet 权重；
- `include_top=False`；
- VGG 使用 Flatten + Dense(4096) + Dense(4096) + Dense(2 softmax)；
- 其他模型使用 GAP + Dense(2 softmax)；
- 根据 `train_backbone` 设置 `base_model.trainable`；
- 使用 Adam `lr=1e-5` 编译。

### 16.3 `train.py`

需要支持命令行参数：

```bash
python -m src.train \
  --data-dir data/processed \
  --backbone mobilenetv2 \
  --train-strategy end_to_end \
  --augmentation true \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 200 \
  --patience 20 \
  --output-dir outputs/mobilenetv2_e2e_aug
```

训练逻辑：

1. 构造 train/val generator；
2. 构造模型；
3. 训练；
4. 使用 `ModelCheckpoint` 保存 validation accuracy 最好的模型；
5. 保存训练 history 到 CSV/JSON；
6. 绘制训练/验证 loss 和 accuracy 曲线。

### 16.4 `evaluate.py`

需要支持：

```bash
python -m src.evaluate \
  --model-path outputs/mobilenetv2_e2e_aug/best_model.keras \
  --test-dir data/processed/test \
  --output-dir outputs/mobilenetv2_e2e_aug/eval
```

评估逻辑：

1. 加载最佳模型；
2. 对测试集预测；
3. 输出类别概率与预测标签；
4. 计算 Accuracy、macro F1、macro Precision、macro Recall、macro Jaccard；
5. 保存 `metrics.json`；
6. 保存 confusion matrix 图和 CSV。

### 16.5 `run_all_experiments.py`

实现 6 个 backbone x 4 个训练策略，共 24 个实验：

```text
backbones = [
  "vgg16", "vgg19", "mobilenetv2",
  "inceptionresnetv2", "inceptionv3", "xception"
]

strategies = [
  {"train_backbone": True,  "augmentation": True},
  {"train_backbone": True,  "augmentation": False},
  {"train_backbone": False, "augmentation": True},
  {"train_backbone": False, "augmentation": False},
]
```

每个实验保存：

```text
outputs/{backbone}_{strategy}/
  best_model.keras
  history.csv
  metrics.json
  confusion_matrix.csv
  confusion_matrix.png
  training_curves.png
```

### 16.6 `gradcam.py`

对最佳 MobileNetV2 模型实现 Grad-CAM：

功能：

- 输入单张图像或测试集图像目录；
- 计算 `crack` 类 heatmap；
- 使用最后一个卷积层；
- 将 heatmap 叠加到原图；
- 保存可视化结果。

命令示例：

```bash
python -m src.gradcam \
  --model-path outputs/mobilenetv2_e2e_aug/best_model.keras \
  --image-dir data/processed/test/crack \
  --class-index 1 \
  --output-dir outputs/mobilenetv2_e2e_aug/gradcam
```

MobileNetV2 最后卷积层可通过模型结构自动查找最后一个 4D 输出层，而不是硬编码层名。

---

## 17. 复现实验的最小版本

如果只想实现论文最推荐的模型，而不是 24 组完整实验，Claude 可优先实现：

```text
Backbone: MobileNetV2
Weights: ImageNet
Input: 227 x 227 x 3
Head: GlobalAveragePooling2D -> Dense(2, softmax)
Training: end-to-end
Augmentation: yes
Loss: categorical_crossentropy
Optimizer: Adam(lr=1e-5)
Batch size: 32
Early stopping: val_accuracy, patience=20
Best model: highest val_accuracy
Evaluation: test accuracy, macro F1, macro precision, macro recall, macro Jaccard, confusion matrix
Expected paper result: 100% accuracy and 100% F1 on 100-image test set
```

---

## 18. 代码生成时的关键注意事项

1. **不要把验证集/测试集用于数据增强。**  
   论文测试结果基于原始未增强图像。

2. **不要在测试集上调参。**  
   模型选择只看 validation accuracy。

3. **使用 `Dense(2, softmax)` 而不是单输出 sigmoid。**  
   这是论文设定。

4. **使用 `categorical_crossentropy`。**

5. **复现时优先使用 `rescale=1./255`，不要默认套用 backbone-specific preprocess。**

6. **VGG 的头部与其他模型不同。**  
   VGG16/VGG19 使用 Flatten + 两个 4096 维全连接层。

7. **评价指标用 macro average。**  
   论文报告的是两个类别的平均分。

8. **类别顺序要固定。**  
   推荐 `non-crack=0`，`crack=1`，confusion matrix 使用 `labels=[0,1]`。

9. **记录随机种子。**  
   论文没有给出 seed。代码应提供 `--seed` 参数，例如 `42`。

10. **官方数据集可能只有裁剪后图像。**  
    如果没有原始 bounding box 标注文件，不需要重新标注，只需按论文的 split 复现训练/评估。

---

## 19. 论文没有完全说明、实现时需做合理假设的部分

以下细节论文没有明确给出，代码中需要显式记录假设：

| 未明确项 | 推荐假设/实现 |
|---|---|
| 随机种子 | 提供 `--seed` 参数，默认 42 |
| 最大训练 epoch | 设置 100 或 200，依赖 early stopping |
| 旋转角度是否为 `[0,45]` 或 `[-45,45]` | 若用 Keras `rotation_range=45`，实际为 `[-45,45]`；若严格按文字可自定义 `[0,45]` |
| 是否使用 ImageNet `preprocess_input` | 不使用，论文只说明 `1/255` rescale |
| split 是否按原图分组 | 论文只写 stratified sampling；如有原图 ID，建议做 group-aware split 以避免泄漏 |
| 类别文件夹名 | 使用 `crack` 和 `non-crack` |
| 模型保存格式 | 推荐 `.keras`，也可兼容 `.h5` |

---

## 20. 给 Claude 的直接开发指令

下面这段可以直接复制给 Claude，让它生成代码：

```text
请实现一个 TensorFlow/Keras 项目，用于复现论文 “Deep learning for crack detection on masonry façades using limited data and transfer learning”。任务是对 227x227 砖砌体图像块进行二分类：crack vs non-crack。

数据目录结构为：
data/processed/train/crack, data/processed/train/non-crack,
data/processed/val/crack, data/processed/val/non-crack,
data/processed/test/crack, data/processed/test/non-crack。

请实现以下模块：
1. src/dataset.py：构建 ImageDataGenerator。训练集支持 augmentation；验证/测试集只 rescale=1/255。target_size=(227,227)，batch_size=32，class_mode='categorical'。
2. src/models.py：支持 vgg16, vgg19, mobilenetv2, inceptionresnetv2, inceptionv3, xception。全部使用 ImageNet weights, include_top=False。VGG 使用 Flatten -> Dense(4096,relu) -> Dense(4096,relu) -> Dense(2,softmax)。其他模型使用 GlobalAveragePooling2D -> Dense(2,softmax)。可通过 train_backbone 控制 base_model.trainable。
3. src/train.py：命令行训练。Adam(lr=1e-5)，categorical_crossentropy，metrics=['accuracy']。使用 ModelCheckpoint 保存 val_accuracy 最高模型，EarlyStopping(monitor='val_accuracy', patience=20, mode='max', restore_best_weights=True)。保存 history.csv 和训练曲线。
4. src/evaluate.py：加载最佳模型，在 test set 上计算 accuracy, macro F1, macro precision, macro recall, macro Jaccard，保存 metrics.json、confusion_matrix.csv、confusion_matrix.png。类别顺序固定为 non-crack=0, crack=1。
5. src/gradcam.py：对最佳 MobileNetV2 模型生成 crack 类 Grad-CAM heatmap，并叠加到原图保存。
6. scripts/run_all_experiments.py：运行 6 个 backbone x 4 种策略：end-to-end+aug, end-to-end+no_aug, frozen+aug, frozen+no_aug。

训练配置：input_shape=(227,227,3)，batch_size=32，lr=1e-5，max_epochs=200，early stopping patience=20。数据增强仅用于训练集，包括 horizontal_flip=True, vertical_flip=True, brightness_range=(0.3,1.0), rotation_range=45, rescale=1/255。验证和测试只使用 rescale=1/255。

请生成完整可运行代码、requirements.txt、README.md、命令行示例，并保证每个实验的输出保存在 outputs/{backbone}_{strategy}/ 下，包括 best_model.keras、history.csv、metrics.json、confusion_matrix.csv、confusion_matrix.png 和 training_curves.png。
```

---

## 21. 最推荐的复现命令

完整跑论文推荐模型：

```bash
python -m src.train \
  --data-dir data/processed \
  --backbone mobilenetv2 \
  --train-backbone true \
  --augmentation true \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 200 \
  --patience 20 \
  --output-dir outputs/mobilenetv2_e2e_aug

python -m src.evaluate \
  --model-path outputs/mobilenetv2_e2e_aug/best_model.keras \
  --test-dir data/processed/test \
  --output-dir outputs/mobilenetv2_e2e_aug/eval

python -m src.gradcam \
  --model-path outputs/mobilenetv2_e2e_aug/best_model.keras \
  --image-dir data/processed/test/crack \
  --class-index 1 \
  --output-dir outputs/mobilenetv2_e2e_aug/gradcam
```

---

## 22. 预期复现结果

如果使用相同数据划分并完全复现论文设置，推荐模型应接近：

```text
Model: MobileNetV2
Training: end-to-end
Augmentation: yes
Test set size: 100
Accuracy: 1.0000
F1-score: 1.0000
Precision: 1.0000
Recall: 1.0000
Jaccard: 1.0000
Confusion matrix: [[50, 0], [0, 50]]
```

如果结果低于论文，优先检查：

1. 数据集 split 是否一致；
2. 类别标签顺序是否一致；
3. 是否错误地对验证/测试集做了随机增强；
4. 是否使用了 `preprocess_input` 而不是 `1/255`；
5. 是否冻结了 backbone；
6. 是否使用 sigmoid/binary_crossentropy 而不是 softmax/categorical_crossentropy；
7. 是否训练轮数不足或 early stopping 监控方向错误。

---

## 23. 一句话总结

这篇论文的可复现核心是：将 700 张平衡的 `227x227` 砖砌体 crop 图像分成 `500/100/100` 的 train/val/test，训练 ImageNet 预训练 CNN 做 `crack` vs `non-crack` 二分类；训练集可使用随机翻转、亮度变化和旋转增强；每个模型用 validation accuracy 选择最佳 epoch，并在未见测试集上报告 macro F1 等指标；最终推荐使用 MobileNetV2 端到端微调加数据增强。
