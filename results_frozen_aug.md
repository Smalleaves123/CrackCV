# 实验结果：冻结卷积基 + 数据增强

> 训练策略：Frozen backbone + Data Augmentation
> 训练参数：batch_size=32, lr=1e-5, max_epochs=200, patience=20

## Table 4
Classification results when the weights of the convolutional base are kept frozen and data augmentation is used.

| Base model | Accuracy | F1-score | Precision | Recall | Jaccard |
|:---|---:|---:|---:|---:|---:|
| VGG16 | **0.8600** | 0.8598 | 0.8623 | 0.8600 | 0.7541 |
| VGG19 | **0.9300** | 0.9299 | 0.9316 | 0.9300 | 0.8691 |
| MobileNetV2 | **0.8300** | 0.8292 | 0.8366 | 0.8300 | 0.7084 |
| InceptionResNetV2 | **0.8000** | 0.8000 | 0.8000 | 0.8000 | 0.6667 |
| InceptionV3 | **0.6700** | 0.6684 | 0.6734 | 0.6700 | 0.5024 |
| Xception | **0.9100** | 0.9098 | 0.9141 | 0.9100 | 0.8345 |

---

## 与论文结果对比

| Base model | 本实验 Accuracy | 论文 Accuracy | 差异 |
|:---|---:|---:|---:|
| VGG16 | 0.8600 | 0.9200 | -0.0600 |
| VGG19 | 0.9300 | 0.8500 | +0.0800 |
| MobileNetV2 | 0.8300 | 0.6800 | +0.1500 |
| InceptionResNetV2 | 0.8000 | 0.8300 | -0.0300 |
| InceptionV3 | 0.6700 | 0.7600 | -0.0900 |
| Xception | 0.9100 | 0.7400 | +0.1700 |
