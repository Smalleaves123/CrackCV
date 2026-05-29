# Brickwork Crack Detection Experiment Report

## Goal

Test our self-built CNN on the brickwork crack classification task and keep the same evaluation metrics used in the reference paper:

- Accuracy
- Precision
- Recall
- F1-score
- Jaccard index
- Confusion matrix

## Dataset

The repository dataset contains 700 balanced image patches:

| Class | Folder | Count |
| --- | --- | ---: |
| non-crack | `dataset/Negative` | 350 |
| crack | `dataset/Positive` | 350 |

The experiment uses a stratified split:

| Split | Total | non-crack | crack |
| --- | ---: | ---: | ---: |
| Train | 500 | 250 | 250 |
| Validation | 100 | 50 | 50 |
| Test | 100 | 50 | 50 |

## Model

`custom_cnn` is our self-built lightweight CNN:

```text
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-AdaptiveAvgPool
Dropout
Linear(128 -> 2)
```

It is trained from scratch, without ImageNet pretrained weights.

## Training Setting

```text
model: custom_cnn
strategy: e2e_aug
pretrained: false
epochs: 10
batch size: 64
image size: 128 x 128
learning rate: 0.001
optimizer: Adam
loss: CrossEntropyLoss
augmentation: horizontal flip, vertical flip, rotation, brightness jitter
```

Command:

```bash
python3 brickwork_model_comparison/src/train_compare_models.py \
  --dataset dataset \
  --models custom_cnn \
  --epochs 10 \
  --batch-size 64 \
  --image-size 128 \
  --strategy e2e_aug \
  --lr 0.001 \
  --output-dir outputs/brickwork_custom_cnn_10ep
```

## Training Curve Summary

| Epoch | Train Acc | Val Acc | Val F1 |
| ---: | ---: | ---: | ---: |
| 1 | 0.6880 | 0.5000 | 0.3333 |
| 2 | 0.7920 | 0.5000 | 0.3333 |
| 3 | 0.8220 | 0.5000 | 0.3333 |
| 4 | 0.8360 | 0.5000 | 0.3333 |
| 5 | 0.8560 | 0.5000 | 0.3333 |
| 6 | 0.8840 | 0.5500 | 0.4357 |
| 7 | 0.8840 | 0.6500 | 0.6011 |
| 8 | 0.8860 | 0.8000 | 0.7917 |
| 9 | 0.9060 | 0.7900 | 0.7803 |
| 10 | 0.9400 | 0.8500 | 0.8465 |

## Test Metrics

| Model | Accuracy | Precision | Recall | F1-score | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| custom_cnn | 0.8000 | 0.8571 | 0.8000 | 0.7917 | 0.6571 |

The confusion matrix image is saved locally at:

```text
outputs/brickwork_custom_cnn_10ep/custom_cnn/confusion_matrix.png
```

## Interpretation

The self-built CNN learns meaningful crack features within 10 epochs. Validation accuracy rises from `0.50` to `0.85`, and the test accuracy reaches `0.80`.

This result is lower than the best paper setting because the paper relies on ImageNet pretrained backbones and longer early-stopping training. The current experiment is still useful as a custom lightweight baseline that can later be compared against VGG16, VGG19, MobileNetV2, InceptionV3, InceptionResNetV2, and Xception.
