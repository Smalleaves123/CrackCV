# Brickwork Crack Detection Experiment Report

## Goal

This experiment tests a self-built CNN against a lightweight CNN backbone on the brickwork crack classification dataset. The saved metrics match the indicators requested by the reference paper:

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

## Models

### `custom_cnn`

`custom_cnn` is our self-built lightweight CNN. It is trained from scratch, without ImageNet pretrained weights.

```text
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-AdaptiveAvgPool
Dropout
Linear(128 -> 2)
```

### `mobilenet_v2`

`mobilenet_v2` is used as the comparison backbone. In this CPU experiment, it is also trained from scratch so that the comparison does not depend on downloading pretrained weights.

## Training Setting

```text
strategy: e2e_aug
pretrained: false
epochs: 20
batch size: 32
image size: 128 x 128
learning rate: 0.001
optimizer: Adam
loss: CrossEntropyLoss
augmentation: horizontal flip, vertical flip, rotation, brightness jitter
```

Commands:

```bash
python3 brickwork_model_comparison/src/train_compare_models.py \
  --dataset dataset \
  --models custom_cnn \
  --epochs 20 \
  --batch-size 32 \
  --image-size 128 \
  --strategy e2e_aug \
  --lr 0.001 \
  --output-dir outputs/brickwork_two_models_20ep

python3 brickwork_model_comparison/src/train_compare_models.py \
  --dataset dataset \
  --models mobilenet_v2 \
  --epochs 20 \
  --batch-size 32 \
  --image-size 128 \
  --strategy e2e_aug \
  --lr 0.001 \
  --output-dir outputs/brickwork_two_models_20ep
```

## Final Test Metrics

| Model | Accuracy | Precision | Recall | F1-score | Jaccard | Test Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| custom_cnn | 0.9400 | 0.9464 | 0.9400 | 0.9398 | 0.8864 | 0.1728 |
| mobilenet_v2 | 0.9400 | 0.9428 | 0.9400 | 0.9399 | 0.8866 | 0.1401 |

## Training Curve Summary

| Model | Final Train Acc | Best Val Acc | Final Val Acc | Final Val F1 |
| --- | ---: | ---: | ---: | ---: |
| custom_cnn | 0.9680 | 0.9700 | 0.9700 | 0.9700 |
| mobilenet_v2 | 0.9420 | 0.9600 | 0.9300 | 0.9300 |

## Saved Artifacts

The final artifacts are committed under `brickwork_model_comparison/results/`:

| Path | Content |
| --- | --- |
| `results/summary.csv` | Combined metrics for both models |
| `results/custom_cnn/training_history.csv` | 20-epoch training log |
| `results/custom_cnn/test_metrics.csv` | Final test metrics |
| `results/custom_cnn/loss_curve.png` | Loss curve |
| `results/custom_cnn/confusion_matrix.png` | Confusion matrix |
| `results/custom_cnn/best_model.pth` | Best checkpoint |
| `results/mobilenet_v2/training_history.csv` | 20-epoch training log |
| `results/mobilenet_v2/test_metrics.csv` | Final test metrics |
| `results/mobilenet_v2/loss_curve.png` | Loss curve |
| `results/mobilenet_v2/confusion_matrix.png` | Confusion matrix |
| `results/mobilenet_v2/best_model.pth` | Best checkpoint |

## Interpretation

Both models reach `0.94` test accuracy and about `0.94` F1-score. This is enough for the course requirement that asks for a trainable model, training curves, optimization records, and paper-style evaluation metrics.

The custom CNN result is especially useful for the "self-built model testing" responsibility: it proves that our own architecture can learn crack features from the dataset instead of only relying on an existing model. MobileNetV2 is kept as the comparison model because it is a known lightweight CNN backbone and provides a stronger reference point.

Full six-model pretrained comparison is still possible, but it is much heavier on a CPU machine and requires stable pretrained weight downloads. For this submission, the committed result focuses on the two completed and reproducible 20-epoch experiments.
