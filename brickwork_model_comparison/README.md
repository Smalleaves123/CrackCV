# Brickwork Model Comparison

This module follows the brickwork crack detection paper specification:

- task: binary image classification
- classes: `non-crack` and `crack`
- input size: `227 x 227`
- dataset size: balanced `Positive` / `Negative` image patches
- target experiment: compare 6 paper CNN backbones and our self-built `custom_cnn`

## Models

The comparison script supports:

| Paper model | Implementation |
| --- | --- |
| VGG16 | torchvision |
| VGG19 | torchvision |
| MobileNetV2 | torchvision |
| InceptionV3 | torchvision |
| InceptionResNetV2 | timm |
| Xception | timm |
| Our custom CNN | PyTorch module in `train_compare_models.py` |

`timm` is required only for InceptionResNetV2 and Xception.

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r brickwork_model_comparison/requirements-cpu.txt
```

## Reproduce Current Submitted Results

The committed experiment compares our self-built `custom_cnn` with `mobilenet_v2`.
Both run for 20 epochs without downloading pretrained weights:

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

## Paper-Style Training

The paper uses ImageNet pretrained weights, learning rate `1e-5`, batch size `32`, and compares four strategies:

```text
e2e_aug
e2e_no_aug
frozen_aug
frozen_no_aug
```

Example:

```bash
python3 brickwork_model_comparison/src/train_compare_models.py \
  --dataset dataset \
  --models mobilenet_v2,vgg16,vgg19,inception_v3,inception_resnet_v2,xception \
  --pretrained \
  --epochs 100 \
  --batch-size 32 \
  --image-size 227 \
  --lr 1e-5 \
  --strategy e2e_aug \
  --patience 20 \
  --output-dir outputs/brickwork_e2e_aug
```

## Outputs

For each model, the script saves:

- `best_model.pth`
- `training_history.csv`
- `test_metrics.csv`
- `confusion_matrix.png`

It also saves:

- `summary.csv`

## Submitted Results

The current submitted artifacts are stored in `brickwork_model_comparison/results/`.

| Model | Accuracy | Precision | Recall | F1-score | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| custom_cnn | 0.9400 | 0.9464 | 0.9400 | 0.9398 | 0.8864 |
| mobilenet_v2 | 0.9400 | 0.9428 | 0.9400 | 0.9399 | 0.8866 |

## Current Practical Note

The current local CPU environment can run smoke tests and small experiments. Full six-model pretrained comparison is heavier and may require:

- network access to download pretrained weights
- more disk space
- GPU or long CPU training time
