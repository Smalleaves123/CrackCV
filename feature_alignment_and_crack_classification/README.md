# Feature Alignment and Crack Classification

This folder contains two main functions:

- `feature_alignment`: ORB/SIFT feature matching and affine validation.
- `crack_classification`: crack/no-crack model training and prediction.

The feature alignment code outputs matched control points for affine or TPS-style warping. The crack classification code trains a CNN model using the existing `dataset/Positive` and `dataset/Negative` folders.

## Folder Structure

```text
feature_alignment_and_crack_classification/
  README.md
  requirements.txt
  requirements-cpu.txt

  src/
    feature_alignment/
      feature_matching.py
      affine_baseline.py
      synthetic_feature_test.py

    crack_classification/
      train_classifier.py
      predict_classifier.py
```

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r feature_alignment_and_crack_classification/requirements-cpu.txt
```

## Feature Alignment

Run feature matching:

```bash
python3 feature_alignment_and_crack_classification/src/feature_alignment/feature_matching.py \
  --reference dataset/Positive/00214.jpg \
  --input dataset/Positive/00340.jpg \
  --method sift \
  --output outputs/real_sift_matches.jpg \
  --points-out outputs/real_sift_points.npz
```

The output point file contains:

- `input_points`: matched points from the input image.
- `reference_points`: matched points from the reference image.

Run affine alignment with matched points:

```bash
python3 feature_alignment_and_crack_classification/src/feature_alignment/affine_baseline.py \
  --reference dataset/Positive/00214.jpg \
  --input dataset/Positive/00340.jpg \
  --points outputs/real_sift_points.npz \
  --output outputs/affine_result.jpg \
  --comparison-output outputs/affine_comparison.jpg
```

Run the controlled synthetic alignment test:

```bash
python3 feature_alignment_and_crack_classification/src/feature_alignment/synthetic_feature_test.py \
  --image dataset/Positive/00214.jpg \
  --method sift \
  --output-dir outputs/synthetic_sift
```

Synthetic affine matching results:

| Method | CLAHE | Good Matches | RANSAC Inliers | Inlier Ratio | Affine Error |
| --- | --- | ---: | ---: | ---: | ---: |
| ORB | no | 516 | 511 | 0.9903 | 0.2504 |
| ORB | yes | 471 | 457 | 0.9703 | 0.1393 |
| SIFT | no | 240 | 237 | 0.9875 | 0.0465 |
| SIFT | yes | 390 | 389 | 0.9974 | 0.0983 |

Directly matching two unrelated crack images is difficult because they are not paired views of the same wall. On the tested real-image pair, SIFT improved the result to 10 matches, but affine estimation was still unreliable. The synthetic test shows the method works when the reference and input images are truly related.

## Crack Classification

Run a quick CPU smoke test:

```bash
python3 feature_alignment_and_crack_classification/src/crack_classification/train_classifier.py \
  --dataset dataset \
  --epochs 1 \
  --batch-size 32 \
  --model small_cnn \
  --output-dir outputs/classifier_smoke
```

Run the selected baseline experiment:

```bash
python3 feature_alignment_and_crack_classification/src/crack_classification/train_classifier.py \
  --dataset dataset \
  --epochs 10 \
  --batch-size 64 \
  --model small_cnn \
  --lr 0.001 \
  --image-size 128 \
  --output-dir outputs/classifier_smallcnn_10ep
```

The training script saves:

- `best_model.pth`
- `training_history.csv`
- `loss_curve.png`
- `accuracy_curve.png`
- `test_metrics.csv`
- `confusion_matrix.png`

Run inference with a saved checkpoint:

```bash
python3 feature_alignment_and_crack_classification/src/crack_classification/predict_classifier.py \
  --checkpoint outputs/classifier_smallcnn_10ep/best_model.pth \
  --image dataset/Positive/00214.jpg
```

## Classification Experiment Summary

The dataset contains 350 positive crack images and 350 negative images. The split is stratified into 492 training images, 104 validation images, and 104 test images.

| Model | LR | Image Size | Epochs | Best Val Acc | Test Acc | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_cnn | 0.001 | 128 | 10 | 0.9038 | 0.9038 | 0.9565 | 0.8462 | 0.8980 |
| small_cnn | 0.0003 | 128 | 10 | 0.9038 | 0.9038 | 0.9773 | 0.8269 | 0.8958 |
| small_cnn | 0.001 | 96 | 10 | 0.9327 | 0.8942 | 0.9556 | 0.8269 | 0.8866 |

The `128 x 128` setting with learning rate `0.001` is selected as the default baseline because it gives the best F1 among the tested settings while keeping CPU training lightweight.

## Course Requirement Mapping

- Requirement 1: preprocessing and augmentation are implemented in `build_transforms`.
- Requirement 2: `small_cnn`, MobileNetV2, or ResNet18 can be trained for crack classification.
- Requirement 3: model structure, learning rate, batch size, image size, and epochs are configurable.
- Requirement 5: loss and accuracy curves are saved after training.
- Requirement 6: `predict_classifier.py` loads a saved checkpoint and outputs class probabilities for a given image.
