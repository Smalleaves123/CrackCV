# Member B: Feature Extraction and Model Training

## Responsibilities

Member B provides two parts of the pipeline:

1. Traditional feature extraction for geometric alignment.
2. A deep learning classifier for Requirement 2.

The feature extraction module outputs matched control points for affine or TPS warping. The classifier module trains a crack/no-crack model using the existing `dataset/Positive` and `dataset/Negative` folders.

## Traditional CV Baseline

Run ORB feature matching:

```bash
python3 src/feature_alignment/feature_matching.py \
  --reference dataset/Positive/00214.jpg \
  --input dataset/Positive/00340.jpg \
  --output outputs/orb_matches.jpg \
  --points-out outputs/orb_points.npz
```

The output point file contains:

- `input_points`: matched points from the input image.
- `reference_points`: matched points from the reference image.

Run affine alignment with the matched points:

```bash
python3 src/feature_alignment/affine_baseline.py \
  --reference dataset/Positive/00214.jpg \
  --input dataset/Positive/00340.jpg \
  --points outputs/orb_points.npz \
  --output outputs/affine_result.jpg \
  --comparison-output outputs/affine_comparison.jpg
```

This validates the interface between feature extraction and geometric warping.

### Feature Matching Experiment

Directly matching two unrelated crack images is difficult because they are not paired views of the same wall. The first ORB baseline found only 3 good matches between `00214.jpg` and `00340.jpg`. After adding SIFT and CLAHE options, the best real-image result on this pair improved to 10 matches with SIFT, but affine estimation was still unreliable.

To verify the feature extraction pipeline under a controlled setting, `synthetic_feature_test.py` creates a known affine warp from one real crack image and then estimates matched points between the original and warped image.

```bash
python3 src/feature_alignment/synthetic_feature_test.py \
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

For the controlled affine test, SIFT gives the lowest affine matrix error. For unrelated real crack images, the current dataset is better suited for classification than geometric warping, because the images do not provide true reference/input correspondence.

## Deep Learning Model

Run a quick CPU smoke test:

```bash
python3 src/crack_classification/train_classifier.py \
  --dataset dataset \
  --epochs 1 \
  --batch-size 32 \
  --model small_cnn \
  --output-dir outputs/classifier_smoke
```

Run a longer training experiment:

```bash
python3 src/crack_classification/train_classifier.py \
  --dataset dataset \
  --epochs 10 \
  --batch-size 64 \
  --model small_cnn \
  --lr 0.001 \
  --image-size 128 \
  --output-dir outputs/classifier_smallcnn_10ep
```

In the current baseline run, `small_cnn` reached `0.9038` test accuracy after 10 epochs.

## Experiment Summary

The dataset contains 350 positive crack images and 350 negative images. The split is stratified into 492 training images, 104 validation images, and 104 test images.

| Model | LR | Image Size | Epochs | Best Val Acc | Test Acc | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_cnn | 0.001 | 128 | 10 | 0.9038 | 0.9038 | 0.9565 | 0.8462 | 0.8980 |
| small_cnn | 0.0003 | 128 | 10 | 0.9038 | 0.9038 | 0.9773 | 0.8269 | 0.8958 |
| small_cnn | 0.001 | 96 | 10 | 0.9327 | 0.8942 | 0.9556 | 0.8269 | 0.8866 |

The `128 x 128` setting with learning rate `0.001` is selected as the default baseline because it gives the best F1 among the tested settings while keeping CPU training lightweight.

The classifier writes:

- `best_model.pth`
- `training_history.csv`
- `loss_curve.png`
- `accuracy_curve.png`

Run inference with a saved checkpoint:

```bash
python3 src/crack_classification/predict_classifier.py \
  --checkpoint outputs/classifier_smallcnn_10ep/best_model.pth \
  --image dataset/Positive/00214.jpg
```

## Course Requirement Mapping

- Requirement 1: preprocessing and augmentation are implemented in `build_transforms`.
- Requirement 2: `small_cnn`, MobileNetV2, or ResNet18 can be trained for crack classification.
- Requirement 3: model structure, learning rate, batch size, image size, and epochs are configurable.
- Requirement 5: loss and accuracy curves are saved after training.
- Requirement 6: `predict_classifier.py` loads a saved checkpoint and outputs class probabilities for a given image.
