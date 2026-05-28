# CrackCV

CrackCV is a computer vision project for brick-wall crack analysis. The current code provides two working baselines:

- Crack classification: train a small CNN to predict `Positive` or `Negative`.
- Feature alignment: extract matching points between two images for affine/TPS-style geometric alignment.

## Project Structure

```text
dataset/
  Positive/                  # crack images
  Negative/                  # non-crack images

src/
  crack_classification/       # deep learning training and prediction
    train_classifier.py
    predict_classifier.py

  feature_alignment/          # feature matching and affine baseline
    feature_matching.py
    affine_baseline.py
    synthetic_feature_test.py

docs/
  member_b_feature_and_model.md
```

## Setup

CPU-only setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
```

General setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Crack Classifier

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

The script saves:

```text
best_model.pth
training_history.csv
loss_curve.png
accuracy_curve.png
test_metrics.csv
confusion_matrix.png
```

## Predict One Image

```bash
python3 src/crack_classification/predict_classifier.py \
  --checkpoint outputs/classifier_smallcnn_10ep/best_model.pth \
  --image dataset/Positive/00214.jpg
```

## Run Feature Matching

```bash
python3 src/feature_alignment/feature_matching.py \
  --reference dataset/Positive/00214.jpg \
  --input dataset/Positive/00340.jpg \
  --method sift \
  --output outputs/real_sift_matches.jpg \
  --points-out outputs/real_sift_points.npz
```

## Run Synthetic Alignment Test

```bash
python3 src/feature_alignment/synthetic_feature_test.py \
  --image dataset/Positive/00214.jpg \
  --method sift \
  --output-dir outputs/synthetic_sift
```

More details are in `docs/member_b_feature_and_model.md`.
