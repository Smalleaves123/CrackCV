import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


CLASS_TO_INDEX = {"Negative": 0, "Positive": 1}
INDEX_TO_CLASS = {0: "non-crack", 1: "crack"}


class CustomCrackCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class BrickworkDataset(Dataset):
    def __init__(self, dataset_dir, image_size, augment=False):
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.augment = augment
        self.samples = []

        for class_name, class_index in CLASS_TO_INDEX.items():
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((image_path, class_index))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.dataset_dir}")

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=(0.3, 1.0)),
                transforms.ToTensor(),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        transform = self.train_transform if self.augment else self.eval_transform
        return transform(image), torch.tensor(label, dtype=torch.long)


def stratified_split(samples, seed):
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for index, (_, label) in enumerate(samples):
        by_class[label].append(index)

    train_indices = []
    val_indices = []
    test_indices = []
    for indices in by_class.values():
        rng.shuffle(indices)
        train_indices.extend(indices[:250])
        val_indices.extend(indices[250:300])
        test_indices.extend(indices[300:350])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def import_timm():
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required for inception_resnet_v2 and xception. Install requirements-cpu.txt.") from exc
    return timm


def build_model(model_name, pretrained=False):
    if model_name == "custom_cnn":
        if pretrained:
            raise ValueError("custom_cnn is self-built and does not support ImageNet pretrained weights.")
        return CustomCrackCNN(num_classes=2)

    weights = None
    if model_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        return model

    if model_name == "vgg19":
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg19(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        return model

    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model

    if model_name == "inception_v3":
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
        return model

    if model_name in {"inception_resnet_v2", "xception"}:
        timm = import_timm()
        timm_name = "inception_resnet_v2" if model_name == "inception_resnet_v2" else "xception"
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=2)

    raise ValueError(f"Unsupported model: {model_name}")


def freeze_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.Linear):
            for parameter in module.parameters():
                parameter.requires_grad = True


def logits_from_output(output):
    if hasattr(output, "logits"):
        return output.logits
    return output


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            output = model(images)
            logits = logits_from_output(output)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(y_true)
    return metrics


def classification_metrics(y_true, y_pred):
    confusion = np.zeros((2, 2), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred):
        confusion[true_label, predicted_label] += 1

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_jaccard = []

    for class_index in [0, 1]:
        tp = confusion[class_index, class_index]
        fp = confusion[:, class_index].sum() - tp
        fn = confusion[class_index, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        jaccard = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
        per_class_jaccard.append(jaccard)

    accuracy = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    return {
        "accuracy": accuracy,
        "precision": float(np.mean(per_class_precision)),
        "recall": float(np.mean(per_class_recall)),
        "f1": float(np.mean(per_class_f1)),
        "jaccard": float(np.mean(per_class_jaccard)),
        "confusion": confusion,
    }


def save_history(history, output_dir):
    with (output_dir / "training_history.csv").open("w", newline="") as csv_file:
        fieldnames = [key for key in history[0].keys() if key != "confusion"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()


def save_test_outputs(metrics, output_dir):
    with (output_dir / "test_metrics.csv").open("w", newline="") as csv_file:
        fieldnames = ["accuracy", "precision", "recall", "f1", "jaccard", "loss"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics[key] for key in fieldnames})

    confusion = metrics["confusion"]
    plt.figure(figsize=(4, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.xticks([0, 1], [INDEX_TO_CLASS[0], INDEX_TO_CLASS[1]])
    plt.yticks([0, 1], [INDEX_TO_CLASS[0], INDEX_TO_CLASS[1]])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(2):
        for col in range(2):
            plt.text(col, row, int(confusion[row, col]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def parse_models(raw):
    if raw == "all":
        return ["vgg16", "vgg19", "mobilenet_v2", "inception_v3", "inception_resnet_v2", "xception"]
    if raw == "all_with_custom":
        return ["custom_cnn", "vgg16", "vgg19", "mobilenet_v2", "inception_v3", "inception_resnet_v2", "xception"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Compare CNN backbones for brickwork crack classification.")
    parser.add_argument("--dataset", default=Path("dataset"), type=Path)
    parser.add_argument("--models", default="mobilenet_v2", help="Comma-separated model names or all.")
    parser.add_argument("--output-dir", default=Path("outputs/brickwork_model_comparison"), type=Path)
    parser.add_argument("--strategy", default="e2e_aug", choices=["e2e_aug", "e2e_no_aug", "frozen_aug", "frozen_no_aug"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--image-size", default=227, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augment = args.strategy.endswith("_aug")
    freeze = args.strategy.startswith("frozen")

    base_dataset = BrickworkDataset(args.dataset, args.image_size, augment=False)
    train_indices, val_indices, test_indices = stratified_split(base_dataset.samples, args.seed)
    train_dataset = BrickworkDataset(args.dataset, args.image_size, augment=augment)
    eval_dataset = BrickworkDataset(args.dataset, args.image_size, augment=False)

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(eval_dataset, val_indices), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(eval_dataset, test_indices), batch_size=args.batch_size, shuffle=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    print(f"device: {device}")
    print(f"train/val/test: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")

    for model_name in parse_models(args.models):
        print(f"training model: {model_name}")
        model_output_dir = args.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model = build_model(model_name, pretrained=args.pretrained).to(device)
        if freeze:
            freeze_backbone(model)

        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_accuracy = -1.0
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = run_epoch(model, val_loader, criterion, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                }
            )

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": model_name,
                        "strategy": args.strategy,
                        "model_state_dict": model.state_dict(),
                        "image_size": args.image_size,
                        "class_to_index": CLASS_TO_INDEX,
                    },
                    model_output_dir / "best_model.pth",
                )
            else:
                epochs_without_improvement += 1

            print(
                f"{model_name} epoch [{epoch}/{args.epochs}] "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )

            if epochs_without_improvement >= args.patience:
                print(f"early stopping: {model_name}")
                break

        checkpoint = torch.load(model_output_dir / "best_model.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = run_epoch(model, test_loader, criterion, device)
        save_history(history, model_output_dir)
        save_test_outputs(test_metrics, model_output_dir)
        summary.append(
            {
                "model": model_name,
                "strategy": args.strategy,
                "pretrained": args.pretrained,
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "jaccard": test_metrics["jaccard"],
            }
        )

    with (args.output_dir / "summary.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"saved summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
