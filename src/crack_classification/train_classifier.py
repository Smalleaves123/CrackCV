import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


CLASS_TO_INDEX = {"Negative": 0, "Positive": 1}


class SmallCrackCNN(nn.Module):
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


class CrackDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def split_indices(dataset, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    indices_by_class = {class_index: [] for class_index in CLASS_TO_INDEX.values()}
    for index, (_, label) in enumerate(dataset.samples):
        indices_by_class[label].append(index)

    train_indices = []
    val_indices = []
    test_indices = []

    for indices in indices_by_class.values():
        rng.shuffle(indices)
        test_count = int(len(indices) * test_ratio)
        val_count = int(len(indices) * val_ratio)

        test_indices.extend(indices[:test_count])
        val_indices.extend(indices[test_count : test_count + val_count])
        train_indices.extend(indices[test_count + val_count :])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def build_model(model_name, pretrained):
    if model_name == "small_cnn":
        if pretrained:
            raise ValueError("small_cnn does not support pretrained weights.")
        return SmallCrackCNN(num_classes=2)

    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def freeze_backbone(model, model_name):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_name == "mobilenet_v2":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if model_name == "resnet18":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unsupported model: {model_name}")


def run_epoch(model, dataloader, criterion, device, optimizer=None, return_confusion=False):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(2, 2, dtype=torch.long)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        if return_confusion:
            for true_label, predicted_label in zip(labels.cpu(), predictions.cpu()):
                confusion[int(true_label), int(predicted_label)] += 1

    loss = total_loss / total
    accuracy = correct / total
    if return_confusion:
        return loss, accuracy, confusion
    return loss, accuracy


def classification_metrics(confusion):
    tn = int(confusion[0, 0])
    fp = int(confusion[0, 1])
    fn = int(confusion[1, 0])
    tp = int(confusion[1, 1])

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_history(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_history.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

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

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_acc"] for row in history], label="train acc")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()


def save_test_metrics(metrics, confusion, output_dir):
    metrics_path = output_dir / "test_metrics.csv"
    with metrics_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    plt.figure(figsize=(4, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for row in range(2):
        for col in range(2):
            plt.text(col, row, int(confusion[row, col]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a crack/no-crack image classifier.")
    parser.add_argument("--dataset", default=Path("dataset"), type=Path, help="Dataset root with Positive and Negative folders.")
    parser.add_argument("--output-dir", default=Path("outputs/classifier"), type=Path, help="Directory for model and curves.")
    parser.add_argument("--model", default="small_cnn", choices=["small_cnn", "mobilenet_v2", "resnet18"], help="Backbone model.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size.")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--image-size", default=224, type=int, help="Input image size.")
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader worker processes.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classification head.")
    parser.add_argument("--val-ratio", default=0.15, type=float, help="Validation split ratio.")
    parser.add_argument("--test-ratio", default=0.15, type=float, help="Test split ratio.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform = build_transforms(args.image_size)

    base_dataset = CrackDataset(args.dataset)
    train_indices, val_indices, test_indices = split_indices(base_dataset, args.val_ratio, args.test_ratio, args.seed)

    train_dataset = CrackDataset(args.dataset, transform=train_transform)
    eval_dataset = CrackDataset(args.dataset, transform=eval_transform)

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        Subset(eval_dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"device: {device}")
    print(f"classes: {CLASS_TO_INDEX}")
    print(f"train/val/test: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")

    model = build_model(args.model, args.pretrained).to(device)
    if args.freeze_backbone:
        freeze_backbone(model, args.model)

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)

    best_val_acc = 0.0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": args.model,
                    "class_to_index": CLASS_TO_INDEX,
                    "image_size": args.image_size,
                },
                best_model_path,
            )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, confusion = run_epoch(model, test_loader, criterion, device, return_confusion=True)
    metrics = classification_metrics(confusion)
    metrics.update({"test_loss": test_loss, "test_acc": test_acc})
    save_history(history, args.output_dir)
    save_test_metrics(metrics, confusion.numpy(), args.output_dir)

    print(f"best val accuracy: {best_val_acc:.4f}")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    print(
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
    print(f"saved model: {best_model_path}")
    print(f"saved curves: {args.output_dir / 'loss_curve.png'}, {args.output_dir / 'accuracy_curve.png'}")
    print(f"saved test metrics: {args.output_dir / 'test_metrics.csv'}, {args.output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
