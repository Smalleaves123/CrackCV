import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train_classifier import build_model


def build_eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(probabilities.argmax().item())

    return predicted_index, probabilities.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Run crack/no-crack classification on one image.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to best_model.pth.")
    parser.add_argument("--image", required=True, type=Path, help="Image to classify.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_name = checkpoint["model"]
    image_size = checkpoint["image_size"]
    class_to_index = checkpoint["class_to_index"]
    index_to_class = {index: class_name for class_name, index in class_to_index.items()}

    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    transform = build_eval_transform(image_size)
    predicted_index, probabilities = predict_image(model, args.image, transform, device)

    print(f"image: {args.image}")
    print(f"prediction: {index_to_class[predicted_index]}")
    for index, probability in enumerate(probabilities):
        print(f"{index_to_class[index]}: {probability:.4f}")


if __name__ == "__main__":
    main()
