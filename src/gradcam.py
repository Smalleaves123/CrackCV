from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from .models import build_model, find_last_4d_layer
from .utils import ensure_dir, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-index", type=int, default=1)
    parser.add_argument("--img-size", type=int, nargs=2, default=(227, 227))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    output_dir = ensure_dir(args.output_dir)

    checkpoint = torch.load(args.model_path, map_location=device)
    model = build_model(
        backbone_name=checkpoint["backbone_name"],
        train_backbone=checkpoint.get("train_backbone", True),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    layer_name, target_layer = find_last_4d_layer(model)
    print("Using Grad-CAM layer: {0}".format(layer_name))

    image_dir = Path(args.image_dir)
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        overlay = generate_gradcam(
            model=model,
            image_path=image_path,
            target_layer=target_layer,
            class_index=args.class_index,
            img_size=tuple(args.img_size),
            device=device,
        )
        cv2.imwrite(str(output_dir / image_path.name), overlay)


def generate_gradcam(
    model,
    image_path: Path,
    target_layer,
    class_index: int,
    img_size: Tuple[int, int],
    device,
):
    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError("Failed to read image: {0}".format(image_path))
    resized = cv2.resize(original, img_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(device)

    model.zero_grad()
    logits = model(tensor)
    score = logits[:, class_index].sum()
    score.backward()

    activation_map = activations[0][0]
    gradient_map = gradients[0][0]
    weights = torch.mean(gradient_map, dim=(1, 2))
    heatmap = torch.zeros(activation_map.shape[1:], device=device)
    for channel_index, weight in enumerate(weights):
        heatmap += weight * activation_map[channel_index]
    heatmap = torch.relu(heatmap)
    heatmap = heatmap.cpu().numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, colored, 0.4, 0)

    forward_handle.remove()
    backward_handle.remove()
    return overlay


if __name__ == "__main__":
    main()
