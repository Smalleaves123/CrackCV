from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from Reproduction.src.data.transforms import build_transforms
from Reproduction.src.models.model_factory import create_model_from_config
from Reproduction.src.utils.checkpoints import load_checkpoint
from Reproduction.src.utils.config import resolve_config_path


def _find_last_conv(module: torch.nn.Module):
    for child in reversed(list(module.modules())):
        if isinstance(child, torch.nn.Conv2d):
            return child
    raise RuntimeError("No convolution layer found for Grad-CAM")


def _build_explain_config(config: dict) -> dict:
    explain_config = {
        **config,
        "model": {**config["model"]},
        "weights": {**config.get("weights", {})},
    }
    explain_config["weights"]["pretrained"] = False
    explain_config["weights"]["offline_mode"] = True
    return explain_config


def generate_gradcam(config: dict, checkpoint_path: str, image_path: str, out_path: str) -> str:
    device = "cuda" if config["project"].get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint(resolve_config_path(config, checkpoint_path), map_location=device)
    model = create_model_from_config(_build_explain_config(config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    target_layer = _find_last_conv(model)
    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    transform = build_transforms(config["data"]["image_size"], {**config["augmentation"], "enabled": False}, train=False)
    image = Image.open(resolve_config_path(config, image_path)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(tensor)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    pred = outputs.argmax(dim=1)
    model.zero_grad()
    outputs[0, pred.item()].backward()
    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations["value"]).sum(dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(image.size)
    heatmap = np.array(cam_img) / 255.0
    original = np.array(image).astype(np.float32) / 255.0
    overlay = original.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + heatmap * 0.7, 0, 1)
    output_path = resolve_config_path(config, out_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.axis("off")
    plt.title("Original")
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.title("Heatmap")
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Overlay")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    forward_handle.remove()
    backward_handle.remove()
    return str(output_path)
