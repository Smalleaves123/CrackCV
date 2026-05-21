from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from Reproduction.src.data.transforms import build_transforms
from Reproduction.src.models.model_factory import create_model_from_config
from Reproduction.src.utils.checkpoints import load_checkpoint


def _build_predict_config(config: dict) -> dict:
    predict_config = {
        **config,
        "model": {**config["model"]},
        "weights": {**config.get("weights", {})},
    }
    predict_config["weights"]["pretrained"] = False
    predict_config["weights"]["offline_mode"] = True
    return predict_config


def predict_image(config: dict, checkpoint_path: str, image_path: str) -> dict:
    device = "cuda" if config["project"].get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = create_model_from_config(_build_predict_config(config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    transform = build_transforms(config["data"]["image_size"], {**config["augmentation"], "enabled": False}, train=False)
    image = Image.open(Path(image_path)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.softmax(outputs, dim=1)[0].cpu().tolist()
    class_names = config["data"]["class_names"]
    pred_idx = int(torch.tensor(probs).argmax().item())
    return {"prediction": class_names[pred_idx], "confidence": probs[pred_idx], "probabilities": dict(zip(class_names, probs))}
