from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


CLASS_MAP = {
    "positive": "crack",
    "crack": "crack",
    "cracks": "crack",
    "negative": "non_crack",
    "non_crack": "non_crack",
    "non-crack": "non_crack",
    "noncrack": "non_crack",
}


def normalize_class_name(name: str) -> str:
    key = name.lower().strip()
    if key not in CLASS_MAP:
        raise ValueError(f"Unsupported class directory name: {name}")
    return CLASS_MAP[key]


def collect_records(src_root: str | Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(Path(src_root).iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = normalize_class_name(class_dir.name)
        label = 1 if class_name == "crack" else 0
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                rows.append({"image_path": str(image_path), "class_name": class_name, "label": label})
    if not rows:
        raise RuntimeError(f"No images found under {src_root}")
    return pd.DataFrame(rows)


def compute_split_sizes(total: int) -> tuple[int, int, int]:
    if total == 700:
        return 500, 100, 100
    train = round(total * 0.714)
    val = round(total * 0.143)
    test = total - train - val
    return train, val, test


def save_resized_copy(src: Path, dst: Path, image_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        image.save(dst)


def build_output_filename(src: Path) -> str:
    # Preserve a stable link back to the source tree and avoid collisions on duplicate filenames.
    stem = "_".join(src.with_suffix("").parts[-3:])
    return f"{stem}{src.suffix.lower()}"


def prepare_dataset(src: str | Path, out_dir: str | Path, splits_out: str | Path, image_size: int, seed: int) -> dict:
    records = collect_records(src)
    train_size, val_size, test_size = compute_split_sizes(len(records))
    train_df, temp_df = train_test_split(
        records,
        train_size=train_size,
        stratify=records["label"],
        random_state=seed,
    )
    relative_test_ratio = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=seed,
    )
    out_root = Path(out_dir)
    splits_root = Path(splits_out)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        saved_rows = []
        for _, row in df.reset_index(drop=True).iterrows():
            src_path = Path(row["image_path"])
            dst_name = build_output_filename(src_path)
            dst_path = out_root / split_name / row["class_name"] / dst_name
            save_resized_copy(src_path, dst_path, image_size)
            saved_rows.append(
                {
                    "image_path": str(dst_path),
                    "source_path": str(src_path),
                    "class_name": row["class_name"],
                    "label": int(row["label"]),
                }
            )
        split_df = pd.DataFrame(saved_rows)
        splits_root.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(splits_root / f"{split_name}.csv", index=False)
    return {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
