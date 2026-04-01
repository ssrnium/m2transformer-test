import argparse
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_ID_PATTERN = re.compile(r"_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract minimal ResNet50 features for M2 Transformer."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_h5", type=str, required=True, help="Path to the output HDF5 file.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, cpu, cuda:0, etc. Default: auto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_h5 if it already exists.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def collect_image_paths(input_dir):
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    image_paths = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    return image_paths


def parse_image_id(filename):
    match = IMAGE_ID_PATTERN.search(filename)
    if not match:
        raise ValueError(
            "Filename does not match the required pattern ending with an underscore "
            f"and digits, e.g. COCO_train2014_000000000001.jpg: {filename}"
        )
    return int(match.group(1))


def build_feature_extractor(device):
    # We reuse the official torchvision preprocessing and remove the final FC layer,
    # keeping the global pooled 2048-dim representation.
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()
    preprocess = weights.transforms()
    return feature_extractor, preprocess


def extract_single_feature(image_path, preprocess, feature_extractor, device):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = feature_extractor(tensor)

    feature = feature.squeeze(-1).squeeze(-1)
    if feature.ndim != 2 or feature.shape != (1, 2048):
        raise RuntimeError(
            f"Unexpected feature shape for {image_path.name}: {tuple(feature.shape)}; expected (1, 2048)"
        )

    return feature.detach().cpu().numpy().astype(np.float32, copy=False)


def main():
    args = parse_args()

    output_path = Path(args.output_h5)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    image_paths = collect_image_paths(args.input_dir)
    if not image_paths:
        raise RuntimeError(f"No jpg/jpeg/png images found in: {args.input_dir}")

    feature_extractor, preprocess = build_feature_extractor(device)

    written = []
    try:
        with h5py.File(output_path, "w") as h5_file:
            for image_path in image_paths:
                image_id = parse_image_id(image_path.name)
                key = f"{image_id}_features"
                feature = extract_single_feature(image_path, preprocess, feature_extractor, device)
                h5_file.create_dataset(key, data=feature, dtype=np.float32)
                written.append((key, tuple(feature.shape)))
    except Exception:
        # If extraction fails midway, remove the partially written file so the
        # next run starts cleanly.
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        raise

    print(f"Device: {device}")
    print(f"Total images found: {len(image_paths)}")
    print(f"Successfully written: {len(written)}")
    for key, shape in written:
        print(f"{key}: shape={shape}")
    print(f"Output HDF5: {output_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc.__class__.__name__}: {exc}", file=sys.stderr)
        raise
