import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np


IMAGE_ID_PATTERN = re.compile(r"_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare minimal COCO-style caption annotations for M2 Transformer."
    )
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input sample list JSON.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write COCO-style outputs.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio. Default: 0.8")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Dev split ratio. Default: 0.1")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio. Default: 0.1")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for split shuffling. Default: 1234")
    return parser.parse_args()


def parse_image_id(file_name):
    match = IMAGE_ID_PATTERN.search(file_name)
    if not match:
        raise ValueError(
            "file_name does not match the required pattern ending with an underscore "
            f"and digits, e.g. COCO_train2014_000000000001.jpg: {file_name}"
        )
    return int(match.group(1))


def load_samples(input_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("input_json must contain a JSON list.")
    if not data:
        raise ValueError("input_json is empty.")

    normalized = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Sample at index {index} is not a JSON object.")
        if "file_name" not in item:
            raise ValueError(f"Sample at index {index} is missing required field: file_name")
        if "caption" not in item:
            raise ValueError(f"Sample at index {index} is missing required field: caption")

        file_name = item["file_name"]
        caption = item["caption"]
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(f"Sample at index {index} has invalid file_name: {file_name!r}")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Sample at index {index} has invalid caption: {caption!r}")

        image_id = parse_image_id(file_name)
        normalized.append(
            {
                "sample_index": index,
                "file_name": file_name,
                "caption": caption,
                "image_id": image_id,
            }
        )
    return normalized


def validate_ratios(train_ratio, dev_ratio, test_ratio):
    ratios = [train_ratio, dev_ratio, test_ratio]
    if any(r <= 0 for r in ratios):
        raise ValueError("train_ratio, dev_ratio, and test_ratio must all be > 0.")
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + dev_ratio + test_ratio must equal 1.0, got {ratio_sum:.6f}"
        )


def split_samples(samples, train_ratio, dev_ratio, test_ratio, seed):
    validate_ratios(train_ratio, dev_ratio, test_ratio)

    total = len(samples)
    if total < 3:
        raise ValueError(
            f"At least 3 samples are required to create non-empty train/dev/test splits, got {total}."
        )

    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    raw_counts = np.array([train_ratio, dev_ratio, test_ratio], dtype=np.float64) * total
    counts = np.floor(raw_counts).astype(np.int64)

    # Ensure every split is non-empty for small datasets.
    counts = np.maximum(counts, 1)

    # Adjust to exactly total while keeping all splits non-empty.
    while counts.sum() > total:
        idx = int(np.argmax(counts))
        if counts[idx] == 1:
            break
        counts[idx] -= 1

    if counts.sum() != total:
        remainders = raw_counts - np.floor(raw_counts)
        order = list(np.argsort(-remainders))
        while counts.sum() < total:
            for idx in order:
                counts[idx] += 1
                if counts.sum() == total:
                    break

    if counts.sum() != total or np.any(counts <= 0):
        raise ValueError(
            f"Could not create non-empty train/dev/test splits from {total} samples. Computed counts: {counts.tolist()}"
        )

    train_count, dev_count, test_count = counts.tolist()
    train_samples = shuffled[:train_count]
    dev_samples = shuffled[train_count:train_count + dev_count]
    test_samples = shuffled[train_count + dev_count:train_count + dev_count + test_count]

    if not train_samples or not dev_samples or not test_samples:
        raise ValueError(
            "Split generation produced an empty split, which is not allowed for this minimal setup."
        )

    return train_samples, dev_samples, test_samples


def build_coco_json(samples, annotation_id_start):
    images = []
    annotations = []
    seen_images = set()
    annotation_ids = []
    next_annotation_id = annotation_id_start

    for sample in samples:
        image_key = (sample["image_id"], sample["file_name"])
        if image_key not in seen_images:
            images.append(
                {
                    "id": sample["image_id"],
                    "file_name": sample["file_name"],
                }
            )
            seen_images.add(image_key)

        annotation = {
            "id": next_annotation_id,
            "image_id": sample["image_id"],
            "caption": sample["caption"],
        }
        annotations.append(annotation)
        annotation_ids.append(next_annotation_id)
        next_annotation_id += 1

    coco_json = {
        "info": {},
        "licenses": [],
        "type": "captions",
        "images": images,
        "annotations": annotations,
    }
    return coco_json, annotation_ids, next_annotation_id


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_npy(path, values):
    arr = np.asarray(values, dtype=np.int64)
    np.save(path, arr)


def main():
    args = parse_args()

    input_json_path = Path(args.input_json)
    if not input_json_path.exists():
        raise FileNotFoundError(f"input_json does not exist: {input_json_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(input_json_path)
    train_samples, dev_samples, test_samples = split_samples(
        samples, args.train_ratio, args.dev_ratio, args.test_ratio, args.seed
    )

    train_json, train_annotation_ids, next_annotation_id = build_coco_json(train_samples, 1)
    val_json, val_annotation_ids, _ = build_coco_json(dev_samples + test_samples, next_annotation_id)

    dev_annotation_ids = val_annotation_ids[:len(dev_samples)]
    test_annotation_ids = val_annotation_ids[len(dev_samples):]

    train_json_path = output_dir / "captions_train2014.json"
    val_json_path = output_dir / "captions_val2014.json"
    train_ids_path = output_dir / "coco_train_ids.npy"
    dev_ids_path = output_dir / "coco_dev_ids.npy"
    test_ids_path = output_dir / "coco_test_ids.npy"
    restval_ids_path = output_dir / "coco_restval_ids.npy"

    write_json(train_json_path, train_json)
    write_json(val_json_path, val_json)
    write_npy(train_ids_path, train_annotation_ids)
    write_npy(dev_ids_path, dev_annotation_ids)
    write_npy(test_ids_path, test_annotation_ids)
    write_npy(restval_ids_path, np.asarray([], dtype=np.int64))

    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Dev samples: {len(dev_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print("Generated files:")
    for path in [
        train_json_path,
        val_json_path,
        train_ids_path,
        dev_ids_path,
        test_ids_path,
        restval_ids_path,
    ]:
        print(f"  - {path.resolve()}")
    print(f"coco_train_ids.npy count: {len(train_annotation_ids)}")
    print(f"coco_dev_ids.npy count: {len(dev_annotation_ids)}")
    print(f"coco_test_ids.npy count: {len(test_annotation_ids)}")
    print("Train annotation example:")
    print(json.dumps(train_json["annotations"][0], ensure_ascii=False, indent=2))
    print("Val annotation example:")
    print(json.dumps(val_json["annotations"][0], ensure_ascii=False, indent=2))
    print("Image placement note:")
    print("  - Put training images under: <your_root>/images/train2014/")
    print("  - Put validation/test images under: <your_root>/images/val2014/")
    print("  - file_name should look like COCO_train2014_000000000001.jpg")
    print("  - This naming is required because the current repo parses image_id with")
    print("    int(x.split('_')[-1].split('.')[0]) in data/field.py.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc.__class__.__name__}: {exc}", file=sys.stderr)
        raise
