import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np


IMAGE_ID_PATTERN = re.compile(r"_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect whether minimal M2 Transformer data can be read by the existing pipeline."
    )
    parser.add_argument("--image_root", type=str, default="mini_data/images")
    parser.add_argument("--ann_root", type=str, default="mini_data/annotations")
    parser.add_argument("--features_path", type=str, default="mini_data/features/mini_detections.hdf5")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def parse_image_id_from_name(file_name):
    match = IMAGE_ID_PATTERN.search(file_name)
    if not match:
        raise ValueError(
            "file_name does not match the required pattern ending with an underscore "
            f"and digits, e.g. COCO_train2014_000000000001.jpg: {file_name}"
        )
    return int(match.group(1))


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required path is not a file: {path}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npy(path):
    arr = np.load(path)
    return np.asarray(arr, dtype=np.int64)


def check_basic_files(ann_root, features_path):
    required = {
        "captions_train2014.json": ann_root / "captions_train2014.json",
        "captions_val2014.json": ann_root / "captions_val2014.json",
        "coco_train_ids.npy": ann_root / "coco_train_ids.npy",
        "coco_dev_ids.npy": ann_root / "coco_dev_ids.npy",
        "coco_test_ids.npy": ann_root / "coco_test_ids.npy",
        "coco_restval_ids.npy": ann_root / "coco_restval_ids.npy",
        "features_path": Path(features_path),
    }
    for _, path in required.items():
        require_file(path)
    return required


def collect_annotation_maps(train_json, val_json):
    train_ann_map = {ann["id"]: ann for ann in train_json.get("annotations", [])}
    val_ann_map = {ann["id"]: ann for ann in val_json.get("annotations", [])}
    all_ids = list(train_ann_map.keys()) + list(val_ann_map.keys())
    duplicates = sorted({x for x in all_ids if all_ids.count(x) > 1})
    return train_ann_map, val_ann_map, duplicates


def validate_split_ids(split_name, ids, valid_map):
    missing = [int(x) for x in ids.tolist() if int(x) not in valid_map]
    if missing:
        raise ValueError(f"{split_name} contains annotation.id values missing from its JSON: {missing}")


def summarize_split_consistency(train_json, val_json, train_ids, dev_ids, test_ids, restval_ids):
    train_ann_map, val_ann_map, duplicates = collect_annotation_maps(train_json, val_json)
    if duplicates:
        raise ValueError(f"Duplicate annotation.id detected across JSON files: {duplicates}")

    validate_split_ids("train", train_ids, train_ann_map)
    validate_split_ids("dev", dev_ids, val_ann_map)
    validate_split_ids("test", test_ids, val_ann_map)
    validate_split_ids("restval", restval_ids, train_ann_map if len(restval_ids) else {})

    if len(train_ids) == 0 or len(dev_ids) == 0 or len(test_ids) == 0:
        raise ValueError(
            f"train/dev/test must all be non-empty, got train={len(train_ids)}, dev={len(dev_ids)}, test={len(test_ids)}"
        )

    return train_ann_map, val_ann_map


def build_image_lookup(coco_json):
    return {img["id"]: img for img in coco_json.get("images", [])}


def check_hdf5_alignment(train_json, val_json, features_path):
    train_images = build_image_lookup(train_json)
    val_images = build_image_lookup(val_json)
    missing_keys = []
    bad_file_names = []
    checked_samples = []

    with h5py.File(features_path, "r") as h5_file:
        h5_keys = sorted(list(h5_file.keys()))
        for split_name, coco_json, image_lookup in [
            ("train", train_json, train_images),
            ("val", val_json, val_images),
        ]:
            for ann in coco_json.get("annotations", []):
                image_id = ann["image_id"]
                if image_id not in image_lookup:
                    raise ValueError(
                        f"{split_name} annotation references missing image_id in images[]: {image_id}"
                    )
                file_name = image_lookup[image_id]["file_name"]
                parsed_image_id = parse_image_id_from_name(file_name)
                if parsed_image_id != int(image_id):
                    bad_file_names.append(
                        {
                            "file_name": file_name,
                            "annotation_image_id": int(image_id),
                            "parsed_image_id": parsed_image_id,
                        }
                    )
                key = f"{int(image_id)}_features"
                if key not in h5_file:
                    missing_keys.append(key)
                checked_samples.append(
                    {
                        "file_name": file_name,
                        "image_id": int(image_id),
                        "caption": ann["caption"],
                    }
                )

    if bad_file_names:
        raise ValueError(f"file_name to image_id mismatch detected: {bad_file_names}")
    if missing_keys:
        raise ValueError(f"Missing HDF5 feature keys: {sorted(set(missing_keys))}")

    return h5_keys, checked_samples


def import_repo_components():
    from data import COCO, DataLoader, ImageDetectionsField, TextField

    return COCO, DataLoader, ImageDetectionsField, TextField


def inspect_dataset_pipeline(image_root, ann_root, features_path, batch_size, num_workers):
    COCO, DataLoader, ImageDetectionsField, TextField = import_repo_components()

    image_field = ImageDetectionsField(
        detections_path=str(features_path),
        max_detections=50,
        load_in_tmp=False,
    )
    text_field = TextField(
        init_token="<bos>",
        eos_token="<eos>",
        lower=True,
        tokenize="spacy",
        remove_punctuation=True,
        nopoints=False,
    )
    dataset = COCO(image_field, text_field, str(image_root), str(ann_root), str(ann_root))
    train_dataset, val_dataset, test_dataset = dataset.splits

    text_field.build_vocab(train_dataset, val_dataset, min_freq=1)

    vocab = text_field.vocab
    special_presence = {
        "<bos>": "<bos>" in vocab.stoi,
        "<eos>": "<eos>" in vocab.stoi,
        "<pad>": "<pad>" in vocab.stoi,
        "<unk>": "<unk>" in vocab.stoi,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    batch = next(iter(train_loader))
    detections, captions = batch

    first_caption_ids = captions[0].tolist()
    decoded_caption = text_field.decode(captions[0])[0]

    return {
        "train_dataset_len": len(train_dataset),
        "val_dataset_len": len(val_dataset),
        "test_dataset_len": len(test_dataset),
        "vocab_size": len(vocab),
        "special_presence": special_presence,
        "detections_type": str(type(detections)),
        "detections_shape": tuple(detections.shape),
        "detections_dtype": str(detections.dtype),
        "captions_type": str(type(captions)),
        "captions_shape": tuple(captions.shape),
        "captions_dtype": str(captions.dtype),
        "first_caption_ids": first_caption_ids,
        "decoded_caption": decoded_caption,
    }


def main():
    args = parse_args()

    ann_root = Path(args.ann_root)
    image_root = Path(args.image_root)
    features_path = Path(args.features_path)

    files = check_basic_files(ann_root, features_path)

    train_json = load_json(files["captions_train2014.json"])
    val_json = load_json(files["captions_val2014.json"])
    train_ids = load_npy(files["coco_train_ids.npy"])
    dev_ids = load_npy(files["coco_dev_ids.npy"])
    test_ids = load_npy(files["coco_test_ids.npy"])
    restval_ids = load_npy(files["coco_restval_ids.npy"])

    summarize_split_consistency(train_json, val_json, train_ids, dev_ids, test_ids, restval_ids)
    h5_keys, checked_samples = check_hdf5_alignment(train_json, val_json, features_path)
    pipeline_info = inspect_dataset_pipeline(
        image_root=image_root,
        ann_root=ann_root,
        features_path=features_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    sample_example = checked_samples[0]

    print("Split counts:")
    print(f"  train: {len(train_ids)}")
    print(f"  dev: {len(dev_ids)}")
    print(f"  test: {len(test_ids)}")
    print(f"  restval: {len(restval_ids)}")
    print(f"Vocab size: {pipeline_info['vocab_size']}")
    print(f"HDF5 total keys: {len(h5_keys)}")
    print("Missing keys count: 0")
    print("Train sample example:")
    print(f"  file_name: {sample_example['file_name']}")
    print(f"  image_id: {sample_example['image_id']}")
    print(f"  caption: {sample_example['caption']}")
    print("Special tokens present:")
    for token, present in pipeline_info["special_presence"].items():
        print(f"  {token}: {present}")
    print("Batch inspection:")
    print(f"  detections.type: {pipeline_info['detections_type']}")
    print(f"  detections.shape: {pipeline_info['detections_shape']}")
    print(f"  detections.dtype: {pipeline_info['detections_dtype']}")
    print(f"  captions.type: {pipeline_info['captions_type']}")
    print(f"  captions.shape: {pipeline_info['captions_shape']}")
    print(f"  captions.dtype: {pipeline_info['captions_dtype']}")
    print(f"  first caption token ids: {pipeline_info['first_caption_ids']}")
    print(f"  decoded first caption: {pipeline_info['decoded_caption']}")
    print("Final result: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Final result: FAIL", file=sys.stderr)
        print(f"[ERROR] {exc.__class__.__name__}: {exc}", file=sys.stderr)
        raise
