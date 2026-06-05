import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

DATASETS = {
    "plantvillage": os.path.join("data", "plantvillage dataset", "color"),
    "rice":         os.path.join("data", "rice"),
    "wheat":        os.path.join("data", "wheat"),
    "sugarcane":    os.path.join("data", "sugarcane"),
    "groundnut":    os.path.join("data", "groundnut"),
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_all_image_paths_and_labels(base_dir="."):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Pass 1: collect every class name from every dataset to find duplicates
    class_name_counts = Counter()
    for dataset_name, rel_path in DATASETS.items():
        dataset_dir = os.path.join(base_dir, rel_path)
        if not os.path.isdir(dataset_dir):
            continue
        for cls in sorted(os.listdir(dataset_dir)):
            cls_dir = os.path.join(dataset_dir, cls)
            if os.path.isdir(cls_dir):
                class_name_counts[cls] += 1

    duplicate_names = {name for name, count in class_name_counts.items() if count > 1}
    if duplicate_names:
        print(f"[INFO] Duplicate class names found across datasets: {duplicate_names}")
        print("[INFO] These will be prefixed with dataset_name___class to avoid merging.")

    # Pass 2: collect images with correct prefixed labels
    all_images = []
    all_labels_str = []

    for dataset_name, rel_path in DATASETS.items():
        dataset_dir = os.path.join(base_dir, rel_path)
        if not os.path.isdir(dataset_dir):
            print(f"[WARNING] Dataset directory not found, skipping: {dataset_dir}")
            continue

        for cls in sorted(os.listdir(dataset_dir)):
            cls_dir = os.path.join(dataset_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            # Prefix duplicate class names with dataset name
            if cls in duplicate_names:
                label = f"{dataset_name}___{cls}"
            else:
                label = cls

            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in valid_extensions:
                    continue
                full_path = os.path.join(cls_dir, fname)
                all_images.append(full_path)
                all_labels_str.append(label)

    # Build class list and numeric mapping
    all_classes = sorted(set(all_labels_str))
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    # Convert string labels to numeric indices
    all_labels = [class_to_idx[l] for l in all_labels_str]

    print(f"Total images found: {len(all_images)}")
    print(f"Total unique classes: {len(all_classes)}")

    return all_images, all_labels, all_classes, class_to_idx


def split_data(image_paths, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print("=" * 60)
    print("DATA LOADER — Sanity Check")
    print("=" * 60)

    all_images, all_labels, all_classes, class_to_idx = get_all_image_paths_and_labels(base_dir=PROJECT_ROOT)

    if len(all_images) == 0:
        print("\nNo images found. Check that your data/ folder is populated.")
    else:
        print(f"\nClasses ({len(all_classes)}):")
        label_counts = Counter(all_labels)
        for cls in all_classes:
            idx = class_to_idx[cls]
            print(f"  [{idx:3d}] {cls}: {label_counts[idx]}")

        print()
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(all_images, all_labels)

        print(f"\nSample train path : {X_train[0]}")
        print(f"Sample train label: {y_train[0]} ({all_classes[y_train[0]]})")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
