import os
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "plantvillage": os.path.join("data", "plantvillage dataset", "color"),
    "rice":         os.path.join("data", "rice"),
    "wheat":        os.path.join("data", "wheat"),
    "sugarcane":    os.path.join("data", "sugarcane"),
    "groundnut":    os.path.join("data", "groundnut"),
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# COLLECT IMAGE PATHS + LABELS
# ============================================================

def get_all_image_paths_and_labels(base_dir="."):

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".webp"
    }

    # --------------------------------------------------------
    # PASS 1: FIND DUPLICATE CLASS NAMES
    # --------------------------------------------------------

    class_name_counts = Counter()

    for dataset_name, rel_path in DATASETS.items():

        dataset_dir = os.path.join(base_dir, rel_path)

        if not os.path.isdir(dataset_dir):
            continue

        for cls in sorted(os.listdir(dataset_dir)):

            cls_dir = os.path.join(dataset_dir, cls)

            if os.path.isdir(cls_dir):
                class_name_counts[cls] += 1

    duplicate_names = {
        name
        for name, count in class_name_counts.items()
        if count > 1
    }

    if duplicate_names:
        print(
            f"[INFO] Duplicate class names found: "
            f"{duplicate_names}"
        )
        print(
            "[INFO] Prefixing duplicates with "
            "dataset_name___class"
        )

    # --------------------------------------------------------
    # PASS 2: COLLECT IMAGES
    # --------------------------------------------------------

    all_images = []
    all_labels_str = []

    for dataset_name, rel_path in DATASETS.items():

        dataset_dir = os.path.join(base_dir, rel_path)

        if not os.path.isdir(dataset_dir):
            print(
                f"[WARNING] Missing dataset: "
                f"{dataset_dir}"
            )
            continue

        for cls in sorted(os.listdir(dataset_dir)):

            cls_dir = os.path.join(dataset_dir, cls)

            if not os.path.isdir(cls_dir):
                continue

            if cls in duplicate_names:
                label = f"{dataset_name}___{cls}"
            else:
                label = cls

            for fname in os.listdir(cls_dir):

                ext = os.path.splitext(fname)[1].lower()

                if ext not in valid_extensions:
                    continue

                full_path = os.path.join(
                    cls_dir,
                    fname
                )

                all_images.append(full_path)
                all_labels_str.append(label)

    # --------------------------------------------------------
    # BUILD CLASS MAP
    # --------------------------------------------------------

    all_classes = sorted(set(all_labels_str))

    class_to_idx = {
        cls: idx
        for idx, cls in enumerate(all_classes)
    }

    all_labels = [
        class_to_idx[label]
        for label in all_labels_str
    ]

    print(f"Total images found: {len(all_images)}")
    print(f"Total unique classes: {len(all_classes)}")

    return (
        all_images,
        all_labels,
        all_classes,
        class_to_idx
    )


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

def split_data(
    image_paths,
    labels,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):

    assert abs(
        train_ratio + val_ratio + test_ratio - 1.0
    ) < 1e-6

    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths,
        labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels
    )

    relative_test_ratio = (
        test_ratio /
        (val_ratio + test_ratio)
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=y_temp
    )

    print(
        f"Train: {len(X_train)} | "
        f"Val: {len(X_val)} | "
        f"Test: {len(X_test)}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_and_preprocess(
    path,
    label,
    img_size=IMG_SIZE
):

    img = tf.io.read_file(path)

    img = tf.image.decode_image(
        img,
        channels=3,
        expand_animations=False
    )

    img = tf.image.resize(
        img,
        img_size
    )

    img = tf.cast(
        img,
        tf.float32
    ) / 255.0

    return img, label


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        "horizontal_and_vertical"
    ),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])


# ============================================================
# TF.DATA PIPELINE
# ============================================================

def build_tf_datasets(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    batch_size=BATCH_SIZE
):

    def make_dataset(
        paths,
        labels,
        shuffle=False,
        augment=False
    ):

        ds = tf.data.Dataset.from_tensor_slices(
            (paths, labels)
        )

        ds = ds.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if shuffle:
            ds = ds.shuffle(
                buffer_size=1000
            )

        if augment:
            ds = ds.map(
                lambda x, y: (
                    data_augmentation(
                        x,
                        training=True
                    ),
                    y
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        ds = ds.batch(batch_size)

        ds = ds.prefetch(
            tf.data.AUTOTUNE
        )

        return ds

    train_ds = make_dataset(
        X_train,
        y_train,
        shuffle=True,
        augment=True
    )

    val_ds = make_dataset(
        X_val,
        y_val
    )

    test_ds = make_dataset(
        X_test,
        y_test
    )

    return (
        train_ds,
        val_ds,
        test_ds
    )


# ============================================================
# SANITY CHECK
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    print("=" * 60)
    print("DATA LOADER — SANITY CHECK")
    print("=" * 60)

    (
        all_images,
        all_labels,
        all_classes,
        class_to_idx
    ) = get_all_image_paths_and_labels(
        base_dir=PROJECT_ROOT
    )

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(
            all_images,
            all_labels
        )
    )

    print()
    print(
        f"Sample train image: "
        f"{X_train[0]}"
    )

    print(
        f"Sample label: "
        f"{y_train[0]}"
    )

    train_ds, val_ds, test_ds = (
        build_tf_datasets(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )
    )

    for images, labels in train_ds.take(1):

        print()
        print(
            f"Batch image shape: "
            f"{images.shape}"
        )

        print(
            f"Batch label shape: "
            f"{labels.shape}"
        )

        print(
            f"Pixel range: "
            f"{images.numpy().min():.2f} "
            f"to "
            f"{images.numpy().max():.2f}"
        )

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)