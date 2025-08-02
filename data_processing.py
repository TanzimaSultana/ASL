import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_train_dir = "./data/asl_alphabet_train"
data_test_dir = "./data/asl_alphabet_test"

def create_label_maps(data_dir):
    # Explicitly include A–Z, space, del, nothing
    class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]
    
    labels = [label for label in class_labels if os.path.isdir(os.path.join(data_dir, label))]
    label_map = {label: idx for idx, label in enumerate(labels)}
    reverse_map = {idx: label for label, idx in label_map.items()}
    
    return label_map, reverse_map

# ----------

def load_asl_alphabet_dataset(img_size, color_mode, max_image_per_class=None):
    """
    Loads ASL dataset for SVM, KNN, CNN or Transfer Learning.

    Args:
        img_size: image size to resize
        color_mode: 'grayscale' or 'rgb'
        max_image_per_class: limit number (500) of images per class (None = no limit)

    Returns:
        (X_train, X_test, y_train, y_test), reverse_map
    """
    X, y = [], []
    label_map, reverse_map = create_label_maps(data_train_dir)

    for label in label_map:
        folder = os.path.join(data_train_dir, label)
        files = os.listdir(folder)
        if max_image_per_class:
            files = files[:max_image_per_class]

        for file in files:
            path = os.path.join(folder, file)

            if color_mode == 'grayscale':
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            elif color_mode == 'rgb':
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("color_mode must be 'grayscale' or 'rgb'")

            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label_map[label])

    X = np.array(X).astype("float32") / 255.0
    y = np.array(y)

    if color_mode == 'grayscale':
        X = X.reshape(-1, img_size, img_size, 1)
    else:
        X = X.reshape(-1, img_size, img_size, 3)

    return train_test_split(X, y, test_size=0.2), reverse_map

def load_asl_alphabet_dataset1(img_size, color_mode, max_image_per_class=None):
    """
    Loads ASL dataset and splits into train/val/test.

    Args:
        img_size: image size to resize
        color_mode: 'grayscale' or 'rgb'
        max_image_per_class: limit number of images per class (None = no limit)

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), reverse_map
    """
    X, y = [], []
    label_map, reverse_map = create_label_maps(data_train_dir)

    for label in label_map:
        folder = os.path.join(data_train_dir, label)
        files = os.listdir(folder)
        if max_image_per_class:
            files = files[:max_image_per_class]

        for file in files:
            path = os.path.join(folder, file)

            if color_mode == 'grayscale':
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            elif color_mode == 'rgb':
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("color_mode must be 'grayscale' or 'rgb'")

            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label_map[label])

    X = np.array(X).astype("float32") / 255.0
    y = np.array(y)

    if color_mode == 'grayscale':
        X = X.reshape(-1, img_size, img_size, 1)
    else:
        X = X.reshape(-1, img_size, img_size, 3)

    # First split off test set (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    # Now split remaining into train (70%) and val (20%)
    val_ratio = 0.20 / (1 - 0.10)  # adjust relative to remaining data
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), reverse_map