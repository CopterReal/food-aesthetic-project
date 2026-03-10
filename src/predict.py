import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"

FEATURE_EXTRACTOR_PATH = MODELS_DIR / "feature_extractor.keras"
CLASSIFIER_PATH = MODELS_DIR / "pairwise_classifier.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

IMG_SIZE = (224, 224)

MENU_TO_FOLDER = {
    "sushi": "sushi",
    "ramen": "ramen",
    "pizza": "pizza",
    "burger": "hamburger",
    "hamburger": "hamburger",
    "dessert": "dessert",
}


# =========================
# GENERAL UTILS
# =========================
def normalize_menu_name(menu_name: str) -> str:
    menu = str(menu_name).strip().lower()
    return MENU_TO_FOLDER.get(menu, menu)


def build_image_index(root_dir: Path) -> dict:
    image_index = {}
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_ext:
            image_index[p.name] = str(p)

    return image_index


def infer_menu_from_filename(image_name: str, training_index: dict) -> str:
    if image_name in training_index:
        p = Path(training_index[image_name])
        return p.parent.name.lower()
    return "unknown"


def resolve_test_image_path(image_name: str, test_image_index: dict, training_image_index: dict) -> str:
    if image_name in test_image_index:
        return test_image_index[image_name]
    if image_name in training_image_index:
        return training_image_index[image_name]
    raise FileNotFoundError(f"Cannot find image: {image_name}")


# =========================
# CNN FEATURE EXTRACTION
# =========================
def load_img_for_cnn(image_path: str) -> np.ndarray:
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img).astype(np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


def extract_cnn_feature_cached(image_path: str, feature_extractor: tf.keras.Model, cache: dict) -> np.ndarray:
    if image_path in cache:
        return cache[image_path]

    img = load_img_for_cnn(image_path)
    feat = feature_extractor.predict(np.expand_dims(img, axis=0), verbose=0)[0]
    feat = feat.astype(np.float32)
    cache[image_path] = feat
    return feat


# =========================
# IMAGE / COLOR UTILS
# =========================
def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc

    deltac = maxc - minc
    s = np.where(maxc == 0, 0, deltac / np.maximum(maxc, 1e-8))

    h = np.zeros_like(maxc)

    mask = deltac > 1e-8
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)

    rc[mask] = (maxc[mask] - r[mask]) / deltac[mask]
    gc[mask] = (maxc[mask] - g[mask]) / deltac[mask]
    bc[mask] = (maxc[mask] - b[mask]) / deltac[mask]

    mask_r = mask & (r == maxc)
    mask_g = mask & (g == maxc)
    mask_b = mask & (b == maxc)

    h[mask_r] = (bc - gc)[mask_r]
    h[mask_g] = (2.0 + rc - bc)[mask_g]
    h[mask_b] = (4.0 + gc - rc)[mask_b]

    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    lap = up + down + left + right - 4.0 * center
    return float(np.var(lap))


# =========================
# BASIC HANDCRAFTED FEATURES
# =========================
def compute_handcrafted_features(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    rgb = np.asarray(img).astype(np.float32) / 255.0

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    brightness_p10 = float(np.percentile(gray, 10))
    brightness_p90 = float(np.percentile(gray, 90))

    hsv = rgb_to_hsv_np(rgb)
    sat = hsv[..., 1]
    val = hsv[..., 2]

    saturation_mean = float(np.mean(sat))
    saturation_std = float(np.std(sat))
    value_mean = float(np.mean(val))

    sharpness = laplacian_variance(gray)

    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    colorfulness = float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

    h, w = gray.shape
    h1, h2 = int(0.25 * h), int(0.75 * h)
    w1, w2 = int(0.25 * w), int(0.75 * w)
    center_region = gray[h1:h2, w1:w2]
    center_brightness = float(np.mean(center_region))
    center_contrast = float(np.std(center_region))

    return np.array([
        brightness_mean,
        brightness_std,
        brightness_p10,
        brightness_p90,
        saturation_mean,
        saturation_std,
        value_mean,
        sharpness,
        colorfulness,
        center_brightness,
        center_contrast,
    ], dtype=np.float32)


# =========================
# ADVANCED FOOD FEATURES
# =========================
def compute_advanced_food_features(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    h, w = gray.shape
    cy1, cy2 = int(0.25 * h), int(0.75 * h)
    cx1, cx2 = int(0.25 * w), int(0.75 * w)

    center_gray = gray[cy1:cy2, cx1:cx2]
    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[cy1:cy2, cx1:cx2] = False
    border_gray = gray[border_mask]

    center_mean = float(np.mean(center_gray))
    border_mean = float(np.mean(border_gray))
    center_emphasis = center_mean - border_mean

    center_sharp = laplacian_variance(center_gray)
    border_img = gray.copy()
    border_img[cy1:cy2, cx1:cx2] = np.mean(border_gray)
    border_sharp = laplacian_variance(border_img)
    focus_contrast = center_sharp - border_sharp

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad_mag = np.sqrt(gx**2 + gy**2)
    edge_density = float(np.mean(grad_mag > np.percentile(grad_mag, 75)))

    hsv = rgb_to_hsv_np(rgb)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    fg_mask = (sat > np.percentile(sat, 55)) & (val > np.percentile(val, 35))
    foreground_ratio = float(np.mean(fg_mask))

    empty_mask = (sat < np.percentile(sat, 25)) & (gray > np.percentile(gray, 60))
    empty_space_ratio = float(np.mean(empty_mask))

    warm_mask = (
        (hue < 0.12) |
        ((hue > 0.90) & (hue <= 1.0)) |
        ((hue > 0.12) & (hue < 0.18))
    )
    warm_ratio = float(np.mean(warm_mask))

    hue_std = float(np.std(hue))

    sat_mean = float(np.mean(sat))
    sat_std = float(np.std(sat))
    sat_balance = float(sat_mean / (sat_std + 1e-6))

    analogous_score = float(np.exp(-8.0 * hue_std))

    hist, _ = np.histogram(hue, bins=24, range=(0.0, 1.0), density=True)
    hist = hist / (hist.sum() + 1e-8)
    best_comp = 0.0
    for i in range(12):
        j = (i + 12) % 24
        score = hist[i] + hist[j]
        if score > best_comp:
            best_comp = float(score)

    left = gray[:, :w // 2]
    right = gray[:, w - left.shape[1]:]
    right_flip = np.fliplr(right)
    symmetry_score = float(1.0 - np.mean(np.abs(left - right_flip)))

    pts = [
        (int(h * 1 / 3), int(w * 1 / 3)),
        (int(h * 1 / 3), int(w * 2 / 3)),
        (int(h * 2 / 3), int(w * 1 / 3)),
        (int(h * 2 / 3), int(w * 2 / 3)),
    ]
    patch_vals = []
    r = max(4, min(h, w) // 20)
    for py, px in pts:
        y1, y2 = max(0, py - r), min(h, py + r)
        x1, x2 = max(0, px - r), min(w, px + r)
        patch_vals.append(np.mean(gray[y1:y2, x1:x2]))
    thirds_emphasis = float(np.mean(patch_vals) - np.mean(gray))

    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    color_sep = float(np.std(rg) + np.std(yb))

    return np.array([
        center_emphasis,
        focus_contrast,
        edge_density,
        foreground_ratio,
        empty_space_ratio,
        warm_ratio,
        hue_std,
        sat_balance,
        analogous_score,
        best_comp,
        symmetry_score,
        thirds_emphasis,
        color_sep,
    ], dtype=np.float32)


def extract_hc_feature_cached(image_path: str, cache: dict) -> np.ndarray:
    if image_path in cache:
        return cache[image_path]

    basic = compute_handcrafted_features(image_path)
    advanced = compute_advanced_food_features(image_path)
    feat = np.concatenate([basic, advanced], axis=0).astype(np.float32)

    cache[image_path] = feat
    return feat


# =========================
# PAIR FEATURE
# =========================
def get_category_one_hot(menu_name: str) -> np.ndarray:
    categories = ["sushi", "ramen", "pizza", "hamburger", "dessert"]
    menu_norm = normalize_menu_name(menu_name)

    one_hot = np.zeros(len(categories), dtype=np.float32)
    if menu_norm in categories:
        one_hot[categories.index(menu_norm)] = 1.0

    return one_hot


def build_pair_feature(
    cnn1: np.ndarray,
    cnn2: np.ndarray,
    hc1: np.ndarray,
    hc2: np.ndarray,
    menu_name: str
) -> np.ndarray:
    cnn_abs_diff = np.abs(cnn1 - cnn2)
    hc_diff = hc1 - hc2
    hc_abs_diff = np.abs(hc_diff)
    one_hot = get_category_one_hot(menu_name)

    feat = np.concatenate([
        cnn_abs_diff,
        hc_diff,
        hc_abs_diff,
        one_hot
    ], axis=0)

    return feat.astype(np.float32)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--test_images", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    test_csv_path = Path(args.test_csv)
    test_images_dir = Path(args.test_images)
    output_csv_path = Path(args.output_csv) if args.output_csv else test_csv_path.parent / "test_filled.csv"

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    global IMG_SIZE
    IMG_SIZE = tuple(metadata["img_size"])

    feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)
    clf = joblib.load(CLASSIFIER_PATH)

    df = pd.read_csv(test_csv_path)

    required_cols = ["Image 1", "Image 2"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in test CSV: {col}")

    test_image_index = build_image_index(test_images_dir)
    training_image_index = build_image_index(DATA_DIR)

    cnn_cache = {}
    hc_cache = {}
    winners = []

    for idx, row in df.iterrows():
        img1_name = row["Image 1"]
        img2_name = row["Image 2"]

        if "Menu" in df.columns and pd.notna(row["Menu"]):
            menu_name = str(row["Menu"])
        else:
            menu_name = infer_menu_from_filename(img1_name, training_image_index)

        img1_path = resolve_test_image_path(img1_name, test_image_index, training_image_index)
        img2_path = resolve_test_image_path(img2_name, test_image_index, training_image_index)

        cnn1 = extract_cnn_feature_cached(img1_path, feature_extractor, cnn_cache)
        cnn2 = extract_cnn_feature_cached(img2_path, feature_extractor, cnn_cache)

        hc1 = extract_hc_feature_cached(img1_path, hc_cache)
        hc2 = extract_hc_feature_cached(img2_path, hc_cache)

        x12 = build_pair_feature(cnn1, cnn2, hc1, hc2, menu_name).reshape(1, -1)
        x21 = build_pair_feature(cnn2, cnn1, hc2, hc1, menu_name).reshape(1, -1)

        p12 = clf.predict_proba(x12)[0][1]
        p21 = clf.predict_proba(x21)[0][1]

        score_img1 = (p12 + (1.0 - p21)) / 2.0
        winner = 1 if score_img1 >= 0.5 else 2
        winners.append(winner)

        print(
            f"[{idx+1}/{len(df)}] {img1_name} vs {img2_name} "
            f"-> p12={p12:.4f}, p21={p21:.4f}, score1={score_img1:.4f}, Winner={winner}"
        )

    df["Winner"] = winners
    df.to_csv(output_csv_path, index=False)

    print(f"\n[INFO] Prediction finished. Output saved to: {output_csv_path}")


if __name__ == "__main__":
    main()