import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"
CSV_PATH = DATA_DIR / "data_from_questionaire.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_EXTRACTOR_PATH = MODELS_DIR / "feature_extractor.keras"
CLASSIFIER_PATH = MODELS_DIR / "pairwise_classifier.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"


# =========================
# SETTINGS
# =========================
IMG_SIZE = (224, 224)
RANDOM_STATE = 42
N_SPLITS = 5

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


def resolve_image_path(image_name: str, menu_name: str, image_index: dict) -> str:
    menu_folder = normalize_menu_name(menu_name)
    candidate = DATA_DIR / menu_folder / image_name

    if candidate.exists():
        return str(candidate)

    if image_name in image_index:
        return image_index[image_name]

    raise FileNotFoundError(f"Cannot find image: {image_name} (menu={menu_name})")


# =========================
# CNN FEATURE EXTRACTION
# =========================
def load_img_for_cnn(image_path: str) -> np.ndarray:
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img).astype(np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


def create_feature_extractor() -> tf.keras.Model:
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False
    return base_model


def extract_cnn_features_for_unique_images(feature_extractor: tf.keras.Model, image_paths: list) -> dict:
    cache = {}
    batch_size = 32
    unique_paths = sorted(list(set(image_paths)))

    print(f"[INFO] Extracting CNN features for {len(unique_paths)} unique images...")

    batch_imgs = []
    batch_paths = []

    def flush_batch(imgs, paths):
        if not imgs:
            return
        x = np.stack(imgs, axis=0)
        feats = feature_extractor.predict(x, verbose=0)
        for p, f in zip(paths, feats):
            cache[p] = f.astype(np.float32)

    for p in unique_paths:
        img = load_img_for_cnn(p)
        batch_imgs.append(img)
        batch_paths.append(p)

        if len(batch_imgs) >= batch_size:
            flush_batch(batch_imgs, batch_paths)
            batch_imgs = []
            batch_paths = []

    flush_batch(batch_imgs, batch_paths)
    return cache


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


def extract_handcrafted_features_for_unique_images(image_paths: list) -> dict:
    cache = {}
    unique_paths = sorted(list(set(image_paths)))

    print(f"[INFO] Extracting handcrafted features for {len(unique_paths)} unique images...")

    for p in unique_paths:
        basic = compute_handcrafted_features(p)
        advanced = compute_advanced_food_features(p)
        cache[p] = np.concatenate([basic, advanced], axis=0).astype(np.float32)

    return cache


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


def compute_sample_weight(row: pd.Series) -> float:
    if "Num Vote 1" in row and "Num Vote 2" in row:
        v1 = float(row["Num Vote 1"])
        v2 = float(row["Num Vote 2"])
        total = max(v1 + v2, 1.0)
        margin = abs(v1 - v2) / total

        if margin < 0.03:
            return 0.20
        elif margin < 0.07:
            return 0.50
        elif margin < 0.15:
            return 1.00
        elif margin < 0.25:
            return 1.50
        else:
            return 2.20

    return 1.0


def make_dataset_from_df(df_part, img1_paths, img2_paths, cnn_cache, hc_cache):
    X = []
    y = []
    sample_weights = []

    for i, row in df_part.iterrows():
        p1 = img1_paths[i]
        p2 = img2_paths[i]

        cnn1 = cnn_cache[p1]
        cnn2 = cnn_cache[p2]
        hc1 = hc_cache[p1]
        hc2 = hc_cache[p2]

        w = compute_sample_weight(row)
        winner = int(row["Winner"])

        X.append(build_pair_feature(cnn1, cnn2, hc1, hc2, row["Menu"]))
        y.append(1 if winner == 1 else 0)
        sample_weights.append(w)

        X.append(build_pair_feature(cnn2, cnn1, hc2, hc1, row["Menu"]))
        y.append(0 if winner == 1 else 1)
        sample_weights.append(w)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    return X, y, sample_weights


# =========================
# MODEL
# =========================
def build_final_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.022,
            subsample=0.82,
            colsample_bytree=0.70,
            reg_lambda=3.5,
            min_child_weight=3,
            gamma=0.20,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=52,
            n_jobs=-1,
        ))
    ])


def fit_model_with_sample_weight(model, X, y, sample_weight):
    step_name = model.steps[-1][0]
    param_name = f"{step_name}__sample_weight"
    model.fit(X, y, **{param_name: sample_weight})
    return model


def cross_validate_single_model(df, img1_paths, img2_paths, cnn_cache, hc_cache):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    print("[INFO] Cross-validating single fixed config...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["Winner"]), start=1):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        X_train, y_train, w_train = make_dataset_from_df(
            df_train, img1_paths, img2_paths, cnn_cache, hc_cache
        )
        X_val, y_val, _ = make_dataset_from_df(
            df_val, img1_paths, img2_paths, cnn_cache, hc_cache
        )

        model = build_final_model()
        fit_model_with_sample_weight(model, X_train, y_train, w_train)

        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        fold_scores.append(acc)

        print(f"[INFO]   Fold {fold_idx}/{N_SPLITS} accuracy = {acc:.4f}")

    mean_acc = float(np.mean(fold_scores))
    std_acc = float(np.std(fold_scores))

    return {
        "scores": fold_scores,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
    }


# =========================
# MAIN
# =========================
def main():
    print("[INFO] Loading training CSV...")
    df = pd.read_csv(CSV_PATH)

    required_cols = ["Image 1", "Image 2", "Menu", "Winner"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"[INFO] Total original pairs: {len(df)}")

    print("[INFO] Building image index...")
    image_index = build_image_index(DATA_DIR)

    img1_paths = {}
    img2_paths = {}

    for i, row in df.iterrows():
        img1_paths[i] = resolve_image_path(row["Image 1"], row["Menu"], image_index)
        img2_paths[i] = resolve_image_path(row["Image 2"], row["Menu"], image_index)

    print("[INFO] Creating feature extractor...")
    feature_extractor = create_feature_extractor()
    feature_extractor.save(FEATURE_EXTRACTOR_PATH)

    all_paths = list(img1_paths.values()) + list(img2_paths.values())

    cnn_cache = extract_cnn_features_for_unique_images(feature_extractor, all_paths)
    hc_cache = extract_handcrafted_features_for_unique_images(all_paths)

    print("[INFO] Building full pairwise dataset...")
    X_full, y_full, w_full = make_dataset_from_df(df, img1_paths, img2_paths, cnn_cache, hc_cache)

    print(f"[INFO] X_full shape: {X_full.shape}")
    print(f"[INFO] y_full shape: {y_full.shape}")

    print("[INFO] Running cross-validation on original dataframe before augmentation...")
    cv_result = cross_validate_single_model(df, img1_paths, img2_paths, cnn_cache, hc_cache)

    print("\n[RESULT] Cross-validation summary")
    print(
        f'  - fixed_xgb: mean={cv_result["mean_accuracy"]:.4f}, '
        f'std={cv_result["std_accuracy"]:.4f}, '
        f'scores={[round(s, 4) for s in cv_result["scores"]]}'
    )

    print(f'\n[RESULT] Fixed Config CV Accuracy = {cv_result["mean_accuracy"]:.4f} ± {cv_result["std_accuracy"]:.4f}')

    final_model = build_final_model()

    print("[INFO] Training final model on full dataset...")
    fit_model_with_sample_weight(final_model, X_full, y_full, w_full)

    print("[INFO] Evaluating final model on full dataset (sanity check only)...")
    full_pred = final_model.predict(X_full)
    full_acc = accuracy_score(y_full, full_pred)

    print(f"\n[RESULT] Full-data Accuracy (sanity check) = {full_acc:.4f}\n")
    print(classification_report(y_full, full_pred, digits=4))

    print("[INFO] Saving classifier...")
    joblib.dump(final_model, CLASSIFIER_PATH)

    metadata = {
        "img_size": IMG_SIZE,
        "csv_path": str(CSV_PATH),
        "feature_extractor_path": str(FEATURE_EXTRACTOR_PATH),
        "classifier_path": str(CLASSIFIER_PATH),
        "menu_to_folder": MENU_TO_FOLDER,
        "basic_handcrafted_features": [
            "brightness_mean",
            "brightness_std",
            "brightness_p10",
            "brightness_p90",
            "saturation_mean",
            "saturation_std",
            "value_mean",
            "sharpness",
            "colorfulness",
            "center_brightness",
            "center_contrast"
        ],
        "advanced_food_features": [
            "center_emphasis",
            "focus_contrast",
            "edge_density",
            "foreground_ratio",
            "empty_space_ratio",
            "warm_ratio",
            "hue_std",
            "sat_balance",
            "analogous_score",
            "complementary_score",
            "symmetry_score",
            "thirds_emphasis",
            "color_separation"
        ],
        "pair_feature_mode": "cnn_abs_diff + hc_diff + hc_abs_diff + one_hot_category",
        "label_definition": {
            "1": "Image 1 wins",
            "0": "Image 2 wins"
        },
        "symmetry_augmentation": True,
        "cross_validation": {
            "n_splits": N_SPLITS,
            "scores": cv_result["scores"],
            "mean_accuracy": cv_result["mean_accuracy"],
            "std_accuracy": cv_result["std_accuracy"],
        },
        "best_xgboost_config": {
            "n_estimators": 450,
            "max_depth": 4,
            "learning_rate": 0.022,
            "subsample": 0.82,
            "colsample_bytree": 0.70,
            "reg_lambda": 3.5,
            "min_child_weight": 3,
            "gamma": 0.20,
            "random_state": 52,
        },
        "classifier": "xgboost",
        "predict_mode": "bidirectional_probability"
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Training finished successfully.")
    print(f"[INFO] Saved feature extractor to: {FEATURE_EXTRACTOR_PATH}")
    print(f"[INFO] Saved classifier to: {CLASSIFIER_PATH}")
    print(f"[INFO] Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()