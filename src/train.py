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

QUESTIONNAIRE_CSV_PATH = DATA_DIR / "data_from_questionaire.csv"
INSTAGRAM_CSV_PATH = DATA_DIR / "data_from_intragram.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_EXTRACTOR_PATH = MODELS_DIR / "feature_extractor.keras"
CLASSIFIER_PATH = MODELS_DIR / "pairwise_classifier.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"


# =========================
# SETTINGS
# =========================
IMG_SIZE = (224, 224)
BASE_RANDOM_STATE = 42
N_SPLITS = 5

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_MENUS = {"sushi", "ramen", "pizza", "burger", "dessert"}

MENU_TO_FOLDER = {
    "sushi": "sushi",
    "ramen": "ramen",
    "pizza": "pizza",
    "burger": "burger",
    "hamburger": "burger",
    "dessert": "dessert",
}

# ---- Conservative pseudo-label settings ----
ANCHORS_PER_MENU = 16
MAX_PSEUDO_PAIRS_PER_MENU = 500
PSEUDO_MIN_SCORE_GAP = 0.20
PSEUDO_BASE_WEIGHT = 0.18
MAX_RANDOM_PAIR_TRIALS_FACTOR = 18

TRUE_PAIR_WEIGHT_MULTIPLIER = {
    "questionnaire": 1.25,
    "instagram": 0.85,
}

# optional: only use pseudo from menus that have enough images
MIN_IMAGES_PER_MENU_FOR_PSEUDO = 50


# =========================
# MULTI-EXPERIMENT SEARCH
# =========================
EXPERIMENTS = [
    {
        "name": "exp01",
        "seed": 42,
        "xgb_params": {
            "n_estimators": 320,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.82,
            "colsample_bytree": 0.72,
            "reg_lambda": 4.0,
            "reg_alpha": 0.2,
            "min_child_weight": 4,
            "gamma": 0.25,
        },
    },
    {
        "name": "exp02",
        "seed": 77,
        "xgb_params": {
            "n_estimators": 420,
            "max_depth": 4,
            "learning_rate": 0.025,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_lambda": 5.0,
            "reg_alpha": 0.3,
            "min_child_weight": 4,
            "gamma": 0.20,
        },
    },
    {
        "name": "exp03",
        "seed": 123,
        "xgb_params": {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.80,
            "colsample_bytree": 0.70,
            "reg_lambda": 6.0,
            "reg_alpha": 0.4,
            "min_child_weight": 5,
            "gamma": 0.30,
        },
    },
    {
        "name": "exp04",
        "seed": 999,
        "xgb_params": {
            "n_estimators": 280,
            "max_depth": 3,
            "learning_rate": 0.04,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "reg_lambda": 3.0,
            "reg_alpha": 0.15,
            "min_child_weight": 3,
            "gamma": 0.15,
        },
    },
]


# =========================
# GENERAL UTILS
# =========================
def normalize_menu_name(menu_name: str) -> str:
    menu = str(menu_name).strip().lower()
    return MENU_TO_FOLDER.get(menu, menu)


def get_menu_from_path(image_path: str) -> str:
    p = Path(image_path)
    folder = p.parent.name.strip().lower()
    return normalize_menu_name(folder)


def build_image_index(root_dir: Path) -> dict:
    image_index = {}
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            key = p.name.lower()
            image_index.setdefault(key, []).append(str(p))
    return image_index


def build_menu_to_images(root_dir: Path) -> dict:
    menu_to_images = {menu: [] for menu in VALID_MENUS}

    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            menu = get_menu_from_path(str(p))
            if menu in VALID_MENUS:
                menu_to_images[menu].append(str(p))

    for menu in menu_to_images:
        menu_to_images[menu] = sorted(list(set(menu_to_images[menu])))

    return menu_to_images


def resolve_image_path(image_name: str, menu_name: str, image_index: dict) -> str:
    image_name = str(image_name).strip()
    image_key = image_name.lower()
    expected_menu = normalize_menu_name(menu_name)

    if expected_menu not in VALID_MENUS:
        raise ValueError(f"Unknown menu '{menu_name}' -> normalized '{expected_menu}'")

    candidate = DATA_DIR / expected_menu / image_name
    if candidate.exists():
        actual_menu = get_menu_from_path(str(candidate))
        if actual_menu != expected_menu:
            raise ValueError(
                f"Category mismatch: file={image_name}, expected={expected_menu}, actual={actual_menu}"
            )
        return str(candidate)

    if image_key not in image_index:
        raise FileNotFoundError(f"Cannot find image: {image_name} (menu={menu_name})")

    matches = image_index[image_key]
    same_menu_matches = [p for p in matches if get_menu_from_path(p) == expected_menu]

    if len(same_menu_matches) == 1:
        return same_menu_matches[0]

    if len(same_menu_matches) > 1:
        raise ValueError(
            f"Ambiguous image name '{image_name}' in menu '{expected_menu}'. Matches: {same_menu_matches}"
        )

    found_menus = sorted({get_menu_from_path(p) for p in matches})
    raise ValueError(
        f"Image '{image_name}' found, but not in expected menu '{expected_menu}'. Found in: {found_menus}"
    )


def validate_pair_category(image1_path: str, image2_path: str, csv_menu: str, row_idx=None):
    menu_csv = normalize_menu_name(csv_menu)
    menu1 = get_menu_from_path(image1_path)
    menu2 = get_menu_from_path(image2_path)

    prefix = f"[row {row_idx}] " if row_idx is not None else ""

    if menu1 != menu2:
        raise ValueError(
            f"{prefix}Pair category mismatch: "
            f"image1={Path(image1_path).name} -> {menu1}, "
            f"image2={Path(image2_path).name} -> {menu2}"
        )

    if menu1 != menu_csv:
        raise ValueError(
            f"{prefix}CSV menu mismatch: "
            f"csv_menu={menu_csv}, image1_menu={menu1}, image2_menu={menu2}"
        )


def load_pair_csv(csv_path: Path, source_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[WARN] CSV not found, skipping: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    required_cols = ["Image 1", "Image 2", "Menu", "Winner"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] CSV missing columns {missing}, skipping: {csv_path}")
        return pd.DataFrame()

    df = df.copy()
    df["Source"] = source_name
    df = df[df["Winner"].isin([1, 2])].reset_index(drop=True)

    print(f"[INFO] Loaded {len(df)} rows from {csv_path.name} (source={source_name})")
    return df


def compute_vote_margin_weight(row: pd.Series) -> float:
    if "Num Vote 1" in row and "Num Vote 2" in row:
        try:
            v1 = float(row["Num Vote 1"])
            v2 = float(row["Num Vote 2"])
            total = max(v1 + v2, 1.0)
            margin = abs(v1 - v2) / total

            if margin < 0.03:
                return 0.25
            elif margin < 0.07:
                return 0.55
            elif margin < 0.15:
                return 1.00
            elif margin < 0.25:
                return 1.35
            else:
                return 1.70
        except Exception:
            return 1.0

    return 1.0


def build_true_pair_records(df_all: pd.DataFrame, image_index: dict) -> list:
    records = []
    bad_rows = []

    for i, row in df_all.iterrows():
        try:
            p1 = resolve_image_path(row["Image 1"], row["Menu"], image_index)
            p2 = resolve_image_path(row["Image 2"], row["Menu"], image_index)

            validate_pair_category(p1, p2, row["Menu"], row_idx=i)

            source_name = str(row.get("Source", "unknown")).strip().lower()
            base_weight = TRUE_PAIR_WEIGHT_MULTIPLIER.get(source_name, 1.0)
            vote_weight = compute_vote_margin_weight(row)

            records.append({
                "image1_path": p1,
                "image2_path": p2,
                "menu": normalize_menu_name(row["Menu"]),
                "winner": int(row["Winner"]),
                "weight": float(base_weight * vote_weight),
                "source": source_name,
                "is_pseudo": False,
            })
        except Exception as e:
            bad_rows.append((i, str(e)))

    if bad_rows:
        print("\n[ERROR] Found invalid labeled pairs:")
        for row_id, err in bad_rows[:20]:
            print(f"  row={row_id}: {err}")
        raise ValueError(f"Found {len(bad_rows)} invalid rows. Fix CSV / file placement first.")

    return records


def collect_paths_from_records(records: list) -> set:
    s = set()
    for r in records:
        s.add(r["image1_path"])
        s.add(r["image2_path"])
    return s


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

    for idx, p in enumerate(unique_paths, start=1):
        img = load_img_for_cnn(p)
        batch_imgs.append(img)
        batch_paths.append(p)

        if len(batch_imgs) >= batch_size:
            flush_batch(batch_imgs, batch_paths)
            batch_imgs = []
            batch_paths = []

        if idx % 500 == 0:
            print(f"[INFO]   CNN progress: {idx}/{len(unique_paths)}")

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
# HANDCRAFTED FEATURES
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
        (hue < 0.12)
        | ((hue > 0.90) & (hue <= 1.0))
        | ((hue > 0.12) & (hue < 0.18))
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

    for idx, p in enumerate(unique_paths, start=1):
        basic = compute_handcrafted_features(p)
        advanced = compute_advanced_food_features(p)
        cache[p] = np.concatenate([basic, advanced], axis=0).astype(np.float32)

        if idx % 500 == 0:
            print(f"[INFO]   HC progress: {idx}/{len(unique_paths)}")

    return cache


# =========================
# PAIR FEATURE
# =========================
def get_category_one_hot(menu_name: str) -> np.ndarray:
    categories = ["sushi", "ramen", "pizza", "burger", "dessert"]
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
    cnn_diff = cnn1 - cnn2
    cnn_abs_diff = np.abs(cnn_diff)

    hc_diff = hc1 - hc2
    hc_abs_diff = np.abs(hc_diff)

    hc_mean = (hc1 + hc2) * 0.5

    one_hot = get_category_one_hot(menu_name)

    feat = np.concatenate([
        cnn_abs_diff,
        cnn_diff[:64],
        hc_diff,
        hc_abs_diff,
        hc_mean,
        one_hot,
    ], axis=0)

    return feat.astype(np.float32)


def records_to_dataset(records, cnn_cache, hc_cache):
    X = []
    y = []
    sample_weights = []

    for rec in records:
        p1 = rec["image1_path"]
        p2 = rec["image2_path"]
        menu = rec["menu"]
        winner = int(rec["winner"])
        w = float(rec["weight"])

        cnn1 = cnn_cache[p1]
        cnn2 = cnn_cache[p2]
        hc1 = hc_cache[p1]
        hc2 = hc_cache[p2]

        X.append(build_pair_feature(cnn1, cnn2, hc1, hc2, menu))
        y.append(1 if winner == 1 else 0)
        sample_weights.append(w)

        X.append(build_pair_feature(cnn2, cnn1, hc2, hc1, menu))
        y.append(0 if winner == 1 else 1)
        sample_weights.append(w)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    return X, y, sample_weights


# =========================
# MODEL
# =========================
def build_model(random_state: int, xgb_params: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            **xgb_params,
        ))
    ])


def fit_model_with_sample_weight(model, X, y, sample_weight):
    step_name = model.steps[-1][0]
    param_name = f"{step_name}__sample_weight"
    model.fit(X, y, **{param_name: sample_weight})
    return model


# =========================
# TRUE-ONLY CV
# =========================
def cross_validate_true_only(records, cnn_cache, hc_cache, random_state: int, xgb_params: dict):
    if not records:
        raise ValueError("No true labeled records for cross-validation.")

    y_base = np.array([1 if int(r["winner"]) == 1 else 0 for r in records], dtype=np.int32)
    idxs = np.arange(len(records))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    fold_scores = []

    print("[INFO] Cross-validating on TRUE pairs only...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(idxs, y_base), start=1):
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        X_train, y_train, w_train = records_to_dataset(train_records, cnn_cache, hc_cache)
        X_val, y_val, _ = records_to_dataset(val_records, cnn_cache, hc_cache)

        model = build_model(random_state=random_state, xgb_params=xgb_params)
        fit_model_with_sample_weight(model, X_train, y_train, w_train)

        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        fold_scores.append(acc)

        print(f"[INFO]   Fold {fold_idx}/{N_SPLITS} TRUE-only acc = {acc:.4f}")

    return {
        "scores": fold_scores,
        "mean_accuracy": float(np.mean(fold_scores)),
        "std_accuracy": float(np.std(fold_scores)),
    }


# =========================
# PSEUDO-LABELING
# =========================
def choose_anchor_paths(image_paths, rng, num_anchors):
    image_paths = list(image_paths)
    if len(image_paths) <= num_anchors:
        return image_paths
    idxs = rng.choice(len(image_paths), size=num_anchors, replace=False)
    return [image_paths[i] for i in idxs]


def score_all_images_with_seed_model(menu_name, image_paths, seed_model, cnn_cache, hc_cache, rng):
    if len(image_paths) == 0:
        return {}

    anchors = choose_anchor_paths(image_paths, rng, ANCHORS_PER_MENU)
    scores = {}

    print(f"[INFO] Scoring images for menu='{menu_name}' using {len(anchors)} anchors...")

    for idx, img_path in enumerate(image_paths, start=1):
        feats = []

        for anchor_path in anchors:
            if anchor_path == img_path:
                continue

            cnn1 = cnn_cache[img_path]
            cnn2 = cnn_cache[anchor_path]
            hc1 = hc_cache[img_path]
            hc2 = hc_cache[anchor_path]

            feats.append(build_pair_feature(cnn1, cnn2, hc1, hc2, menu_name))

        if len(feats) == 0:
            scores[img_path] = 0.5
        else:
            X = np.asarray(feats, dtype=np.float32)
            probs = seed_model.predict_proba(X)[:, 1]
            scores[img_path] = float(np.mean(probs))

        if idx % 500 == 0:
            print(f"[INFO]   scoring progress {menu_name}: {idx}/{len(image_paths)}")

    return scores


def pseudo_weight_from_gap(gap: float) -> float:
    if gap >= 0.40:
        return PSEUDO_BASE_WEIGHT * 1.60
    elif gap >= 0.32:
        return PSEUDO_BASE_WEIGHT * 1.35
    elif gap >= 0.26:
        return PSEUDO_BASE_WEIGHT * 1.15
    return PSEUDO_BASE_WEIGHT


def generate_pseudo_pairs_for_menu(menu_name, image_paths, image_scores, rng):
    pseudo_records = []

    if len(image_paths) < 2:
        return pseudo_records

    if len(image_paths) < MIN_IMAGES_PER_MENU_FOR_PSEUDO:
        print(f"[INFO] Skip pseudo for menu='{menu_name}' because images too few ({len(image_paths)})")
        return pseudo_records

    n_target = min(MAX_PSEUDO_PAIRS_PER_MENU, max(120, len(image_paths) // 10))
    max_trials = n_target * MAX_RANDOM_PAIR_TRIALS_FACTOR

    seen = set()
    trials = 0

    while len(pseudo_records) < n_target and trials < max_trials:
        i1, i2 = rng.choice(len(image_paths), size=2, replace=False)
        p1 = image_paths[i1]
        p2 = image_paths[i2]
        trials += 1

        key = tuple(sorted((p1, p2)))
        if key in seen:
            continue
        seen.add(key)

        s1 = image_scores[p1]
        s2 = image_scores[p2]
        gap = abs(s1 - s2)

        if gap < PSEUDO_MIN_SCORE_GAP:
            continue

        winner = 1 if s1 >= s2 else 2
        pseudo_w = pseudo_weight_from_gap(gap)

        pseudo_records.append({
            "image1_path": p1,
            "image2_path": p2,
            "menu": menu_name,
            "winner": winner,
            "weight": float(pseudo_w),
            "source": "pseudo",
            "is_pseudo": True,
            "score_gap": float(gap),
        })

    print(
        f"[INFO] Generated {len(pseudo_records)} pseudo pairs for menu='{menu_name}' "
        f"(trials={trials}, target={n_target})"
    )
    return pseudo_records


def build_fold_safe_menu_to_images(menu_to_images: dict, forbidden_paths: set) -> dict:
    out = {}
    for menu, paths in menu_to_images.items():
        out[menu] = [p for p in paths if p not in forbidden_paths]
    return out


def generate_pseudo_records_from_seed_model(seed_model, menu_to_images_safe, cnn_cache, hc_cache, rng):
    pseudo_records = []

    for menu in sorted(menu_to_images_safe.keys()):
        image_paths = menu_to_images_safe[menu]
        if len(image_paths) < 2:
            continue

        image_scores = score_all_images_with_seed_model(
            menu_name=menu,
            image_paths=image_paths,
            seed_model=seed_model,
            cnn_cache=cnn_cache,
            hc_cache=hc_cache,
            rng=rng,
        )

        menu_pseudo = generate_pseudo_pairs_for_menu(
            menu_name=menu,
            image_paths=image_paths,
            image_scores=image_scores,
            rng=rng,
        )
        pseudo_records.extend(menu_pseudo)

    return pseudo_records


# =========================
# PSEUDO-ASSISTED CV
# =========================
def cross_validate_with_pseudo(true_records, menu_to_images, cnn_cache, hc_cache, random_state: int, xgb_params: dict):
    if not true_records:
        raise ValueError("No true labeled records for pseudo-assisted CV.")

    y_base = np.array([1 if int(r["winner"]) == 1 else 0 for r in true_records], dtype=np.int32)
    idxs = np.arange(len(true_records))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    fold_scores = []
    pseudo_counts = []

    print("[INFO] Cross-validating with fold-safe pseudo-labeling...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(idxs, y_base), start=1):
        rng = np.random.default_rng(random_state + fold_idx)

        train_records = [true_records[i] for i in train_idx]
        val_records = [true_records[i] for i in val_idx]

        X_seed, y_seed, w_seed = records_to_dataset(train_records, cnn_cache, hc_cache)
        seed_model = build_model(random_state=random_state, xgb_params=xgb_params)
        fit_model_with_sample_weight(seed_model, X_seed, y_seed, w_seed)

        val_paths = collect_paths_from_records(val_records)
        safe_menu_to_images = build_fold_safe_menu_to_images(menu_to_images, val_paths)

        pseudo_records = generate_pseudo_records_from_seed_model(
            seed_model=seed_model,
            menu_to_images_safe=safe_menu_to_images,
            cnn_cache=cnn_cache,
            hc_cache=hc_cache,
            rng=rng,
        )

        train_plus_pseudo = train_records + pseudo_records

        X_train, y_train, w_train = records_to_dataset(train_plus_pseudo, cnn_cache, hc_cache)
        X_val, y_val, _ = records_to_dataset(val_records, cnn_cache, hc_cache)

        final_model = build_model(random_state=random_state, xgb_params=xgb_params)
        fit_model_with_sample_weight(final_model, X_train, y_train, w_train)

        val_pred = final_model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        fold_scores.append(acc)
        pseudo_counts.append(len(pseudo_records))

        print(
            f"[INFO]   Fold {fold_idx}/{N_SPLITS} pseudo-assisted acc = {acc:.4f} "
            f"(pseudo_pairs={len(pseudo_records)})"
        )

    return {
        "scores": fold_scores,
        "mean_accuracy": float(np.mean(fold_scores)),
        "std_accuracy": float(np.std(fold_scores)),
        "pseudo_pair_counts": pseudo_counts,
        "pseudo_pair_count_mean": float(np.mean(pseudo_counts)),
    }


# =========================
# FINAL TRAIN
# =========================
def train_final_model_with_conservative_pseudo(true_records, menu_to_images, cnn_cache, hc_cache, random_state: int, xgb_params: dict):
    rng = np.random.default_rng(random_state)

    print("[INFO] Training seed model on ALL true pairs...")
    X_seed, y_seed, w_seed = records_to_dataset(true_records, cnn_cache, hc_cache)
    seed_model = build_model(random_state=random_state, xgb_params=xgb_params)
    fit_model_with_sample_weight(seed_model, X_seed, y_seed, w_seed)

    print("[INFO] Generating conservative pseudo pairs from ALL images...")
    true_paths = collect_paths_from_records(true_records)

    menu_to_images_for_pseudo = {
        menu: [p for p in paths if p not in true_paths]
        for menu, paths in menu_to_images.items()
    }

    pseudo_records = generate_pseudo_records_from_seed_model(
        seed_model=seed_model,
        menu_to_images_safe=menu_to_images_for_pseudo,
        cnn_cache=cnn_cache,
        hc_cache=hc_cache,
        rng=rng,
    )

    print(f"[INFO] Total pseudo pairs for FINAL training: {len(pseudo_records)}")

    final_records = true_records + pseudo_records
    X_full, y_full, w_full = records_to_dataset(final_records, cnn_cache, hc_cache)

    print(f"[INFO] X_full shape: {X_full.shape}")
    print(f"[INFO] y_full shape: {y_full.shape}")

    final_model = build_model(random_state=random_state, xgb_params=xgb_params)
    fit_model_with_sample_weight(final_model, X_full, y_full, w_full)

    return final_model, pseudo_records


# =========================
# EXPERIMENT SEARCH
# =========================
def is_better_experiment(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True

    cand_pseudo = candidate["pseudo_cv"]["mean_accuracy"]
    best_pseudo = best["pseudo_cv"]["mean_accuracy"]

    if cand_pseudo > best_pseudo:
        return True
    if cand_pseudo < best_pseudo:
        return False

    cand_true = candidate["baseline_cv"]["mean_accuracy"]
    best_true = best["baseline_cv"]["mean_accuracy"]

    if cand_true > best_true:
        return True
    if cand_true < best_true:
        return False

    cand_pseudo_std = candidate["pseudo_cv"]["std_accuracy"]
    best_pseudo_std = best["pseudo_cv"]["std_accuracy"]

    if cand_pseudo_std < best_pseudo_std:
        return True
    if cand_pseudo_std > best_pseudo_std:
        return False

    return candidate["experiment"]["name"] < best["experiment"]["name"]


def run_experiment(experiment, true_records, menu_to_images, cnn_cache, hc_cache):
    exp_name = experiment["name"]
    exp_seed = int(experiment["seed"])
    xgb_params = experiment["xgb_params"]

    print("\n" + "=" * 80)
    print(f"[INFO] START EXPERIMENT: {exp_name}")
    print(json.dumps(experiment, indent=2, ensure_ascii=False))
    print("=" * 80)

    baseline_cv = cross_validate_true_only(
        records=true_records,
        cnn_cache=cnn_cache,
        hc_cache=hc_cache,
        random_state=exp_seed,
        xgb_params=xgb_params,
    )

    print("\n[RESULT] TRUE-only cross-validation summary")
    print(
        f'  - mean={baseline_cv["mean_accuracy"]:.4f}, '
        f'std={baseline_cv["std_accuracy"]:.4f}, '
        f'scores={[round(s, 4) for s in baseline_cv["scores"]]}'
    )

    pseudo_cv = cross_validate_with_pseudo(
        true_records=true_records,
        menu_to_images=menu_to_images,
        cnn_cache=cnn_cache,
        hc_cache=hc_cache,
        random_state=exp_seed,
        xgb_params=xgb_params,
    )

    print("\n[RESULT] PSEUDO-assisted cross-validation summary")
    print(
        f'  - mean={pseudo_cv["mean_accuracy"]:.4f}, '
        f'std={pseudo_cv["std_accuracy"]:.4f}, '
        f'scores={[round(s, 4) for s in pseudo_cv["scores"]]}'
    )
    print(
        f'  - pseudo pair counts per fold={pseudo_cv["pseudo_pair_counts"]}, '
        f'mean={pseudo_cv["pseudo_pair_count_mean"]:.1f}'
    )

    return {
        "experiment": experiment,
        "baseline_cv": baseline_cv,
        "pseudo_cv": pseudo_cv,
    }


# =========================
# MAIN
# =========================
def main():
    print("[INFO] Loading labeled pair CSVs...")
    df_questionnaire = load_pair_csv(QUESTIONNAIRE_CSV_PATH, "questionnaire")
    df_instagram = load_pair_csv(INSTAGRAM_CSV_PATH, "instagram")

    df_all = pd.concat([df_questionnaire, df_instagram], ignore_index=True)
    if len(df_all) == 0:
        raise ValueError("No valid labeled pair rows found in questionnaire/instagram CSVs.")

    print(f"[INFO] Total labeled pairs: {len(df_all)}")

    print("[INFO] Building image index...")
    image_index = build_image_index(DATA_DIR)

    print("[INFO] Validating labeled pairs...")
    true_records = build_true_pair_records(df_all, image_index)
    print(f"[INFO] Valid labeled pairs after checking: {len(true_records)}")

    print("[INFO] Scanning ALL images in Data/ ...")
    menu_to_images = build_menu_to_images(DATA_DIR)
    for menu in sorted(menu_to_images.keys()):
        print(f"[INFO]   menu={menu:<8} images={len(menu_to_images[menu])}")

    all_image_paths = []
    for menu in sorted(menu_to_images.keys()):
        all_image_paths.extend(menu_to_images[menu])
    all_image_paths = sorted(list(set(all_image_paths)))

    print(f"[INFO] Total ALL images used for feature extraction: {len(all_image_paths)}")

    print("[INFO] Creating feature extractor...")
    feature_extractor = create_feature_extractor()
    feature_extractor.save(FEATURE_EXTRACTOR_PATH)

    cnn_cache = extract_cnn_features_for_unique_images(feature_extractor, all_image_paths)
    hc_cache = extract_handcrafted_features_for_unique_images(all_image_paths)

    # -------------------------
    # MULTI-EXPERIMENT SEARCH
    # -------------------------
    all_results = []
    best_result = None

    for experiment in EXPERIMENTS:
        result = run_experiment(
            experiment=experiment,
            true_records=true_records,
            menu_to_images=menu_to_images,
            cnn_cache=cnn_cache,
            hc_cache=hc_cache,
        )
        all_results.append(result)

        if is_better_experiment(result, best_result):
            best_result = result
            print(f"\n[INFO] Current BEST experiment = {best_result['experiment']['name']}")

    if best_result is None:
        raise RuntimeError("No experiment result found.")

    best_experiment = best_result["experiment"]
    best_seed = int(best_experiment["seed"])
    best_xgb_params = best_experiment["xgb_params"]
    baseline_cv = best_result["baseline_cv"]
    pseudo_cv = best_result["pseudo_cv"]

    print("\n" + "#" * 80)
    print("[RESULT] BEST EXPERIMENT SELECTED")
    print(json.dumps(best_experiment, indent=2, ensure_ascii=False))
    print(
        f"[RESULT] Best TRUE-only CV mean={baseline_cv['mean_accuracy']:.4f}, "
        f"std={baseline_cv['std_accuracy']:.4f}"
    )
    print(
        f"[RESULT] Best PSEUDO-assisted CV mean={pseudo_cv['mean_accuracy']:.4f}, "
        f"std={pseudo_cv['std_accuracy']:.4f}"
    )
    print("#" * 80)

    # -------------------------
    # FINAL TRAIN
    # -------------------------
    print("\n[INFO] Training FINAL model using BEST experiment...")
    final_model, final_pseudo_records = train_final_model_with_conservative_pseudo(
        true_records=true_records,
        menu_to_images=menu_to_images,
        cnn_cache=cnn_cache,
        hc_cache=hc_cache,
        random_state=best_seed,
        xgb_params=best_xgb_params,
    )

    print("[INFO] Evaluating FINAL model on TRUE labeled pairs (sanity check only)...")
    X_true_eval, y_true_eval, _ = records_to_dataset(true_records, cnn_cache, hc_cache)
    true_eval_pred = final_model.predict(X_true_eval)
    true_eval_acc = accuracy_score(y_true_eval, true_eval_pred)

    print(f"\n[RESULT] FINAL model sanity-check accuracy on TRUE labeled pairs = {true_eval_acc:.4f}\n")
    print(classification_report(y_true_eval, true_eval_pred, digits=4))

    print("[INFO] Saving classifier...")
    joblib.dump(final_model, CLASSIFIER_PATH)

    metadata = {
        "img_size": IMG_SIZE,
        "questionnaire_csv_path": str(QUESTIONNAIRE_CSV_PATH),
        "instagram_csv_path": str(INSTAGRAM_CSV_PATH),
        "feature_extractor_path": str(FEATURE_EXTRACTOR_PATH),
        "classifier_path": str(CLASSIFIER_PATH),
        "menu_to_folder": MENU_TO_FOLDER,
        "valid_menus": sorted(list(VALID_MENUS)),
        "all_images_used_count": len(all_image_paths),
        "image_count_per_menu": {k: len(v) for k, v in menu_to_images.items()},
        "true_pair_count": len(true_records),
        "final_pseudo_pair_count": len(final_pseudo_records),

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
        "pair_feature_mode": "cnn_abs_diff + cnn_diff_head64 + hc_diff + hc_abs_diff + hc_mean + one_hot_category",
        "label_definition": {
            "1": "Image 1 wins",
            "0": "Image 2 wins"
        },

        "best_experiment": best_experiment,
        "all_experiment_results": [
            {
                "experiment": r["experiment"],
                "true_only_cross_validation": r["baseline_cv"],
                "pseudo_assisted_cross_validation": r["pseudo_cv"],
            }
            for r in all_results
        ],

        "true_only_cross_validation": {
            "n_splits": N_SPLITS,
            "scores": baseline_cv["scores"],
            "mean_accuracy": baseline_cv["mean_accuracy"],
            "std_accuracy": baseline_cv["std_accuracy"],
        },
        "pseudo_assisted_cross_validation": {
            "n_splits": N_SPLITS,
            "scores": pseudo_cv["scores"],
            "mean_accuracy": pseudo_cv["mean_accuracy"],
            "std_accuracy": pseudo_cv["std_accuracy"],
            "pseudo_pair_counts": pseudo_cv["pseudo_pair_counts"],
            "pseudo_pair_count_mean": pseudo_cv["pseudo_pair_count_mean"],
        },

        "pseudo_labeling": {
            "enabled": True,
            "anchors_per_menu": ANCHORS_PER_MENU,
            "max_pseudo_pairs_per_menu": MAX_PSEUDO_PAIRS_PER_MENU,
            "pseudo_min_score_gap": PSEUDO_MIN_SCORE_GAP,
            "pseudo_base_weight": PSEUDO_BASE_WEIGHT,
            "min_images_per_menu_for_pseudo": MIN_IMAGES_PER_MENU_FOR_PSEUDO,
            "note": "validation-fold images are excluded from pseudo generation during CV",
        },

        "xgboost_config": best_xgb_params,
        "classifier": "xgboost",
        "predict_mode": "bidirectional_probability",
        "strict_same_category_check": True,
        "final_sanity_check_accuracy_on_true_pairs": float(true_eval_acc),
        "base_random_state": BASE_RANDOM_STATE,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Training finished successfully.")
    print(f"[INFO] Best experiment: {best_experiment['name']}")
    print(f"[INFO] Saved feature extractor to: {FEATURE_EXTRACTOR_PATH}")
    print(f"[INFO] Saved classifier to: {CLASSIFIER_PATH}")
    print(f"[INFO] Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()