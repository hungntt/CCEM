import glob
import os
import sys
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 3))


def print_step(step_name):
    print(f"\n{'-' * 80}\n STEP: {step_name}\n{'-' * 80}")


def print_success(msg):
    print(f"   [SUCCESS] {msg}")


def print_warning(msg):
    print(f"   [WARNING] {msg}")


def print_error_and_exit(msg, error=None):
    print(f"   [FAILED] {msg}")
    if error is not None:
        print(f"\n SYSTEM ERROR DETAILS:\n{error}")
    sys.exit(1)


def min_max_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.min(x)
    return x / (np.max(x) + 1e-8)


def map_messidor_label(dr_str):
    dr_str = str(dr_str).strip().upper()

    for i in range(5):
        if str(i) in dr_str:
            return i

    return 0


def predict_grade(raw_value, coef):
    if raw_value < coef[0]:
        return 0
    if raw_value < coef[1]:
        return 1
    if raw_value < coef[2]:
        return 2
    if raw_value < coef[3]:
        return 3
    return 4


def crop_image_and_mask(img, mask, tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask_bool = gray_img > tol
    coords = np.argwhere(mask_bool)

    if coords.size == 0:
        return img, mask

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def get_exact_image_path(base_name, search_dir):
    for root, _, files in os.walk(search_dir):
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            filename = base_name + ext
            if filename in files:
                return os.path.join(root, filename)

    return None


def get_true_labels_dict(maples_dir):
    true_labels = {}

    csv_files = glob.glob(
        os.path.join(maples_dir, "**", "diagnosis.csv"),
        recursive=True,
    )

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            true_labels[str(row["name"])] = map_messidor_label(row["DR"])

    return true_labels


def calculate_localization_metrics(heatmap, mask_binary):
    total_energy = np.sum(heatmap) + 1e-8
    energy_score = np.sum(heatmap * mask_binary) / total_energy

    try:
        auc_score = roc_auc_score(mask_binary.flatten(), heatmap.flatten())
    except ValueError:
        auc_score = 0.5

    heatmap_binary = (heatmap > 0.5).astype(np.float32)
    intersection = np.sum(heatmap_binary * mask_binary)
    union = np.sum(heatmap_binary) + np.sum(mask_binary) - intersection
    iou_score = intersection / (union + 1e-8)

    return energy_score, auc_score, iou_score


def load_state_dict_safely(model, weight_path, device, use_cbam):
    state_dict = torch.load(weight_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    cleaned = OrderedDict()

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    has_cbam_weights = any(k.startswith("cbam.") for k in cleaned.keys())

    if has_cbam_weights and not use_cbam:
        raise RuntimeError(
            "This checkpoint contains CBAM weights, but use_cbam=False.\n"
            "Fix: add --use_cbam to the command."
        )

    model.load_state_dict(cleaned, strict=True)

    return model


def collect_maples_base_names(maples_dir, lesion_types):
    base_names = set()

    for lesion in lesion_types:
        mask_files = glob.glob(
            os.path.join(maples_dir, "**", lesion, "*.png"),
            recursive=True,
        )

        for mask_path in mask_files:
            base_names.add(os.path.basename(mask_path).replace(".png", ""))

    return sorted(base_names)


def load_maples_master_mask(base_name, maples_dir, image_shape, lesion_types):
    master_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for lesion in lesion_types:
        lesion_paths = [
            os.path.join(maples_dir, "train", lesion, f"{base_name}.png"),
            os.path.join(maples_dir, "test", lesion, f"{base_name}.png"),
        ]

        lesion_path = next(
            (path for path in lesion_paths if os.path.exists(path)),
            None,
        )

        if lesion_path is None:
            continue

        lesion_mask = cv2.imread(lesion_path, cv2.IMREAD_GRAYSCALE)

        if lesion_mask is None:
            continue

        if lesion_mask.shape[:2] != master_mask.shape[:2]:
            lesion_mask = cv2.resize(
                lesion_mask,
                (master_mask.shape[1], master_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        master_mask = cv2.bitwise_or(master_mask, lesion_mask)

    return master_mask


def get_fundus_mask(image_rgb, tol=8, erode_size=21):
    if image_rgb.max() <= 1.0:
        img = (image_rgb * 255).astype(np.uint8)
    else:
        img = image_rgb.astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > tol).astype(np.uint8)

    kernel_close = np.ones((31, 31), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_erode = np.ones((erode_size, erode_size), np.uint8)
    mask = cv2.erode(mask, kernel_erode, iterations=1)

    return mask.astype(np.float32)


def detect_optic_disc_mask(image_rgb, valid_mask=None, dilate_size=45):
    if image_rgb.max() <= 1.0:
        img = (image_rgb * 255).astype(np.uint8)
    else:
        img = image_rgb.astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]

    if valid_mask is None:
        valid_mask = get_fundus_mask(img)

    valid_mask = (valid_mask > 0).astype(np.uint8)
    values = l_channel[valid_mask > 0]

    if values.size == 0:
        return np.zeros(l_channel.shape, dtype=np.float32)

    threshold = np.percentile(values, 97.5)
    bright = ((l_channel >= threshold) & (valid_mask > 0)).astype(np.uint8)

    kernel = np.ones((15, 15), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright,
        connectivity=8,
    )

    if num_labels <= 1:
        return np.zeros(l_channel.shape, dtype=np.float32)

    h, w = l_channel.shape
    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        if area < 50:
            continue

        aspect = bw / (bh + 1e-8)

        if aspect < 0.35 or aspect > 3.0:
            continue

        cx, cy = centroids[label]
        center_distance = abs(cx - w / 2) / w
        score = area * (1.0 + center_distance)

        if score > best_score:
            best_score = score
            best_label = label

    od_mask = np.zeros_like(l_channel, dtype=np.uint8)

    if best_label is not None:
        od_mask[labels == best_label] = 1
        kernel_dilate = np.ones((dilate_size, dilate_size), np.uint8)
        od_mask = cv2.dilate(od_mask, kernel_dilate, iterations=1)

    return od_mask.astype(np.float32)


def lesion_candidate_prior(image_rgb, valid_mask=None):
    if image_rgb.max() <= 1.0:
        img = (image_rgb * 255).astype(np.uint8)
    else:
        img = image_rgb.astype(np.uint8)

    green = img[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green = clahe.apply(green)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

    bright_response = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel_mid)
    dark_response = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel_small)

    bright_response = min_max_normalize(bright_response)
    dark_response = min_max_normalize(dark_response)

    prior = 0.45 * bright_response + 0.55 * dark_response
    prior = cv2.GaussianBlur(prior, (0, 0), 1.0)

    if valid_mask is not None:
        prior = prior * (valid_mask > 0).astype(np.float32)

    return min_max_normalize(prior).astype(np.float32)


def _is_usable_heatmap(heatmap, valid_mask=None, eps=1e-8):
    hm = np.asarray(heatmap, dtype=np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.maximum(hm, 0.0)

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=np.float32)
        if mask.shape != hm.shape:
            mask = cv2.resize(
                mask,
                (hm.shape[1], hm.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        hm = hm * (mask > 0).astype(np.float32)

    return bool(hm.size and float(np.max(hm)) > eps and float(np.sum(hm)) > eps)


def _safe_stage(candidate, fallback, valid_mask=None, eps=1e-8):
    if _is_usable_heatmap(candidate, valid_mask=valid_mask, eps=eps):
        return candidate.astype(np.float32)
    return fallback.astype(np.float32)


def remove_bad_components(
    heatmap,
    min_area=4,
    max_area=2200,
    max_aspect_ratio=7.0,
    threshold=0.20,
):
    hm = min_max_normalize(heatmap)

    if not _is_usable_heatmap(hm):
        return hm.astype(np.float32)

    binary = (hm >= threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    keep = np.zeros_like(hm, dtype=np.float32)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if area < min_area:
            continue

        if area > max_area:
            continue

        aspect_ratio = max(w / max(h, 1), h / max(w, 1))

        if aspect_ratio > max_aspect_ratio:
            continue

        keep[labels == label] = 1.0

    filtered = hm * keep

    if not _is_usable_heatmap(filtered):
        return hm.astype(np.float32)

    return min_max_normalize(filtered).astype(np.float32)


def final_compact_xai_map(
    raw_heatmap,
    image_rgb,
    keep_percentile=94.0,
    gamma=2.5,
    blur_sigma=0.6,
    use_lesion_prior=True,
    suppress_optic_disc=True,
):
    valid_mask = get_fundus_mask(image_rgb, erode_size=21)

    raw = min_max_normalize(raw_heatmap)

    if not _is_usable_heatmap(raw):
        return raw.astype(np.float32)

    hm = _safe_stage(raw * valid_mask, raw, valid_mask=None)
    last_good = hm.copy()

    if suppress_optic_disc:
        od_mask = detect_optic_disc_mask(image_rgb, valid_mask=valid_mask)
        candidate = hm * (1.0 - od_mask)
        hm = _safe_stage(candidate, last_good, valid_mask=valid_mask)
        last_good = hm.copy()

    if use_lesion_prior:
        prior = lesion_candidate_prior(image_rgb, valid_mask=valid_mask)
        candidate = hm * (0.20 + 0.80 * prior)
        hm = _safe_stage(candidate, last_good, valid_mask=valid_mask)
        last_good = hm.copy()

    values = hm[valid_mask > 0]
    positive_values = values[values > 1e-8]
    threshold_values = positive_values if positive_values.size > 0 else values

    if threshold_values.size > 0:
        thresholded = None

        for pct in [keep_percentile, 90.0, 85.0, 80.0, 70.0]:
            pct = float(np.clip(pct, 0.0, 100.0))
            threshold = np.percentile(threshold_values, pct)
            candidate = np.where(hm >= threshold, hm, 0.0).astype(np.float32)

            if _is_usable_heatmap(candidate, valid_mask=valid_mask):
                thresholded = candidate
                break

        if thresholded is not None:
            hm = thresholded
            last_good = hm.copy()

    candidate = hm ** gamma
    hm = _safe_stage(candidate, last_good, valid_mask=valid_mask)
    last_good = hm.copy()

    if blur_sigma is not None and blur_sigma > 0:
        candidate = cv2.GaussianBlur(hm, (0, 0), blur_sigma)
        hm = _safe_stage(candidate, last_good, valid_mask=valid_mask)
        last_good = hm.copy()

    candidate = hm * valid_mask
    hm = _safe_stage(candidate, last_good, valid_mask=None)
    last_good = hm.copy()

    candidate = remove_bad_components(
        hm,
        min_area=4,
        max_area=2200,
        max_aspect_ratio=7.0,
        threshold=0.20,
    )
    hm = _safe_stage(candidate, last_good, valid_mask=None)

    return min_max_normalize(hm).astype(np.float32)


def make_compact_sample_figure(
    save_path,
    vis_img_float,
    mask_binary,
    heatmap,
    overlay,
    true_class,
    pred_class,
    method_name,
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(vis_img_float)
    axes[0].set_title(f"Orig | True: {true_class}, Pred: {pred_class}")
    axes[0].axis("off")

    mask_overlay = np.zeros_like(vis_img_float)
    mask_overlay[:, :, 1] = mask_binary

    gt_vis = cv2.addWeighted(
        vis_img_float.astype(np.float32),
        0.7,
        mask_overlay.astype(np.float32),
        0.5,
        0,
    )

    axes[1].imshow(gt_vis)
    axes[1].set_title("Expert Master Mask")
    axes[1].axis("off")

    axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title(f"{method_name} Compact")
    axes[2].axis("off")

    axes[3].imshow(vis_img_float)

    masked_heatmap = np.ma.masked_where(heatmap <= 1e-6, heatmap)

    axes[3].imshow(
        masked_heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=0.60,
        interpolation="bilinear",
    )

    axes[3].set_title(f"{method_name} Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()