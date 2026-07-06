import os
import sys
import glob
import json
import itertools
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

# Path routing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XAI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))        # src/xai
SRC_DIR = os.path.abspath(os.path.join(XAI_DIR, ".."))            # src
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))       # repo root
TRAINING_DIR = os.path.join(SRC_DIR, "training")
for _p in (XAI_DIR, TRAINING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import APTOSModel
from augmentation import get_valid_transforms
from preprocessing import apply_clahe
from pytorch_grad_cam.utils.image import show_cam_on_image
from explanation.maple_utils import get_fundus_mask

# IMPORT FROM CORE
import CCEM.ccem_core as ccem_core
from CCEM.ccem_core import (
    ensure_2d_float_map,
    generate_ccem,
    heatmap_for_overlay,
    calculate_advanced_metrics,
    calculate_extended_metrics,
)

# ==========================================
# SYSTEM LOGGER & FILE UTILS
# ==========================================
def print_step(step_name):
    print(f"\n{'-'*80}\n STEP: {step_name}\n{'-'*80}")

def print_success(msg):
    print(f"   [SUCCESS] {msg}")

def print_error_and_exit(msg, error=""):
    print(f"   [FAILED] {msg}\n SYSTEM ERROR DETAILS: {error}")
    sys.exit(1)

def print_warning(msg):
    print(f"   [WARNING] {msg}")

def map_messidor_label(dr_str):
    dr_str = str(dr_str).strip().upper()
    for i in range(5):
        if str(i) in dr_str: return i
    return 0

def crop_image_and_mask(img, mask, tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask_bool = gray_img > tol
    coords = np.argwhere(mask_bool)
    if coords.size == 0: return img, mask
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

def get_exact_image_path(base_name, search_dir):
    for root, dirs, files in os.walk(search_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            if base_name + ext in files:
                return os.path.join(root, base_name + ext)
    return None

def find_method_npy(base_dir, img_id, method_keywords):
    search_pattern = os.path.join(base_dir, "**", f"*{img_id}*.npy")
    possible_files = glob.glob(search_pattern, recursive=True)
    for f in possible_files:
        f_lower = os.path.basename(f).lower()
        if any(kw.lower() in f_lower for kw in method_keywords):
            return f
    return None

def extract_image_id_from_xai_npy(npy_path):
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    known_suffixes = (
        "_GradCAMpp_compact",
        "_AdaSISE_compact",
        "_IGSG_compact",
        "_SmoothIG_compact",
        "_Smooth-IG_compact",
        "_GradCAMpp",
        "_AdaSISE",
        "_IGSG",
        "_SmoothIG",
        "_Smooth-IG",
    )
    stem_lower = stem.lower()
    for suffix in known_suffixes:
        if stem_lower.endswith(suffix.lower()):
            return stem[: -len(suffix)]
    return None

def collect_image_ids_from_xai_dir(xai_dir):
    image_ids = set()
    csv_files = sorted(glob.glob(os.path.join(xai_dir, "**", "*_results.csv"), recursive=True))

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print_warning(f"Could not read CSV {csv_path}: {exc}")
            continue

        if "Image_ID" not in df.columns:
            print_warning(f"Skipping CSV without Image_ID column: {csv_path}")
            continue

        image_ids.update(df["Image_ID"].dropna().astype(str))

    if image_ids:
        print_success(f"Image IDs extracted from {len(csv_files)} result CSV file(s).")
        return image_ids

    npy_files = glob.glob(os.path.join(xai_dir, "**", "*.npy"), recursive=True)
    for npy_path in npy_files:
        img_id = extract_image_id_from_xai_npy(npy_path)
        if img_id is not None:
            image_ids.add(img_id)

    if image_ids:
        print_warning("No result CSV files found; extracted image IDs from XAI NPY filenames instead.")

    return image_ids

def resolve_existing_dir(name, candidates):
    for path in candidates:
        if path and os.path.isdir(path):
            return path

    print_error_and_exit(
        f"{name} directory not found. Pass the correct path on the command line.",
        "Checked: " + ", ".join(str(p) for p in candidates if p),
    )

def resize_heatmap_to_shape(heatmap, target_shape):
    heatmap = ensure_2d_float_map(heatmap)
    if heatmap.shape[:2] == target_shape[:2]:
        return heatmap
    return cv2.resize(
        heatmap,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

def show_percentile_overlay(image_rgb_float, heatmap):
    clipped = heatmap_for_overlay(heatmap)
    clipped = resize_heatmap_to_shape(clipped, image_rgb_float.shape[:2])
    return show_cam_on_image(
        image_rgb_float,
        clipped,
        use_rgb=True,
        colormap=cv2.COLORMAP_JET,
    )

def safe_nanmean(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))

def safe_nanstd(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanstd(values))

def get_heatmap_stats(heatmap):
    hm = np.asarray(heatmap, dtype=np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "min": float(np.min(hm)) if hm.size else 0.0,
        "max": float(np.max(hm)) if hm.size else 0.0,
        "sum": float(np.sum(hm)) if hm.size else 0.0,
        "nonzero_ratio": float(np.mean(np.abs(hm) > 1e-8)) if hm.size else 0.0,
    }

def build_ccem_grid():
    grid = []
    for w_ig, w_adasise in itertools.product(
        [0.50, 0.60, 0.70, 0.80],
        [0.10, 0.20, 0.30, 0.40],
    ):
        w_gradcam = 1.0 - w_ig - w_adasise
        if w_gradcam < -1e-8:
            continue
        w_gradcam = max(0.0, w_gradcam)
        for ig_gate_threshold, threshold_percentile, top_k_components, blur_sigma in itertools.product(
            [0.25, 0.35, 0.45, 0.55],
            [85, 88, 90, 92],
            [3, 5, 8, 12],
            [0.0, 0.5, 0.8, 1.0],
        ):
            grid.append({
                "w_gradcam": w_gradcam,
                "w_adasise": w_adasise,
                "w_ig": w_ig,
                "ig_gate_threshold": ig_gate_threshold,
                "threshold_percentile": threshold_percentile,
                "top_k_components": top_k_components,
                "blur_sigma": blur_sigma,
            })
    return grid

# ==========================================
# COMMAND LINE PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate CCEM and ODExAI Metrics from NPY files")
    parser.add_argument('--xai_dir', type=str, required=True, help="Path to directory containing NPY files")
    parser.add_argument('--weight_path', type=str, required=True, help="Path to .pth model weights")
    parser.add_argument('--threshold_path', type=str, default=None, help="Path to best_thresholds.npy; defaults to the weight directory")
    parser.add_argument('--messidor_img_dir', type=str, default=None, help="Path to MESSIDOR/MAPLES image directory")
    parser.add_argument('--maples_dir', type=str, default=None, help="Path to MAPLES-DR masks and diagnosis.csv directory")
    parser.add_argument('--model', type=str, default='B7', help="Model version")
    parser.add_argument('--use_cbam', action='store_true', help="Flag for CBAM version")
    parser.add_argument('--max_samples', type=int, default=None, help="Optional maximum number of images to process")
    parser.add_argument(
        '--ccem_mode',
        type=str,
        default='adaptive_reliability',
        choices=['ig_only', 'simple_average', 'soft_agreement', 'ig_anchored', 'adaptive_reliability', 'adaptive', 'adaptive_ccem', 'legacy_soft_union'],
        help="CCEM fusion mode",
    )
    parser.add_argument('--ccem_w_ig', type=float, default=0.80, help="IG/SmoothGrad fusion weight")
    parser.add_argument('--ccem_w_adasise', type=float, default=0.15, help="Ada-SISE fusion weight")
    parser.add_argument('--ccem_w_gradcam', type=float, default=0.05, help="Grad-CAM++ fusion weight")
    parser.add_argument('--ccem_ig_gate_threshold', type=float, default=0.15, help="IG gate saturation threshold")
    parser.add_argument('--ccem_agreement_threshold', type=float, default=0.80, help="Agreement high-confidence threshold")
    parser.add_argument('--ccem_gate_floor', type=float, default=0.80, help="Minimum IG gate multiplier")
    parser.add_argument('--ccem_agreement_bonus', type=float, default=0.05, help="Agreement multiplier strength")
    parser.add_argument('--ccem_blur_sigma', type=float, default=1.0, help="Gaussian blur sigma after gating and masking")
    parser.add_argument('--ccem_threshold_percentile', type=float, default=88, help="Connected-component threshold percentile")
    parser.add_argument('--ccem_top_k_components', type=int, default=8, help="Maximum connected components retained")
    parser.add_argument('--ccem_min_area', type=int, default=20, help="Minimum connected-component area")
    parser.add_argument('--ccem_max_area_ratio', type=float, default=0.10, help="Maximum connected-component area as image ratio")
    parser.add_argument(
        '--ccem_apply_component_filter',
        action='store_true',
        help="Enable CCEM component filtering. Disabled by default because it can erase maps.",
    )
    parser.add_argument("--adaptive_temperature", type=float, default=0.25)
    parser.add_argument("--adaptive_faithfulness_weight", type=float, default=0.35)
    parser.add_argument("--adaptive_agreement_weight", type=float, default=0.20)
    parser.add_argument("--adaptive_compactness_weight", type=float, default=0.20)
    parser.add_argument("--adaptive_containment_weight", type=float, default=0.15)
    parser.add_argument("--adaptive_peak_weight", type=float, default=0.10)
    parser.add_argument("--adaptive_sharpen_gamma", type=float, default=1.5)
    parser.add_argument(
        "--adaptive_soft_keep_percentile",
        type=float,
        default=None,
        help="Optional soft top-percentile filter after fusion. Try 50 or 60. Default: disabled.",
    )
    parser.add_argument(
        "--save_ccem_debug",
        action="store_true",
        help="Save adaptive CCEM reliability scores and fusion weights per image.",
    )
    parser.add_argument('--ccem_grid_search', action='store_true', help="Run optional CCEM validation grid search; use only on validation split inputs")
    parser.add_argument('--metric_steps', type=int, default=20, help="Number of steps for deletion and insertion metrics")
    parser.add_argument('--ccem_best_params_path', type=str, default=None, help="Optional JSON path for best grid-search parameters")
    return parser.parse_args()

# ==========================================
# MAIN RUNNER
# ==========================================
def main():
    args = parse_args()
    
    IMG_SIZE = 600
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LESION_TYPES = ["Microaneurysms", "Hemorrhages", "Exudates", "CottonWoolSpots"]
    if args.ccem_mode in ("adaptive", "adaptive_reliability", "adaptive_ccem"):
        CCEM_METHOD = "CCEM_Adaptive"
    elif args.ccem_mode == "ig_anchored":
        CCEM_METHOD = "CCEM"
    elif args.ccem_mode == "simple_average":
        CCEM_METHOD = "CCEM_SimpleAvg"
    elif args.ccem_mode == "soft_agreement":
        CCEM_METHOD = "CCEM_SoftAgreement"
    elif args.ccem_mode == "ig_only":
        CCEM_METHOD = "CCEM_IGOnly"
    else:
        CCEM_METHOD = "CCEM"
    
    MESSIDOR_IMG_DIR = resolve_existing_dir(
        "MESSIDOR image",
        [
            args.messidor_img_dir,
            os.path.join(PROJECT_ROOT, "datasets", "datasets1"),
            os.path.join(PROJECT_ROOT, "datasets1"),
        ],
    )
    MAPLES_DIR = resolve_existing_dir(
        "MAPLES-DR",
        [
            args.maples_dir,
            os.path.join(PROJECT_ROOT, "datasets", "MAPLES-DR"),
            os.path.join(PROJECT_ROOT, "MAPLES-DR"),
        ],
    )
    
    OUTPUT_DIR = os.path.join(args.xai_dir, "CCEM_Evaluation_Results")
    os.makedirs(os.path.join(OUTPUT_DIR, "visuals"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "npy"), exist_ok=True)

    print_step("CCEM & METRICS EVALUATION INITIALIZATION")
    print_success(f"MESSIDOR image dir: {MESSIDOR_IMG_DIR}")
    print_success(f"MAPLES-DR dir: {MAPLES_DIR}")
    print_success(f"CCEM mode: {args.ccem_mode}")
    if args.ccem_grid_search:
        print_warning("CCEM grid search is enabled. Use this only with a validation split, not a held-out test split.")
    
    # Load Thresholds
    weight_dir = os.path.dirname(args.weight_path)
    threshold_path = args.threshold_path or os.path.join(weight_dir, "best_thresholds.npy")
    if not os.path.exists(threshold_path):
        print_error_and_exit(
            "Thresholds file not found! Pass --threshold_path /path/to/best_thresholds.npy or place best_thresholds.npy beside --weight_path.",
            threshold_path,
        )
    coef = np.load(threshold_path)
    print_success("Regression thresholds loaded: " + str(coef))

    # Load Model (Required for Faithfulness Metrics)
    model_name_timm = f"tf_efficientnet_{args.model.lower()}_ns"
    model = APTOSModel(model_name=model_name_timm, num_classes=1, use_cbam=args.use_cbam, pretrained=False)
    if not os.path.exists(args.weight_path):
        print_error_and_exit("Weights file not found!", args.weight_path)
    model.load_state_dict(torch.load(args.weight_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    print_success("Model weights loaded.")

    transform = get_valid_transforms(IMG_SIZE)

    # Trace True Labels
    true_labels = {}
    csv_files = glob.glob(os.path.join(MAPLES_DIR, "**", "diagnosis.csv"), recursive=True)
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            true_labels[str(row['name'])] = map_messidor_label(row['DR'])

    # Find unique image IDs from result CSVs, or from generated NPY filenames when CSVs are absent.
    image_ids = collect_image_ids_from_xai_dir(args.xai_dir)
    if not image_ids:
        print_error_and_exit(
            "No image IDs found. Expected recursive *_results.csv files with Image_ID or known XAI NPY filenames.",
            args.xai_dir,
        )

    print(f"[INFO] Found {len(image_ids)} unique image IDs in {args.xai_dir}")

    # Metrics Tracking
    method_order = ["GradCAM++", "Ada-SISE", "IG_Smooth", CCEM_METHOD]
    metric_names = [
        "EBPG",
        "SoftPG",
        "TopKPG_0_5",
        "TopKPG_1",
        "TopKPG_2",
        "Precision_0_5",
        "Precision_1",
        "Precision_2",
        "DistancePG",
        "Deletion",
        "Insertion",
        "OA",
    ]
    xai_metrics = {m: {metric_name: [] for metric_name in metric_names} for m in method_order}
    metric_diagnostics = {m: {"total": 0, "zero_heatmap": 0} for m in method_order}
    rows = []
    processed_count = 0
    skip_counts = {
        "missing_label": 0,
        "missing_xai_maps": 0,
        "missing_image": 0,
        "unreadable_image": 0,
        "empty_mask": 0,
    }
    ccem_grid = build_ccem_grid() if args.ccem_grid_search else []
    ccem_grid_scores = [[] for _ in ccem_grid]
    if ccem_grid:
        print_success(f"CCEM validation grid size: {len(ccem_grid)} parameter sets")

    print_step("FUSING MAPS AND CALCULATING METRICS")

    for img_id in tqdm(image_ids, desc="Evaluating"):
        if args.max_samples is not None and processed_count >= args.max_samples:
            break

        if img_id not in true_labels:
            skip_counts["missing_label"] += 1
            continue

        # Load NPY files
        gcam_path = find_method_npy(args.xai_dir, img_id, ["gradcam", "gcam"])
        adas_path = find_method_npy(args.xai_dir, img_id, ["adasise", "ada-sise"])
        igsg_path = find_method_npy(args.xai_dir, img_id, ["smoothig", "smooth-ig", "igsg"])

        if not (gcam_path and adas_path and igsg_path):
            skip_counts["missing_xai_maps"] += 1
            continue

        G = np.load(gcam_path)
        AdaS = np.load(adas_path)
        IG_map = np.load(igsg_path)

        # Image Processing for Model Input & Visualization
        img_path = get_exact_image_path(img_id, MESSIDOR_IMG_DIR)
        if not img_path:
            skip_counts["missing_image"] += 1
            continue

        original_img = cv2.imread(img_path)
        if original_img is None:
            skip_counts["unreadable_image"] += 1
            continue
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        master_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        for lesion in LESION_TYPES:
            l_path_train = os.path.join(MAPLES_DIR, "train", lesion, f"{img_id}.png")
            l_path_test = os.path.join(MAPLES_DIR, "test", lesion, f"{img_id}.png")
            l_path = l_path_train if os.path.exists(l_path_train) else l_path_test
            
            if os.path.exists(l_path):
                lesion_mask = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
                if lesion_mask is None:
                    continue
                if lesion_mask.shape[:2] != master_mask.shape[:2]:
                    lesion_mask = cv2.resize(lesion_mask, (master_mask.shape[1], master_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                master_mask = cv2.bitwise_or(master_mask, lesion_mask)

        clean_img, clean_mask = crop_image_and_mask(img_rgb, master_mask)
        clean_img = apply_clahe(clean_img)
        clean_img_resized = cv2.resize(clean_img, (IMG_SIZE, IMG_SIZE))
        clean_img_resized = np.clip(clean_img_resized, 0, 255).astype(np.uint8)
        vis_img_float = np.float32(clean_img_resized) / 255.0
        mask_binary = (cv2.resize(clean_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
        retina_mask = get_fundus_mask(vis_img_float, erode_size=21)
        input_tensor = transform(image=clean_img)['image'].unsqueeze(0).to(DEVICE)

        # Generate CCEM after retinal foreground extraction so background is suppressed.
        ccem_result = generate_ccem(
            G,
            AdaS,
            IG_map,
            retina_mask=retina_mask,
            image=vis_img_float,
            model=model,
            image_tensor=input_tensor,
            mode=args.ccem_mode,
            weights=(args.ccem_w_gradcam, args.ccem_w_adasise, args.ccem_w_ig),
            blur_sigma=args.ccem_blur_sigma,
            apply_component_filter=args.ccem_apply_component_filter,
            ig_gate_threshold=args.ccem_ig_gate_threshold,
            agreement_threshold=args.ccem_agreement_threshold,
            gate_floor=args.ccem_gate_floor,
            agreement_bonus=args.ccem_agreement_bonus,
            threshold_percentile=args.ccem_threshold_percentile,
            min_area=args.ccem_min_area,
            max_area_ratio=args.ccem_max_area_ratio,
            top_k_components=args.ccem_top_k_components,
            adaptive_temperature=args.adaptive_temperature,
            adaptive_faithfulness_weight=args.adaptive_faithfulness_weight,
            adaptive_agreement_weight=args.adaptive_agreement_weight,
            adaptive_compactness_weight=args.adaptive_compactness_weight,
            adaptive_containment_weight=args.adaptive_containment_weight,
            adaptive_peak_weight=args.adaptive_peak_weight,
            adaptive_sharpen_gamma=args.adaptive_sharpen_gamma,
            adaptive_soft_keep_percentile=args.adaptive_soft_keep_percentile,
            return_debug=args.save_ccem_debug,
        )
        if args.save_ccem_debug and isinstance(ccem_result, tuple):
            C, ccem_debug = ccem_result
        else:
            C = ccem_result
            ccem_debug = None
        np.save(os.path.join(OUTPUT_DIR, "npy", f"{img_id}_{CCEM_METHOD}.npy"), C)
        np.save(os.path.join(OUTPUT_DIR, "npy", f"{img_id}_CCEM.npy"), C)

        target_shape = mask_binary.shape
        G_eval = resize_heatmap_to_shape(G, target_shape)
        AdaS_eval = resize_heatmap_to_shape(AdaS, target_shape)
        IG_eval = resize_heatmap_to_shape(IG_map, target_shape)
        C_eval = resize_heatmap_to_shape(C, target_shape)

        # Get Prediction
        with torch.no_grad():
            pred_value = model(input_tensor).item()
            
        if pred_value < coef[0]: pred_class = 0
        elif pred_value < coef[1]: pred_class = 1
        elif pred_value < coef[2]: pred_class = 2
        elif pred_value < coef[3]: pred_class = 3
        else: pred_class = 4
            
        true_class = true_labels[img_id]

        # Metric Calculations (CALLING CORE)
        if np.sum(mask_binary) > 0:
            heatmaps = {"GradCAM++": G_eval, "Ada-SISE": AdaS_eval, "IG_Smooth": IG_eval, CCEM_METHOD: C_eval}
            row_data = {"Image_ID": img_id, "True_Grade": true_class, "Pred_Grade": pred_class}
            if ccem_debug is not None:
                for method_name, weight in ccem_debug["weights"].items():
                    row_data[f"CCEM_Weight_{method_name}"] = weight
                for method_name, score in ccem_debug["reliability"].items():
                    row_data[f"CCEM_Reliability_{method_name}"] = score
            
            for method, hm in heatmaps.items():
                stats = get_heatmap_stats(hm)
                is_zero_map = stats["max"] <= 1e-8 or stats["sum"] <= 1e-8
                if is_zero_map:
                    metric_diagnostics[method]["zero_heatmap"] += 1
                metric_diagnostics[method]["total"] += 1

                metrics = calculate_extended_metrics(
                    model=model,
                    image_tensor=input_tensor,
                    heatmap=hm,
                    mask_binary=mask_binary,
                    n_steps=args.metric_steps,
                )
                for metric_name, metric_value in metrics.items():
                    if metric_name not in metric_names:
                        continue
                    xai_metrics[method][metric_name].append(metric_value)
                    row_data[f"{method}_{metric_name}"] = metric_value
                row_data[f"{method}_HeatmapMin"] = stats["min"]
                row_data[f"{method}_HeatmapMax"] = stats["max"]
                row_data[f"{method}_HeatmapSum"] = stats["sum"]
                row_data[f"{method}_HeatmapNonzeroRatio"] = stats["nonzero_ratio"]

            rows.append(row_data)

            # Visualization
            fig, axes = plt.subplots(1, 6, figsize=(30, 5))
            axes[0].imshow(vis_img_float); axes[0].set_title(f"Orig | True: {true_class}, Pred: {pred_class}"); axes[0].axis('off')
            
            mask_overlay = np.zeros_like(vis_img_float)
            mask_overlay[:, :, 1] = mask_binary
            gt_vis = cv2.addWeighted(vis_img_float, 0.7, mask_overlay, 0.5, 0)
            axes[1].imshow(gt_vis); axes[1].set_title("Expert Master Mask"); axes[1].axis('off')
            
            axes[2].imshow(show_percentile_overlay(vis_img_float, G_eval)); axes[2].set_title("Grad-CAM++"); axes[2].axis('off')
            axes[3].imshow(show_percentile_overlay(vis_img_float, AdaS_eval)); axes[3].set_title("Ada-SISE"); axes[3].axis('off')
            axes[4].imshow(show_percentile_overlay(vis_img_float, IG_eval)); axes[4].set_title("IG + SmoothGrad"); axes[4].axis('off')
            axes[5].imshow(show_percentile_overlay(vis_img_float, C_eval)); axes[5].set_title(CCEM_METHOD, color='red', fontweight='bold'); axes[5].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "visuals", f"{img_id}_fusion.png"), dpi=150, bbox_inches='tight')
            plt.close()

            if ccem_grid:
                for grid_idx, params in enumerate(ccem_grid):
                    grid_map = generate_ccem(
                        G,
                        AdaS,
                        IG_map,
                        retina_mask=retina_mask,
                        image=vis_img_float,
                        mode="ig_anchored",
                        weights=(params["w_gradcam"], params["w_adasise"], params["w_ig"]),
                        blur_sigma=params["blur_sigma"],
                        apply_component_filter=True,
                        ig_gate_threshold=params["ig_gate_threshold"],
                        agreement_threshold=args.ccem_agreement_threshold,
                        gate_floor=args.ccem_gate_floor,
                        agreement_bonus=args.ccem_agreement_bonus,
                        threshold_percentile=params["threshold_percentile"],
                        min_area=args.ccem_min_area,
                        max_area_ratio=args.ccem_max_area_ratio,
                        top_k_components=params["top_k_components"],
                    )
                    grid_map = resize_heatmap_to_shape(grid_map, target_shape)
                    _, _, _, _, _, grid_oa = calculate_advanced_metrics(model, input_tensor, grid_map, mask_binary)
                    ccem_grid_scores[grid_idx].append(grid_oa)
        else:
            skip_counts["empty_mask"] += 1

        processed_count += 1

    for reason, count in skip_counts.items():
        if count > 0:
            print_warning(f"Skipped {count} image(s): {reason}")

    if processed_count == 0:
        details = "; ".join(f"{reason}={count}" for reason, count in skip_counts.items())
        print_error_and_exit("No images were fused. See skip counts above.", details)

    if ccem_grid:
        scored_grid = [
            (safe_nanmean(scores), params)
            for params, scores in zip(ccem_grid, ccem_grid_scores)
            if scores and not np.isnan(safe_nanmean(scores))
        ]
        if scored_grid:
            best_score, best_params = max(scored_grid, key=lambda item: item[0])
            best_payload = {"selection_metric": "ODExAI_Overall_OA", "mean_oa": best_score, "params": best_params}
            best_params_path = args.ccem_best_params_path or os.path.join(OUTPUT_DIR, "ccem_best_params.json")
            os.makedirs(os.path.dirname(best_params_path) or ".", exist_ok=True)
            with open(best_params_path, "w") as f:
                json.dump(best_payload, f, indent=2)
            print_success(f"Best CCEM validation params saved to {best_params_path}: {best_payload}")
        else:
            print_warning("CCEM grid search produced no scored parameter sets.")

    print_step("FINAL EVALUATION REPORT")
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "ccem_metrics_details.csv"), index=False)

    report_content = f"""
================================================================================
                    FINAL XAI FUSION REPORT (FROM NPY FILES)                  
================================================================================
Total Images Fused : {processed_count}

[1] ODExAI EVALUATION METRICS:
--------------------------------------------------------------------------------
Method               | EBPG (↑) | SoftPG (↑) | P@1 (↑) | DPG (↓) | Deletion (↓) | Insertion (↑) | OA (↑)
---------------------------------------------------------------------------------------------------------------
"""
    coverage_notes = []
    for method in method_order:
        if len(xai_metrics[method]['EBPG']) > 0:
            m_ebpg = safe_nanmean(xai_metrics[method]['EBPG']) * 100
            m_softpg = safe_nanmean(xai_metrics[method]['SoftPG']) * 100
            m_precision_1 = safe_nanmean(xai_metrics[method]['Precision_1']) * 100
            m_dpg = safe_nanmean(xai_metrics[method]['DistancePG'])
            m_del = safe_nanmean(xai_metrics[method]['Deletion'])
            m_ins = safe_nanmean(xai_metrics[method]['Insertion'])
            m_oa = safe_nanmean(xai_metrics[method]['OA'])
            total = metric_diagnostics[method]["total"]
            del_values = np.asarray(xai_metrics[method]["Deletion"], dtype=np.float32)
            ins_values = np.asarray(xai_metrics[method]["Insertion"], dtype=np.float32)
            oa_values = np.asarray(xai_metrics[method]["OA"], dtype=np.float32)
            valid = int(np.sum(~np.isnan(del_values) & ~np.isnan(ins_values) & ~np.isnan(oa_values))) if total > 0 else 0
            coverage = valid / total if total > 0 else 0.0
            oa_text = f"{m_oa:>8.4f}" + ("*" if coverage < 0.80 else "")
            if coverage < 0.80:
                coverage_notes.append(f"* {method}: only {valid} / {total} samples had valid faithfulness curves.")
            report_content += f"{method:<20} | {m_ebpg:>7.2f}% | {m_softpg:>9.2f}% | {m_precision_1:>6.2f}% | {m_dpg:>6.3f} | {m_del:>12.4f} | {m_ins:>13.4f} | {oa_text:>7}\n"
        else:
            report_content += f"{method:<20} | {'N/A':>8} | {'N/A':>10} | {'N/A':>7} | {'N/A':>7} | {'N/A':>12} | {'N/A':>13} | {'N/A':>7}\n"

    if coverage_notes:
        report_content += "\nWARNING: faithfulness coverage below 80%; Deletion/Insertion/OA are not comparable.\n"
        report_content += "\n".join(coverage_notes) + "\n"
    
    report_content += "\n[2] FAITHFULNESS DIAGNOSTICS:\n"
    for method in method_order:
        total = metric_diagnostics[method]["total"]
        del_values = np.asarray(xai_metrics[method]["Deletion"], dtype=np.float32)
        ins_values = np.asarray(xai_metrics[method]["Insertion"], dtype=np.float32)
        oa_values = np.asarray(xai_metrics[method]["OA"], dtype=np.float32)
        if total > 0:
            valid_faithfulness = int(np.sum(~np.isnan(del_values) & ~np.isnan(ins_values) & ~np.isnan(oa_values)))
        else:
            valid_faithfulness = 0
        undefined_faithfulness = total - valid_faithfulness
        zero_heatmaps = metric_diagnostics[method]["zero_heatmap"]
        coverage = valid_faithfulness / total if total > 0 else 0.0
        report_content += (
            f"Method: {method}\n"
            f"Valid faithfulness samples: {valid_faithfulness} / {total}\n"
            f"Undefined faithfulness samples: {undefined_faithfulness} / {total}\n"
            f"Zero-heatmap samples: {zero_heatmaps} / {total}\n"
        )

    if args.ccem_apply_component_filter:
        report_content += f"CCEM component-filter fallbacks: {ccem_core.COMPONENT_FILTER_FALLBACK_COUNT}\n"

    report_content += "================================================================================\n"
    
    print(report_content)
    with open(os.path.join(OUTPUT_DIR, "final_ccem_report.txt"), "w") as f:
        f.write(report_content)
    print_success(f"Execution completed. All results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()