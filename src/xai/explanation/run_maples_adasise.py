import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # src/xai/explanation
XAI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))          # src/xai
SRC_DIR = os.path.abspath(os.path.join(XAI_DIR, ".."))              # src
TRAINING_DIR = os.path.join(SRC_DIR, "training")
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))        # repo root

for _p in (XAI_DIR, TRAINING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from augmentation import get_valid_transforms
from model import APTOSModel
from preprocessing import apply_clahe

from explanation.explainer.adasise import (
    AdaSISE,
    select_adasise_target_layers,
)

from explanation.maple_utils import (
    calculate_localization_metrics,
    collect_maples_base_names,
    crop_image_and_mask,
    final_compact_xai_map,
    get_exact_image_path,
    get_true_labels_dict,
    load_maples_master_mask,
    load_state_dict_safely,
    make_compact_sample_figure,
    predict_grade,
    print_error_and_exit,
    print_step,
    print_success,
    print_warning,
)


@dataclass
class MaplesAdaSiseConfig:
    project_root: str
    model_version: str
    weight_path: str
    use_cbam: bool = False

    max_samples: int = 10
    img_size: int = 600
    gpu_batch: int = 16

    target_layer_mode: str = "lesion"

    keep_percentile: float = 94.0
    gamma: float = 2.5
    blur_sigma: float = 0.6
    alpha: float = 0.30

    mask_power: float = 2.0
    max_mask_area_ratio: float = 0.20
    min_mask_area_ratio: float = 0.0005
    otsu_relax_factor: float = 1.0
    min_selected_channels: int | None = None
    max_selected_channels: int | None = None

    messidor_img_dir: str | None = None
    maples_dir: str | None = None
    output_dir: str | None = None

    run_version: str = datetime.now().strftime("%Y%m%d_%H%M")


def run_maples_adasise_sample(config: MaplesAdaSiseConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lesion_types = [
        "Microaneurysms",
        "Hemorrhages",
        "Exudates",
        "CottonWoolSpots",
    ]

    if config.messidor_img_dir is None:
        config.messidor_img_dir = os.path.join(config.project_root, "datasets1")

    if config.maples_dir is None:
        config.maples_dir = os.path.join(config.project_root, "MAPLES-DR")

    if config.output_dir is None:
        exp_name = f"AdaSISE_Eff_{config.model_version.upper()}"
        exp_name += "_CBAM" if config.use_cbam else "_Base"
        exp_name += f"_{config.target_layer_mode}"
        exp_name += f"_{config.run_version}"

        config.output_dir = os.path.join(
            config.project_root,
            "scripts",
            "adasise_results",
            exp_name,
        )

    visuals_dir = os.path.join(config.output_dir, "visuals")
    npy_dir = os.path.join(config.output_dir, "npy")

    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    print_step("ADASISE MAPLES SAMPLE INITIALIZATION")
    print_success(f"Device: {device}")
    print_success(f"Output dir: {config.output_dir}")

    weight_dir = os.path.dirname(config.weight_path)
    threshold_path = os.path.join(weight_dir, "best_thresholds.npy")

    if not os.path.exists(config.weight_path):
        print_error_and_exit("Weights file not found.", config.weight_path)

    if not os.path.exists(threshold_path):
        print_error_and_exit("Thresholds file not found.", threshold_path)

    coef = np.load(threshold_path)
    print_success(f"Regression thresholds loaded: {coef}")

    valid_split_path = os.path.join(weight_dir, "messidor_valid_split.csv")
    valid_ids = None

    if os.path.exists(valid_split_path):
        df_valid = pd.read_csv(valid_split_path)
        valid_ids = set(df_valid["image_id"].astype(str))
        print_success(
            f"Found validation split. Restricting to {len(valid_ids)} validation images."
        )
    else:
        print_warning("No validation split found. Using all available MAPLES images.")

    model_name_timm = f"tf_efficientnet_{config.model_version.lower()}_ns"

    model = APTOSModel(
        model_name=model_name_timm,
        num_classes=1,
        use_cbam=config.use_cbam,
        pretrained=False,
    )

    model = load_state_dict_safely(
        model=model,
        weight_path=config.weight_path,
        device=device,
        use_cbam=config.use_cbam,
    )

    model = model.to(device).eval()

    print_success(
        f"Model loaded: {model_name_timm} | CBAM: {config.use_cbam}"
    )

    target_layers = select_adasise_target_layers(
        model=model,
        mode=config.target_layer_mode,
    )

    if target_layers is None:
        print_success("AdaSISE target layers: auto")
    else:
        print_success(
            "AdaSISE target layers: " + ", ".join(name for name, _ in target_layers)
        )

    adasise = AdaSISE(
        model=model,
        input_size=(config.img_size, config.img_size),
        target_layers=target_layers,
        gpu_batch=config.gpu_batch,
        device=device,
        score_mode="raw",
        gradient_score_mode="raw",
        alpha=config.alpha,
        mask_power=config.mask_power,
        max_mask_area_ratio=config.max_mask_area_ratio,
        min_mask_area_ratio=config.min_mask_area_ratio,
        otsu_relax_factor=config.otsu_relax_factor,
        min_selected_channels=config.min_selected_channels,
        max_selected_channels=config.max_selected_channels,
    )

    transform = get_valid_transforms(config.img_size)

    true_labels_dict = get_true_labels_dict(config.maples_dir)
    base_names = collect_maples_base_names(config.maples_dir, lesion_types)

    y_true_list = []
    y_pred_list = []
    rows = []

    metrics = {
        "Energy": [],
        "AUC": [],
        "IoU": [],
    }

    processed_count = 0

    print_step(f"RUNNING COMPACT ADASISE | max_samples={config.max_samples}")

    for base_name in base_names:
        if valid_ids is not None and base_name not in valid_ids:
            continue

        if processed_count >= config.max_samples:
            break

        if base_name not in true_labels_dict:
            continue

        img_path = get_exact_image_path(base_name, config.messidor_img_dir)

        if img_path is None:
            continue

        original_img = cv2.imread(img_path)

        if original_img is None:
            print_warning(f"Could not read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        master_mask = load_maples_master_mask(
            base_name=base_name,
            maples_dir=config.maples_dir,
            image_shape=img_rgb.shape,
            lesion_types=lesion_types,
        )

        clean_img, clean_mask = crop_image_and_mask(img_rgb, master_mask)

        clean_img = apply_clahe(clean_img)

        clean_img_resized = cv2.resize(
            clean_img,
            (config.img_size, config.img_size),
            interpolation=cv2.INTER_LINEAR,
        )

        vis_img_float = np.float32(clean_img_resized) / 255.0

        mask_binary = (
            cv2.resize(
                clean_mask,
                (config.img_size, config.img_size),
                interpolation=cv2.INTER_NEAREST,
            )
            > 127
        ).astype(np.float32)

        input_tensor = transform(image=clean_img)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            raw_value = model(input_tensor).item()

        pred_class = predict_grade(raw_value, coef)
        true_class = true_labels_dict[base_name]

        y_true_list.append(true_class)
        y_pred_list.append(pred_class)

        raw_heatmap, target_class, debug = adasise.generate_heatmap(
            input_tensor=input_tensor,
            target_class=0,
            return_debug=True,
        )

        heatmap = final_compact_xai_map(
            raw_heatmap=raw_heatmap,
            image_rgb=vis_img_float,
            keep_percentile=config.keep_percentile,
            gamma=config.gamma,
            blur_sigma=config.blur_sigma,
            use_lesion_prior=True,
            suppress_optic_disc=True,
        )

        overlay = adasise.overlay(vis_img_float, heatmap)

        np.save(
            os.path.join(npy_dir, f"{base_name}_AdaSISE_compact.npy"),
            heatmap,
        )

        energy_score = auc_score = iou_score = np.nan

        if np.sum(mask_binary) > 0:
            energy_score, auc_score, iou_score = calculate_localization_metrics(
                heatmap,
                mask_binary,
            )

            metrics["Energy"].append(energy_score)
            metrics["AUC"].append(auc_score)
            metrics["IoU"].append(iou_score)

        fig_path = os.path.join(
            visuals_dir,
            f"{base_name}_adasise_compact.png",
        )

        make_compact_sample_figure(
            save_path=fig_path,
            vis_img_float=vis_img_float,
            mask_binary=mask_binary,
            heatmap=heatmap,
            overlay=overlay,
            true_class=true_class,
            pred_class=pred_class,
            method_name="Ada-SISE",
        )

        rows.append(
            {
                "Image_ID": base_name,
                "True_Grade": true_class,
                "Predicted_Grade": pred_class,
                "Raw_Value": round(raw_value, 4),
                "Target_Class": target_class,
                "Layer_Maps": debug.get("num_layer_maps", 0),
                "Energy": energy_score,
                "AUC": auc_score,
                "IoU": iou_score,
                "Target_Layer_Mode": config.target_layer_mode,
                "Keep_Percentile": config.keep_percentile,
                "Gamma": config.gamma,
                "Blur_Sigma": config.blur_sigma,
                "Mask_Power": config.mask_power,
                "Max_Mask_Area_Ratio": config.max_mask_area_ratio,
                "Min_Mask_Area_Ratio": config.min_mask_area_ratio,
                "Otsu_Relax_Factor": config.otsu_relax_factor,
                "Min_Selected_Channels": config.min_selected_channels,
                "Max_Selected_Channels": config.max_selected_channels,
            }
        )

        processed_count += 1

        print(
            f"   [DONE] {processed_count}: {base_name} | "
            f"true={true_class}, pred={pred_class}, "
            f"IoU={iou_score:.4f}, Energy={energy_score:.4f}, AUC={auc_score:.4f}"
        )

    csv_path = os.path.join(config.output_dir, "adasise_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    qwk_score = (
        cohen_kappa_score(y_true_list, y_pred_list, weights="quadratic")
        if len(y_true_list) > 1
        else 0.0
    )

    acc_score = accuracy_score(y_true_list, y_pred_list) if y_true_list else 0.0

    report_path = os.path.join(config.output_dir, "adasise_report.txt")

    mean_energy = np.mean(metrics["Energy"]) if metrics["Energy"] else float("nan")
    mean_auc = np.mean(metrics["AUC"]) if metrics["AUC"] else float("nan")
    mean_iou = np.mean(metrics["IoU"]) if metrics["IoU"] else float("nan")

    report_content = f"""
====================================================================================================
                                COMPACT ADASISE MAPLES REPORT
====================================================================================================
Total Images Explained : {processed_count}

[1] CLASSIFICATION PERFORMANCE
    Accuracy Score     : {acc_score:.4f}
    QWK Score          : {qwk_score:.4f}

[2] COMPACT ADASISE LOCALIZATION
    Mean Energy        : {mean_energy:.4f}
    Mean AUC-ROC       : {mean_auc:.4f}
    Mean IoU           : {mean_iou:.4f}

[3] SETTINGS
    Model              : EfficientNet-{config.model_version.upper()}
    CBAM               : {config.use_cbam}
    Target Layer Mode  : {config.target_layer_mode}
    Image Size         : {config.img_size}
    Keep Percentile    : {config.keep_percentile}
    Gamma              : {config.gamma}
    Blur Sigma         : {config.blur_sigma}
    Alpha              : {config.alpha}
    Mask Power         : {config.mask_power}
    Max Mask Area      : {config.max_mask_area_ratio}
    Min Mask Area      : {config.min_mask_area_ratio}
    Otsu Relax Factor  : {config.otsu_relax_factor}
    Min Selected Ch.   : {config.min_selected_channels}
    Max Selected Ch.   : {config.max_selected_channels}

[4] OUTPUTS
    Visuals            : {visuals_dir}
    Heatmaps           : {npy_dir}
    CSV                : {csv_path}
====================================================================================================
"""

    print(report_content)

    with open(report_path, "w") as f:
        f.write(report_content)

    adasise.remove_hook()

    return {
        "output_dir": config.output_dir,
        "visuals_dir": visuals_dir,
        "npy_dir": npy_dir,
        "csv_path": csv_path,
        "report_path": report_path,
        "processed_count": processed_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Compact AdaSISE on MAPLES/MESSIDOR samples."
    )

    parser.add_argument("--model", type=str, default="B7")
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--use_cbam", action="store_true")

    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=600)
    parser.add_argument("--gpu_batch", type=int, default=16)

    parser.add_argument(
        "--target_layer_mode",
        type=str,
        default="lesion",
        choices=["auto", "lesion", "mid", "late", "all", "semantic"],
    )

    parser.add_argument("--keep_percentile", type=float, default=94.0)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--blur_sigma", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.30)

    parser.add_argument("--mask_power", type=float, default=2.0)
    parser.add_argument("--max_mask_area_ratio", type=float, default=0.20)
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.0005)

    parser.add_argument("--otsu_relax_factor", type=float, default=1.0)
    parser.add_argument("--min_selected_channels", type=int, default=None)
    parser.add_argument("--max_selected_channels", type=int, default=None)

    parser.add_argument("--messidor_img_dir", type=str, default=None)
    parser.add_argument("--maples_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    config = MaplesAdaSiseConfig(
        project_root=PROJECT_ROOT,
        model_version=args.model,
        weight_path=args.weight_path,
        use_cbam=args.use_cbam,
        max_samples=args.max_samples,
        img_size=args.img_size,
        gpu_batch=args.gpu_batch,
        target_layer_mode=args.target_layer_mode,
        keep_percentile=args.keep_percentile,
        gamma=args.gamma,
        blur_sigma=args.blur_sigma,
        alpha=args.alpha,
        mask_power=args.mask_power,
        max_mask_area_ratio=args.max_mask_area_ratio,
        min_mask_area_ratio=args.min_mask_area_ratio,
        otsu_relax_factor=args.otsu_relax_factor,
        min_selected_channels=args.min_selected_channels,
        max_selected_channels=args.max_selected_channels,
        messidor_img_dir=args.messidor_img_dir,
        maples_dir=args.maples_dir,
        output_dir=args.output_dir,
    )

    result = run_maples_adasise_sample(config)

    print("\nDone.")
    print(f"Visuals:  {result['visuals_dir']}")
    print(f"Heatmaps: {result['npy_dir']}")
    print(f"CSV:      {result['csv_path']}")
    print(f"Report:   {result['report_path']}")


if __name__ == "__main__":
    main()