import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.explanation.run_maples_adasise import (
    MaplesAdaSiseConfig,
    run_maples_adasise_sample,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Compact AdaSISE on MAPLES/MESSIDOR samples."
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
    """Sample code
    python scripts/test_maples_adasise.py \
        --model B7 \
        --weight_path src/Stage2_Finetune_MESSIDOR_B7_Batch16_CBAM/stage2_best_model.pth \
        --use_cbam \
        --max_samples 5 \
        --img_size 600 \
        --gpu_batch 8 \
        --target_layer_mode lesion \
        --keep_percentile 90 \
        --gamma 1.5 \
        --blur_sigma 1.0 \
        --alpha 0.60
    """
    main()