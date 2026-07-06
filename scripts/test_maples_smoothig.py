import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.explanation.run_maples_smoothig import (
    MaplesIGSmoothGradConfig,
    run_maples_ig_smoothgrad_sample,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Compact IG + SmoothGrad on MAPLES/MESSIDOR samples."
    )

    parser.add_argument("--model", type=str, default="B7")
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--use_cbam", action="store_true")

    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=600)

    parser.add_argument("--nt_samples", type=int, default=64)
    parser.add_argument("--stdevs", type=float, default=0.10)
    parser.add_argument("--n_steps", type=int, default=80)

    parser.add_argument(
        "--nt_type",
        type=str,
        default="smoothgrad_sq",
        choices=["smoothgrad", "smoothgrad_sq", "vargrad"],
    )

    parser.add_argument(
        "--attribution_mode",
        type=str,
        default="abs",
        choices=["abs", "positive", "signed"],
    )

    parser.add_argument("--internal_batch_size", type=int, default=None)

    parser.add_argument("--keep_percentile", type=float, default=94.0)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--blur_sigma", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.30)

    parser.add_argument("--heatmap_blur", action="store_true")
    parser.add_argument("--no_heatmap_blur", action="store_true")
    parser.add_argument("--heatmap_blur_ksize", type=int, default=15)

    parser.add_argument("--messidor_img_dir", type=str, default=None)
    parser.add_argument("--maples_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    heatmap_blur = True

    if args.no_heatmap_blur:
        heatmap_blur = False

    if args.heatmap_blur:
        heatmap_blur = True

    config = MaplesIGSmoothGradConfig(
        project_root=PROJECT_ROOT,
        model_version=args.model,
        weight_path=args.weight_path,
        use_cbam=args.use_cbam,
        max_samples=args.max_samples,
        img_size=args.img_size,
        nt_samples=args.nt_samples,
        stdevs=args.stdevs,
        n_steps=args.n_steps,
        nt_type=args.nt_type,
        attribution_mode=args.attribution_mode,
        internal_batch_size=args.internal_batch_size,
        keep_percentile=args.keep_percentile,
        gamma=args.gamma,
        blur_sigma=args.blur_sigma,
        alpha=args.alpha,
        heatmap_blur=heatmap_blur,
        heatmap_blur_ksize=args.heatmap_blur_ksize,
        messidor_img_dir=args.messidor_img_dir,
        maples_dir=args.maples_dir,
        output_dir=args.output_dir,
    )

    result = run_maples_ig_smoothgrad_sample(config)

    print("\nDone.")
    print(f"Visuals:  {result['visuals_dir']}")
    print(f"Heatmaps: {result['npy_dir']}")
    print(f"CSV:      {result['csv_path']}")
    print(f"Report:   {result['report_path']}")


if __name__ == "__main__":
    """Sample code
    python scripts/test_maples_smoothig.py \
        --model B7 \
        --weight_path src/Stage2_Finetune_MESSIDOR_B7_Batch16_CBAM/stage2_best_model.pth \
        --use_cbam \
        --max_samples 5 \
        --img_size 600 \
        --nt_samples 16 \
        --stdevs 0.10 \
        --n_steps 24 \
        --nt_type smoothgrad_sq \
        --attribution_mode abs \
        --keep_percentile 94 \
        --gamma 2.5 \
        --blur_sigma 0.6 \
        --alpha 0.30
    """
    main()
