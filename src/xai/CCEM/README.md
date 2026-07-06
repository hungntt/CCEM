## IG-Anchored Consensus-Calibrated Explanation Map (CCEM)

The active CCEM algorithm now uses IG + SmoothGrad as the anchor and treats Ada-SISE and Grad-CAM++ as supporting evidence. The default fusion is:

```text
0.60 * IG_Smooth + 0.30 * Ada-SISE + 0.10 * GradCAM++
```

The weighted map is gated by IG support, receives a small agreement multiplier, can be constrained to the retinal foreground, is lightly blurred, normalized to `[0, 1]`, and then optionally cleaned with connected-component filtering. Expert lesion masks are used only for evaluation metrics, never inside fusion.

A legacy soft-union / Noisy-OR mode remains available with `--ccem_mode legacy_soft_union` for reproducibility, but the runner defaults to `--ccem_mode ig_anchored` and reports the fused method as `CCEM`.

### Directory Structure

```text
src/xai/CCEM/
  ccem_core.py               # Fusion utilities, IG-anchored CCEM, legacy soft-union CCEM, ODExAI metrics
  run_maples_ccem.py         # CLI runner to fuse .npy files and generate metrics/visuals
```

### Example Usage

Run from the project root after generating the Grad-CAM++, AdaSISE, and SmoothIG `.npy` outputs into the same `--xai_dir`.

```text
python src/xai/CCEM/run_maples_ccem.py \
  --xai_dir scripts/XAI_10ex_run \
  --weight_path src/experiments/Stage2_Finetune_B7_CBAM/stage2_best_model.pth \
  --threshold_path src/experiments/Stage2_Finetune_B7_CBAM/best_thresholds.npy \
  --model B7 \
  --use_cbam \
  --ccem_mode ig_anchored \
  --ccem_w_ig 0.60 \
  --ccem_w_adasise 0.30 \
  --ccem_w_gradcam 0.10 \
  --ccem_ig_gate_threshold 0.35 \
  --ccem_threshold_percentile 88 \
  --ccem_top_k_components 8 \
  --ccem_blur_sigma 0.8
```

`--threshold_path` should point to the `best_thresholds.npy` produced by the same training run as the checkpoint. If omitted, the runner looks for `best_thresholds.npy` beside `--weight_path`.

Optional validation tuning is available with `--ccem_grid_search`; use it only on a validation split, not a held-out test split. The best parameter set is saved to `CCEM_Evaluation_Results/ccem_best_params.json` unless `--ccem_best_params_path` is provided.

### Output

```text
scripts/XAI_10ex_run/CCEM_Evaluation_Results/
  visuals/                 # 6-panel comparison figures
  npy/                     # CCEM heatmaps and compatibility CCEM aliases
  ccem_metrics_details.csv # Per-image CCEM and XAI metrics
  final_ccem_report.txt    # Global mean metrics summary
```
