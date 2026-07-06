# MAPLES Explanation README

This folder contains the MAPLES/MESSIDOR explanation runners for two compact
XAI methods:

- **AdaSISE**: activation/gradient-based channel selection over EfficientNet
  feature layers.
- **SmoothIG**: Integrated Gradients with SmoothGrad-style noise aggregation,
  implemented with Captum.

Both runners load a trained diabetic retinopathy regression model, convert the
raw model output to a DR grade with saved thresholds, generate a compact heatmap
for each MAPLES image, and evaluate localization against the MAPLES lesion masks.

## Files

```text
src/xai/explanation/
  run_maples_adasise.py      # Main AdaSISE CLI and Python runner
  run_maples_smoothig.py     # Main SmoothIG CLI and Python runner
  maple_utils.py             # Shared MAPLES loading, metrics, and heatmap cleanup
  explainer/
    adasise.py               # AdaSISE implementation
    smoothig.py              # Integrated Gradients + SmoothGrad implementation
    rise.py                  # RISE explainer implementation
```

Convenience wrappers are also available in:

```text
scripts/test_maples_adasise.py
scripts/test_maples_smoothig.py
```

## Required Inputs

Run commands from the project root.

The default dataset paths are:

```text
datasets1/      # MESSIDOR image files
MAPLES-DR/      # MAPLES masks and diagnosis.csv files
```

You can override them with `--messidor_img_dir` and `--maples_dir`.

The model checkpoint path is required:

```text
--weight_path src/experiments/Stage2_Finetune_MESSIDOR_B7_Batch16_CBAM/stage2_best_model.pth
```

The runner expects these files beside the checkpoint:

```text
best_thresholds.npy        # Regression thresholds used to map raw values to grades 0-4
messidor_valid_split.csv   # Optional; if present, restricts MAPLES images to validation IDs
```

If the checkpoint contains CBAM weights, pass `--use_cbam`.

## Grad-CAM++

Grad-CAM++ computes pixel importance by weighting the spatial gradients flowing into the final convolutional layer (or the CBAM module). This implementation utilizes the `pytorch_grad_cam` library, making it extremely fast, stable, and capable of highlighting multiple small lesion regions. 

Example:

```bash
python src/xai/explanation/run_maples_gradcampp.py \
  --model B7 \
  --weight_path src/experiments/Stage2_Finetune_MESSIDOR_B7_Batch16_CBAM/stage2_best_model.pth \
  --use_cbam \
  --max_samples 5 \
  --img_size 600 \
  --keep_percentile 94 \
  --gamma 2.5 \
  --blur_sigma 0.6 \
  --alpha 0.30
```

Important Grad-CAM++ options:

| Option | Default | Description |
| --- | ---: | --- |
| `--use_cbam` | off | If provided, Grad-CAM++ will target the CBAM spatial attention module instead of the default EfficientNet `conv_head`. |

## AdaSISE

AdaSISE selects activation channels from one or more target layers, builds masks
from those channels, scores their effect on the model output, and combines them
into a normalized heatmap. The runner then applies the shared compact-map
post-processing: fundus masking, optic-disc suppression, lesion candidate prior,
percentile pruning, gamma sharpening, blur, and component filtering.

Example:

```bash
python src/xai/explanation/run_maples_adasise.py \
  --model B7 \
  --weight_path src/experiments/Stage2_Finetune_MESSIDOR_B7_Batch16_CBAM/stage2_best_model.pth \
  --use_cbam \
  --max_samples 5 \
  --img_size 600 \
  --gpu_batch 8 \
  --target_layer_mode lesion \
  --keep_percentile 94 \
  --gamma 2.5 \
  --blur_sigma 0.6 \
  --alpha 0.30
```

Important AdaSISE options:

| Option | Default | Description |
| --- | ---: | --- |
| `--target_layer_mode` | `lesion` | Target EfficientNet layers. Choices: `auto`, `lesion`, `mid`, `late`, `all`, `semantic`. |
| `--gpu_batch` | `16` | Batch size used when scoring generated masks. Lower this if GPU memory is limited. |
| `--mask_power` | `2.0` | Sharpens selected activation masks before scoring. |
| `--max_mask_area_ratio` | `0.20` | Reject masks that cover too much of the image. |
| `--min_mask_area_ratio` | `0.0005` | Reject masks that are too small. |
| `--otsu_relax_factor` | `1.0` | Adjusts the Otsu threshold used for channel masks. |
| `--min_selected_channels` | `None` | Optional lower bound for selected channels. |
| `--max_selected_channels` | `None` | Optional upper bound for selected channels. |

Target layer modes:

| Mode | Meaning |
| --- | --- |
| `auto` | Let the explainer infer its default target layers. |
| `lesion` | Early/mid EfficientNet blocks, intended for small lesion evidence. |
| `mid` | Mid-level blocks. |
| `late` | Last three EfficientNet blocks. |
| `all` | Blocks from index 1 onward. |
| `semantic` | Late blocks plus `conv_head`; includes CBAM when available. |

## SmoothIG

SmoothIG computes Integrated Gradients over noisy copies of the input and
aggregates the attributions with one of the SmoothGrad variants. The
implementation streams noise samples in small chunks to reduce peak GPU memory.

Example:

```bash
python src/xai/explanation/run_maples_smoothig.py \
  --model B7 \
  --weight_path src/experiments/Stage2_Finetune_MESSIDOR_B7_Batch16_CBAM/stage2_best_model.pth \
  --use_cbam \
  --max_samples 5 \
  --img_size 600 \
  --nt_samples 16 \
  --stdevs 0.10 \
  --n_steps 24 \
  --nt_type smoothgrad_sq \
  --attribution_mode abs \
  --internal_batch_size 4 \
  --keep_percentile 94 \
  --gamma 2.5 \
  --blur_sigma 0.6 \
  --alpha 0.30
```

Important SmoothIG options:

| Option | Default | Description |
| --- | ---: | --- |
| `--nt_samples` | `64` | Number of noisy samples. Higher is smoother but slower. |
| `--stdevs` | `0.10` | Standard deviation of the input noise. |
| `--n_steps` | `80` | Number of Integrated Gradients interpolation steps. |
| `--nt_type` | `smoothgrad_sq` | Aggregation mode. Choices: `smoothgrad`, `smoothgrad_sq`, `vargrad`. |
| `--attribution_mode` | `abs` | Channel reduction mode. Choices: `abs`, `positive`, `signed`. |
| `--internal_batch_size` | `None` | Captum internal batch size. Use a smaller value for limited GPU memory. |
| `--no_heatmap_blur` | off | Disable Gaussian blur in the raw SmoothIG heatmap post-processing. |
| `--heatmap_blur_ksize` | `15` | Gaussian blur kernel size for raw SmoothIG heatmaps. |

## Shared Compact Heatmap Options

These options are available in both runners:

| Option | Default | Description |
| --- | ---: | --- |
| `--max_samples` | `10` | Maximum number of images to explain. |
| `--img_size` | `600` | Model input and heatmap resolution. |
| `--keep_percentile` | `94.0` | Keep only the highest heatmap values inside the fundus mask. |
| `--gamma` | `2.5` | Sharpen compact heatmaps after percentile pruning. |
| `--blur_sigma` | `0.6` | Final compact-map Gaussian blur sigma. |
| `--alpha` | `0.30` | Overlay blending value used by the explainer overlay helper. |
| `--output_dir` | auto | Custom output directory. |

## Outputs

If `--output_dir` is not provided, outputs are written under `scripts`:

```text
scripts/adasise_results/AdaSISE_Eff_<MODEL>_<CBAM|Base>_<MODE>_<YYYYMMDD_HHMM>/
scripts/igsg_results/IGSG_Eff_<MODEL>_<CBAM|Base>_<NT_TYPE>_<YYYYMMDD_HHMM>/
```

Each run creates:

```text
visuals/                 # PNG figures: original, expert mask, heatmap, overlay
npy/                     # Compact heatmaps as .npy arrays
adasise_results.csv      # AdaSISE per-image metrics, or
igsg_results.csv         # SmoothIG per-image metrics
adasise_report.txt       # AdaSISE run summary, or
igsg_report.txt          # SmoothIG run summary
```

The CSV files include true grade, predicted grade, raw model value, target class,
localization metrics, and the main run settings.

## Metrics

Localization is evaluated against a MAPLES master lesion mask built from:

```text
Microaneurysms
Hemorrhages
Exudates
CottonWoolSpots
```

The reported metrics are:

| Metric | Meaning |
| --- | --- |
| `Energy` | Fraction of heatmap energy inside the expert lesion mask. |
| `AUC` | Pixel-level ROC-AUC between heatmap intensity and lesion mask. |
| `IoU` | Intersection-over-union after thresholding the heatmap at `0.5`. |

Classification metrics are also reported:

| Metric | Meaning |
| --- | --- |
| `Accuracy` | Exact DR grade accuracy on explained samples. |
| `QWK` | Quadratic weighted kappa on explained samples. |

## Troubleshooting

- **Weights file not found**: check `--weight_path`.
- **Thresholds file not found**: place `best_thresholds.npy` in the same folder
  as the checkpoint.
- **CBAM mismatch**: add `--use_cbam` when loading a checkpoint that contains
  `cbam.*` weights.
- **Out of memory with AdaSISE**: reduce `--gpu_batch`.
- **Out of memory with SmoothIG**: reduce `--nt_samples`, `--n_steps`, or
  `--internal_batch_size`.
- **No images processed**: verify that MESSIDOR image names match MAPLES mask
  base names, and check whether `messidor_valid_split.csv` is restricting the
  run to IDs that are not present in `MAPLES-DR`.
