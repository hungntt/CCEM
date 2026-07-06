# CCEM: A Lesion-Aware Consensus-Calibrated Explanation for Diabetic Retinopathy Grading

Diabetic Retinopathy (DR) grading with a proposed Explainable AI (XAI) method:
CCEM — Consensus-Calibrated Explanation Map.

This project develops a DR grading framework based on an EfficientNet regression
model, optionally enhanced with a CBAM attention module, for predicting diabetic
retinopathy severity from fundus images.

The main contribution of the project is CCEM, a proposed explanation-fusion
method that combines complementary evidence from multiple XAI sources into a
single calibrated heatmap. CCEM is designed to produce explanation maps that are
more spatially coherent, model-faithful, and clinically interpretable than using
individual explanation methods alone.

The project also includes an evaluation protocol using expert-annotated lesion
masks from MAPLES-DR and faithfulness metrics such as deletion and insertion.
These evaluations are used to analyze the behavior of CCEM and compare it with
baseline explanation methods, including GradCAM++, Ada-SISE, and SmoothIG.

## Datasets

Located under `datasets/` and not tracked in git:

| Dataset | Purpose |
| --- | --- |
| APTOS 2019 Blindness Detection | Stage 2 fine-tuning / main DR grading dataset |
| EyePACS 2015 | Stage 1 large-scale pre-training |
| MESSIDOR-2 | Stage 2 fine-tuning + XAI benchmarking |
| MAPLES-DR | Expert lesion masks used as XAI ground truth |

Helper scripts: `datasets/download_maples.py`, `datasets/check_health.py`,
`datasets/check_maples.py`.

## Project Layout

See [`src/docs/localize.md`](src/docs/localize.md) for the full directory map.
High level:

```text
src/
  data/            # Preprocessing, folds, label auditing, blind grading prep
  training/        # Model (EfficientNet + CBAM), augmentation, 3-stage training
  xai/
    explanation/   # Grad-CAM++, AdaSISE, SmoothIG explainers + runners
    CCEM/          # Consensus fusion of the explanation maps
    improvement_v1/
  Archive/         # Deprecated scripts kept for reference/ablation
scripts/                  # Convenience wrappers and small test/rerun scripts
  rerun_gradcampp.sh      # Re-run Grad-CAM++ and refresh compact maps for CCEM input
  rerun_ccem.sh           # Re-run CCEM adaptive fusion and write evaluation outputs
results/                  # Benchmark reports and visuals
scripts/XAI_10ex_run/     # Sample XAI outputs used for 10-image benchmarking
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Model Training

Three-stage training pipeline, implemented under `src/training/`:

1. `train_stage1.py` - pre-train on EyePACS 2015.
2. `train_stage2.py` / `train_stage2_messidor.py` - fine-tune on APTOS 2019 / MESSIDOR.
3. `train_stage3_attention.py` - attention-guided fine-tuning using MAPLES-DR masks
   to supervise the CBAM module.

Example:

```bash
python src/training/train_stage2.py --model B7 --use_cbam --batch_size 16 --epochs 15
```

## Explainability Pipeline

Each explainer loads a trained checkpoint, produces a heatmap, and scores it
against MAPLES-DR lesion masks. See
[`src/xai/explanation/README.md`](src/xai/explanation/README.md) for full options.

```bash
python src/xai/explanation/run_maples_gradcampp.py \
  --model B7 --weight_path <checkpoint>.pth --use_cbam --img_size 600

python src/xai/explanation/run_maples_adasise.py \
  --model B7 --weight_path <checkpoint>.pth --use_cbam --target_layer_mode lesion

python src/xai/explanation/run_maples_smoothig.py \
  --model B7 --weight_path <checkpoint>.pth --use_cbam --nt_samples 16
```

## CCEM: Consensus-Calibrated Explanation Map

CCEM is a fusion layer over the available XAI maps. It does not use expert lesion
masks to construct the heatmap; masks are used only after fusion for evaluation.
The goal is to keep the complementary strengths of the input explainers while
suppressing unstable or visually diffuse evidence.

### Mechanism

At a high level, CCEM performs the following steps:

1. **Shape alignment and robust normalization**
   - Grad-CAM++, Ada-SISE, and SmoothIG maps are resized to a common spatial grid.
   - Each map is converted to a 2D non-negative saliency map.
   - Percentile normalization is used instead of raw min-max scaling so that rare
     outlier pixels do not dominate the fusion.

2. **Evidence fusion**
   - The current implementation supports several fusion modes:
     - `adaptive_reliability`: estimates a per-image reliability score for each
       explainer and converts those scores into soft fusion weights.
     - `ig_anchored`: a fixed weighted mode retained as a controlled ablation.
     - `simple_average`, `soft_agreement`, `ig_only`, and `legacy_soft_union` for
       reproducibility and ablation studies.
   - In the recommended adaptive mode, reliability is estimated from five internal
     signals: quick deletion faithfulness, agreement with other maps, compactness,
     retinal foreground containment, and peak dominance.
   - These reliability scores are passed through a softmax temperature, producing
     per-image weights instead of using one fixed global mixture for every case.

3. **Agreement and foreground calibration**
   - CCEM gives mild preference to regions supported by more than one explanation
     source, but it avoids hard intersection rules because true lesions can be
     missed by one explainer.
   - If the original retinal image is available, a foreground mask is estimated so
     that saliency outside the fundus region is suppressed.

4. **Map refinement and safety fallbacks**
   - The fused map is lightly blurred for overlay readability.
   - Optional connected-component filtering can keep the strongest plausible
     regions, but it includes fallbacks so that a nonzero explanation is not
     erased just because the component filter is too strict.
   - Final normalization maps the output to `[0, 1]` for visualization and metrics.

This design should be interpreted as a consensus-and-calibration strategy rather
than a new standalone attribution algorithm. The fused map is only as reliable as
the input explainers and the trained DR model.

### Running CCEM

Run CCEM after generating `.npy` heatmaps for Grad-CAM++, Ada-SISE, and SmoothIG
in the same `--xai_dir`.

```bash
python src/xai/CCEM/run_maples_ccem.py \
  --xai_dir scripts/XAI_10ex_run \
  --weight_path <checkpoint>.pth \
  --threshold_path <best_thresholds.npy> \
  --model B7 \
  --use_cbam \
  --ccem_mode adaptive_reliability
```

Outputs are written to:

```text
scripts/XAI_10ex_run/CCEM_Evaluation_Results/
  visuals/                 # Comparison figures
  npy/                     # CCEM heatmaps
  ccem_metrics_details.csv # Per-image metrics
  final_ccem_report.txt    # Global mean summary
```

Two convenience scripts are also provided for repeated local experiments:

```bash
bash scripts/rerun_gradcampp.sh
bash scripts/rerun_ccem.sh
```

`rerun_gradcampp.sh` clears the previous Grad-CAM++ output and regenerates maps
for the CCEM input directory. `rerun_ccem.sh` clears the previous CCEM evaluation
folder and reruns adaptive CCEM fusion with debug outputs enabled.

## ODExAI Evaluation Metrics

The current report uses the following metrics:

| Metric | Direction | Meaning |
| --- | --- | --- |
| EBPG | Higher is better | Fraction of total saliency energy inside the expert lesion mask. |
| SoftPG | Higher is better | Soft pointing score; gives credit when strong saliency overlaps the lesion even if the absolute peak is elsewhere. |
| P@1 | Higher is better | Precision of the top 1% most salient pixels against the lesion mask. |
| DPG | Lower is better | Distance-based pointing game; lower means the peak is closer to the lesion mask. |
| Deletion | Lower is better | AUC after progressively removing salient pixels; lower means important evidence is removed earlier. |
| Insertion | Higher is better | AUC after progressively inserting salient pixels; higher means the highlighted evidence restores the model response quickly. |
| OA | Higher is better | Overall faithfulness balance, computed as `Insertion - Deletion`. |

## XAI Evaluation Result

The following run fused and evaluated MAPLES/MESSIDOR dataset:

| Method | EBPG | SoftPG | P@1 | DPG | Deletion | Insertion | OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GradCAM++ | 5.86% | 43.19% | 2.96% | 0.422 | 0.6944 | 0.7038 | 0.0093 |
| Ada-SISE | 3.46% | 30.18% | 2.96% | 0.130 | 0.6394 | 0.6456 | 0.0062 |
| IG_Smooth | 10.53% | 56.83% | 8.71% | 0.344 | 0.6437 | 0.6910 | 0.0474 |
| CCEM_Adaptive | 9.19% | 68.03% | 7.78% | 0.279 | 0.5719 | 0.6835 | 0.1117 |

