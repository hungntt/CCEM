#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

XAI_DIR="xai_result/ccem_input"
OUTPUT_DIR="${XAI_DIR}/CCEM_Evaluation_Results"

echo "==> Removing old CCEM output: ${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}"

echo "==> Rerunning CCEM"
python src/xai/CCEM/run_maples_ccem.py \
  --xai_dir "${XAI_DIR}" \
  --weight_path src/experiments/Stage2_Finetune_B7_CBAM/stage2_best_model.pth \
  --threshold_path src/experiments/Stage2_Finetune_B7_CBAM/best_thresholds.npy \
  --maples_dir datasets/MAPLES-DR \
  --messidor_img_dir datasets/messidor-2-combined \
  --model B7 --use_cbam \
  --ccem_mode adaptive_reliability \
  --max_samples 999 \
  --save_ccem_debug
