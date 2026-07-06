#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_DIR="xai_result/gradcampp"
CCEM_INPUT_DIR="xai_result/ccem_input"

echo "==> Removing old Grad-CAM++ output: ${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}"

if [ -d "${CCEM_INPUT_DIR}" ]; then
    echo "==> Removing stale Grad-CAM++ maps from ${CCEM_INPUT_DIR}"
    rm -f "${CCEM_INPUT_DIR}"/*_GradCAMpp_compact.npy
fi

echo "==> Rerunning Grad-CAM++"
python src/xai/explanation/run_maples_gradcampp.py \
  --model B7 --use_cbam \
  --weight_path src/experiments/Stage2_Finetune_B7_CBAM/stage2_best_model.pth \
  --img_size 600 \
  --maples_dir datasets/MAPLES-DR \
  --messidor_img_dir datasets/messidor-2-combined \
  --max_samples 999 \
  --output_dir "${OUTPUT_DIR}"
