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

echo "==> Copying regenerated Grad-CAM++ maps into CCEM input"
mkdir -p "${CCEM_INPUT_DIR}"

find "${OUTPUT_DIR}/npy" -name "*_GradCAMpp_compact.npy" -exec cp -f {} "${CCEM_INPUT_DIR}/" \;

echo "==> Verifying copied Grad-CAM++ maps"
python - <<'PY'
import glob
import numpy as np
import os

paths = sorted(glob.glob("xai_result/ccem_input/*_GradCAMpp_compact.npy"))

zero = []
for p in paths:
    hm = np.load(p)
    if hm.size == 0 or float(np.max(hm)) <= 1e-8 or float(np.sum(hm)) <= 1e-8:
        zero.append(p)

print(f"Grad-CAM++ maps in ccem_input: {len(paths)}")
print(f"Zero Grad-CAM++ maps in ccem_input: {len(zero)}")

if zero:
    print("First zero examples:")
    for p in zero[:10]:
        print("  ", p)

raise SystemExit(1 if zero else 0)
PY