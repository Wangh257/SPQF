#!/usr/bin/env bash
set -euo pipefail

# ======================== 只需修改这里 ========================
# 与 train.sh 使用同一个相位目录和 manifest。
phase_dir="data/raw_phase_no_mask"
phase_pattern='PSP_{index}.mat'
manifest="codes/phase_field/data/calibration_manifest_raw_no_mask.json"

# test：随机留出的3张；train：训练标定板；all：全部标定板。
split="test"

# .pt 为 Direct MLP；.npz 为普通多项式或逆多项式。
model_checkpoint="outputs/mlp_raw/best.pt"

# 每张标定板的三维结果和总体 metrics.json 保存到这里。
output_dir="results/mlp_raw_test_masked"

# 测试精度只统计 mask=True 的可靠区域，黑色圆点区域不计入误差。
use_mask="true"
mask_dir="cal/mask"
mask_pattern='mask_{index}.bmp'

device="auto"
python_bin="${PYTHON_BIN:-python}"
# =============================================================

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR:?}/.." && pwd)"
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="${PROJECT_ROOT:?}/codes${PYTHONPATH:+:${PYTHONPATH}}"

command=(
  "${python_bin}" -m phase_field.batch_test
  --model "${model_checkpoint}"
  --manifest "${manifest}"
  --phase-dir "${phase_dir}"
  --phase-pattern "${phase_pattern}"
  --split "${split}"
  --output "${output_dir}"
  --device "${device}"
)

case "${use_mask}" in
  true|TRUE|1|yes|YES)
    command+=(--mask-dir "${mask_dir}" --mask-pattern "${mask_pattern}")
    ;;
  false|FALSE|0|no|NO|"")
    ;;
  *)
    echo "use_mask 只能填写 true 或 false。"
    exit 2
    ;;
esac

"${command[@]}"
