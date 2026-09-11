#!/usr/bin/env bash
set -euo pipefail

# ======================== 只需修改这里 ========================
camera_params="data/camera_numeric.mat"
phase_dir="data/raw_phase_no_mask"
output_path="codes/phase_field/data/calibration_manifest_new.json"
test_count="3"
seed="42"
python_bin="${PYTHON_BIN:-python}"
# 不传 --mask-dir：不把评价 mask 写进训练清单。
# 新相机/新标定姿态必须生成新清单；仅更换同批相位处理方式不必重建。
# =============================================================
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR:?}/.." && pwd)"
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="${PROJECT_ROOT:?}/codes${PYTHONPATH:+:${PYTHONPATH}}"
[[ ! -e "${output_path}" ]] || { echo "输出已存在，请改 output_path：${output_path}"; exit 1; }
"${python_bin}" -m phase_field.prepare_manifest \
  --camera-params "${camera_params}" --phase-dir "${phase_dir}" \
  --phase-glob 'PSP_*.mat' --val-count 0 --test-count "${test_count}" \
  --seed "${seed}" --output "${output_path}"
