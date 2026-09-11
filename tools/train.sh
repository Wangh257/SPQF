#!/usr/bin/env bash
set -euo pipefail

# ======================== 只需修改这里 ========================
method="mlp"                       # mlp / poly / inverse_poly
degree="3"                         # 仅多项式使用，可改为 2、4、5 等
phase_dir="data/raw_phase_no_mask"  # 读取已准备好的相位，不执行 CPC
phase_pattern='PSP_{index}.mat'
manifest="codes/phase_field/data/calibration_manifest_raw_no_mask.json"
config="codes/phase_field/configs/direct_mlp_raw_no_mask.yaml"
output_dir="outputs/mlp_raw"        # 不同方法/相位请使用不同目录
use_mask="true"                    # 仅用于评价，不改变训练相位
mask_dir="cal/mask"
mask_pattern='mask_{index}.bmp'
python_bin="${PYTHON_BIN:-python}"
# =============================================================
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR:?}/.." && pwd)"
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="${PROJECT_ROOT:?}/codes${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "$#" -ne 0 ]]; then
  echo "请修改顶部配置，然后运行 bash tools/train.sh（不加参数）。"
  exit 2
fi
[[ -d "${phase_dir}" ]] || { echo "相位目录不存在：${phase_dir}"; exit 1; }
[[ -f "${manifest}" ]] || { echo "manifest 不存在：${manifest}"; exit 1; }
eval_args=()
case "${use_mask}" in
  true) eval_args=(--eval-mask-dir "${mask_dir}" --eval-mask-pattern "${mask_pattern}") ;;
  false) ;;
  *) echo 'use_mask 只能为 true 或 false'; exit 2 ;;
esac
case "${method}" in
  mlp)
    "${python_bin}" -m phase_field.train \
      --config "${config}" --manifest "${manifest}" \
      --phase-dir "${phase_dir}" --phase-pattern "${phase_pattern}" \
      --output-dir "${output_dir}" ${eval_args[@]+"${eval_args[@]}"}
    ;;
  poly|inverse_poly)
    [[ "${degree}" =~ ^[1-9][0-9]*$ ]] || { echo 'degree 必须为正整数'; exit 2; }
    module="phase_field.polynomial_calibrate"
    [[ "${method}" != inverse_poly ]] || module="phase_field.inverse_polynomial_calibrate"
    "${python_bin}" -m "${module}" \
      --manifest "${manifest}" --phase-dir "${phase_dir}" \
      --phase-pattern "${phase_pattern}" --degree "${degree}" \
      --fit-split train --eval-split test --output "${output_dir}" ${eval_args[@]+"${eval_args[@]}"}
    ;;
  *) echo 'method 可选：mlp / poly / inverse_poly'; exit 2 ;;
esac
echo "训练/标定结果：${output_dir}"
