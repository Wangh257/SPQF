#!/usr/bin/env bash
set -euo pipefail

# ======================== 只需修改这里 ========================
# 每个标定板姿态的 48 张相移图所在目录。
# 白图和 mask 请分开存放，不要放进这个目录。
input_dir="cal/PSPImg"

# 原始展开相位输出目录。
output_dir="data/raw_phase_no_mask"

# true 会重新计算并覆盖已有 PSP_*.mat；false 会跳过已有文件。
overwrite="false"

python_bin="${PYTHON_BIN:-python}"
# =============================================================

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR:?}/.." && pwd)"

if [[ "$#" -ne 0 ]]; then
  echo "无需在命令后填参数，请修改 extract_raw_phase.sh 顶部的 input_dir 和 output_dir。"
  echo "然后运行：bash tools/extract_raw_phase.sh"
  exit 2
fi

cd "${PROJECT_ROOT:?}"
export PYTHONPATH="${PROJECT_ROOT:?}/codes${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d "${input_dir}" ]]; then
  echo "未找到原始相移图目录：${input_dir}"
  exit 1
fi

command=(
  "${python_bin}" -m phase_field.extract_raw_phase
  --image-dir "${input_dir}"
  --output "${output_dir}"
  --image-glob '*.bmp'
  --num-frequencies 4
  --num-steps 12
  --frequency-ratios 4 4 4
  --blank-images-per-pose 0
)

case "${overwrite}" in
  true|TRUE|1|yes|YES)
    command+=(--overwrite)
    ;;
  false|FALSE|0|no|NO|"")
    ;;
  *)
    echo "overwrite 只能填写 true 或 false，当前为：${overwrite}"
    exit 2
    ;;
esac

"${command[@]}"
echo "原始展开相位已保存到：${output_dir}"
