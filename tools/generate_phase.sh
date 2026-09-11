#!/usr/bin/env bash
set -euo pipefail

# ======================== 只需修改这里 ========================
# 可选模式：
# gaussian             全图高斯随机噪声
# offset               固定区域增加普通相位偏置
# fringe_jump_region   固定区域块整体增加/减少 2pi
# fringe_jump_blocks   随机位置的多个小块，每块随机 +/-2pi
# fringe_jump_pixels   随机零散像素，每个像素随机 +/-2pi
# cpc                  mask + CPC 相位曲面拟合
mode="cpc"

# 原始相位目录
input_dir="data/raw_phase_no_mask"

# 留空时根据 mode 和参数自动生成目录名；也可以自己指定。
output_dir=""

# gaussian 参数：加入均值为 mean、标准差为 sigma 的随机误差，单位 rad。
sigma="0.05"
mean="0.0"
seed="42"

# offset 参数：指定区域整体增加 value rad。
value="0.1"
# 原 crop_area=(230,206,870,750)；这里取其中心附近 50x50 的小区域。
region=(525 453 50 50)  # x y width height

# fringe_jump 参数：1 表示区域相位 +2pi，-1 表示 -2pi。
orders="1"

# fringe_jump_blocks：在原 crop_area 内随机放置小块，每块随机 +/-2pi。
block_count="10"
block_size="10"  # 一次选择一种：5、10 或 20
blocks_region=(0 0 1200 900)  # x y width height
# blocks_region=(230 206 640 544)  # x y width height

# fringe_jump_pixels：在原 crop_area 内随机选择零散像素。
pixel_count="500"
# pixels_region=(230 206 640 544)  # x y width height
pixels_region=(0 0 1200 900)  # x y width height

# cpc 参数
mask_dir="cal/mask"
mask_pattern='mask_{index}.bmp'
cpc_order="2"

# true 会覆盖已有相位；false 会跳过已有文件。
overwrite="false"
python_bin="${PYTHON_BIN:-python}"
# =============================================================

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR:?}/.." && pwd)"
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="${PROJECT_ROOT:?}/codes${PYTHONPATH:+:${PYTHONPATH}}"

command=(
  "${python_bin}" -m phase_field.generate_phase_variants
  --input-dir "${input_dir}"
)

case "${mode}" in
  gaussian)
    output_dir="${output_dir:-data/phase_variants/gaussian_sigma${sigma}_seed${seed}}"
    command+=(
      --output-dir "${output_dir}"
      --mode gaussian
      --sigma "${sigma}"
      --mean "${mean}"
      --seed "${seed}"
    )
    ;;
  offset)
    output_dir="${output_dir:-data/phase_variants/offset_${value}_x${region[0]}_y${region[1]}_w${region[2]}_h${region[3]}}"
    command+=(
      --output-dir "${output_dir}"
      --mode offset
      --value "${value}"
      --region "${region[@]}"
    )
    ;;
  fringe_jump|fringe_jump_region)
    output_dir="${output_dir:-data/phase_variants/fringe_jump_region_${orders}_x${region[0]}_y${region[1]}_w${region[2]}_h${region[3]}}"
    command+=(
      --output-dir "${output_dir}"
      --mode fringe_jump_region
      --orders "${orders}"
      --region "${region[@]}"
    )
    ;;
  fringe_jump_pixels)
    output_dir="${output_dir:-data/phase_variants/fringe_jump_pixels_n${pixel_count}_seed${seed}}"
    command+=(
      --output-dir "${output_dir}"
      --mode fringe_jump_pixels
      --orders "${orders}"
      --pixel-count "${pixel_count}"
      --region "${pixels_region[@]}"
      --seed "${seed}"
    )
    ;;
  fringe_jump_blocks)
    output_dir="${output_dir:-data/phase_variants/fringe_jump_blocks_size${block_size}_n${block_count}_seed${seed}}"
    command+=(
      --output-dir "${output_dir}"
      --mode fringe_jump_blocks
      --orders "${orders}"
      --block-count "${block_count}"
      --block-sizes "${block_size}"
      --region "${blocks_region[@]}"
      --seed "${seed}"
    )
    ;;
  cpc)
    output_dir="${output_dir:-data/raw_phase_cpc_mask}"
    command+=(
      --output-dir "${output_dir}"
      --mode cpc
      --mask-dir "${mask_dir}"
      --mask-pattern "${mask_pattern}"
      --cpc-order "${cpc_order}"
    )
    ;;
  *)
    echo "未知 mode：${mode}"
    echo "可选：gaussian、offset、fringe_jump_region、fringe_jump_blocks、fringe_jump_pixels、cpc"
    exit 2
    ;;
esac

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
echo "完成：mode=${mode}"
echo "相位目录：${output_dir}"
