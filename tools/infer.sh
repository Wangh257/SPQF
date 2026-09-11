#!/usr/bin/env bash
set -euo pipefail

# ======================== 只需修改这里 ========================
# 原始条纹图像文件夹（可以包含一组或多组完整序列）
input_dir="data/object_images"

# Direct MLP 填 best.pt；多项式填 .npz 模型
model_checkpoint="outputs/mlp_raw/best.pt"

# 三维结果保存目录
save_dir="results/object_mlp_raw"

# 正常物体建议 false。只在确实想限制输出区域时设为 true。
use_mask="false"

# use_mask=true 时填写。单组可填 mask.bmp，多组填包含 mask_1.bmp ... 的目录。
mask_path=""

# 采集参数：4 频×12 步=48张；白图单独存放时不计入这一组。
# 只有白图确实混在每组条纹图最前面时，才改为 1。
num_frequencies="4"
num_steps="12"
frequency_ratios=(4 4 4)
blank_images_per_group="0"

# 默认使用当前终端的 python；也可填 Conda Python 的完整路径。
python_bin="${PYTHON_BIN:-python}"
# =============================================================

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR:?}/.." && pwd)"

resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    echo "${value}"
  else
    echo "${PROJECT_ROOT:?}/${value}"
  fi
}

if [[ -z "${input_dir}" || -z "${model_checkpoint}" || -z "${save_dir}" ]]; then
  echo "请先编辑 tools/infer.sh 顶部的 input_dir、model_checkpoint 和 save_dir。"
  exit 2
fi

INPUT_PATH="$(resolve_path "${input_dir}")"
MODEL_PATH="$(resolve_path "${model_checkpoint}")"
SAVE_PATH="$(resolve_path "${save_dir}")"

if [[ ! -d "${INPUT_PATH}" ]]; then
  echo "原始图像目录不存在：${INPUT_PATH}"
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "模型文件不存在：${MODEL_PATH}"
  exit 1
fi

case "${MODEL_PATH}" in
  *.pt)
    model_type="mlp"
    ;;
  *.npz)
    model_type="polynomial"
    ;;
  *)
    echo "无法识别模型类型：Direct MLP 应为 .pt，多项式应为 .npz"
    exit 1
    ;;
esac

case "${use_mask}" in
  true|TRUE|1|yes|YES|on|ON)
    mask_enabled="true"
    if [[ -z "${mask_path}" ]]; then
      echo "use_mask=true 时必须填写 mask_path。"
      exit 2
    fi
    MASK_SOURCE="$(resolve_path "${mask_path}")"
    if [[ ! -e "${MASK_SOURCE}" ]]; then
      echo "mask 不存在：${MASK_SOURCE}"
      exit 1
    fi
    ;;
  false|FALSE|0|no|NO|off|OFF|"")
    mask_enabled="false"
    ;;
  *)
    echo "use_mask 只能填 true 或 false，当前为：${use_mask}"
    exit 2
    ;;
esac

cd "${PROJECT_ROOT:?}"
export PYTHONPATH="${PROJECT_ROOT:?}/codes${PYTHONPATH:+:${PYTHONPATH}}"
PHASE_DIR="${SAVE_PATH:?}/extracted_phase"
if [[ -d "${SAVE_PATH}" && -n "$(ls -A "${SAVE_PATH}")" ]]; then
  echo "save_dir 非空，请换一个新目录，避免复用其他物体的旧相位：${SAVE_PATH}"
  exit 1
fi
mkdir -p "${PHASE_DIR:?}"

"${python_bin}" -m phase_field.extract_raw_phase \
  --image-dir "${INPUT_PATH}" \
  --output "${PHASE_DIR}" \
  --num-frequencies "${num_frequencies}" \
  --num-steps "${num_steps}" \
  --frequency-ratios "${frequency_ratios[@]}" \
  --blank-images-per-pose "${blank_images_per_group}"

shopt -s nullglob
PHASE_FILES=("${PHASE_DIR}"/PSP_*.mat)
shopt -u nullglob
if [[ ${#PHASE_FILES[@]} -eq 0 ]]; then
  echo "未生成任何展开相位。"
  exit 1
fi

for PHASE_FILE in "${PHASE_FILES[@]}"; do
  PHASE_NAME="$(basename -- "${PHASE_FILE}")"
  PHASE_NAME="${PHASE_NAME%.mat}"
  PHASE_INDEX="${PHASE_NAME#PSP_}"
  RESULT_DIR="${SAVE_PATH:?}/${PHASE_NAME}"
  inference_command=()

  if [[ "${model_type}" = "mlp" ]]; then
    inference_command=(
      "${python_bin}" -m phase_field.infer
      --checkpoint "${MODEL_PATH}"
      --phase "${PHASE_FILE}"
      --output "${RESULT_DIR}"
    )
  else
    inference_command=(
      "${python_bin}" -m phase_field.polynomial_infer
      --model "${MODEL_PATH}"
      --phase "${PHASE_FILE}"
      --output "${RESULT_DIR}"
    )
  fi

  if [[ "${mask_enabled}" = "true" ]]; then
    if [[ -d "${MASK_SOURCE}" ]]; then
      CURRENT_MASK="${MASK_SOURCE}/mask_${PHASE_INDEX}.bmp"
    else
      CURRENT_MASK="${MASK_SOURCE}"
    fi
    if [[ ! -f "${CURRENT_MASK}" ]]; then
      echo "未找到 ${PHASE_NAME} 的 mask：${CURRENT_MASK}"
      exit 1
    fi
    inference_command+=(--mask "${CURRENT_MASK}")
  fi

  "${inference_command[@]}"
  echo "${PHASE_NAME} 三维结果：${RESULT_DIR}"
done

echo "全部完成，输出根目录：${SAVE_PATH}"
