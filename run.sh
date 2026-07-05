#!/bin/bash
# RTM3D V5 Competition Pipeline — 一键运行
# Usage:
#   bash run.sh                           # Board mode (RKNN), demo_100
#   bash run.sh --server                  # Server mode (PyTorch), demo_100
#   bash run.sh --dir demo_diverse        # Diverse lighting demo
#   bash run.sh --image test.jpg          # Single image inference

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${SCRIPT_DIR}/N-RTM3D-int8.rknn"
MODE="board"
IMAGE_DIR="${SCRIPT_DIR}/demo_100/images"
CALIB_DIR="${SCRIPT_DIR}/demo_100/calib"
OUTPUT="${SCRIPT_DIR}/output"
ENHANCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --server) MODE="server"; shift ;;
        --board)  MODE="board"; shift ;;
        --dir)    IMAGE_DIR="$2"; CALIB_DIR="$2/calib"; shift 2 ;;
        --image)  IMAGE_DIR=""; SINGLE_IMAGE="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --enhance) ENHANCE="--enhance"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "RTM3D V5 — Competition Pipeline"
echo "============================================"
echo "Mode:   ${MODE}"
echo "Model:  ${MODEL}"
echo "Output: ${OUTPUT}"
echo ""

if [[ "${MODE}" == "board" ]]; then
    BOARD_FLAG="--board"
    # On actual board, model is the RKNN file
    MODEL_ARG="--model ${MODEL}"
else
    BOARD_FLAG=""
    # Server: use PyTorch weights
    MODEL_ARG="--model ${SCRIPT_DIR}/N-RTM3D.pth"
fi

if [[ -n "${SINGLE_IMAGE}" ]]; then
    # Single image mode
    python3 rknn_infer.py --model "${MODEL}" --image "${SINGLE_IMAGE}" --calib "${CALIB_DIR}/camera_intrinsic_000032.json" --conf 0.15
elif [[ -n "${IMAGE_DIR}" ]]; then
    # Batch mode
    python3 main_pipeline.py \
        ${BOARD_FLAG} \
        ${MODEL_ARG} \
        --image-dir "${IMAGE_DIR}" \
        --calib-dir "${CALIB_DIR}" \
        --output "${OUTPUT}" \
        ${ENHANCE}
fi
