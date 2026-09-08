#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
KROOT="${KROOT:?Set KROOT to Kvasir-SEG}"
OUT="${OUT:?Set OUT to output directory}"
WEIGHTS="${WEIGHTS:?Set WEIGHTS to resnet18-5c106cde.pth}"

export NO_ALBUMENTATIONS_UPDATE=1
export TORCH_HOME="$(dirname "$(dirname "$(dirname "$WEIGHTS")")")"
export USPG_RESNET18_WEIGHTS="$WEIGHTS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

common=(--batch_size 16 --image_size 256 --kvasir_root "$KROOT"
        --num_workers 0 --output_root "$OUT" --no-amp --quiet)

for seed in 42 43 44; do
  "$PYTHON_BIN" train.py --model_type resunet --epochs 8 --min_epochs 6 \
    --patience 2 --lr 3e-4 --seed "$seed" "${common[@]}"
done

for seed in 42 43 44; do
  init="$OUT/resunet/seed${seed}/best_model.pth"
  "$PYTHON_BIN" train.py --model_type resunet_spc --epochs 4 --min_epochs 3 \
    --patience 2 --lr 5e-5 --seed "$seed" --init_checkpoint "$init" \
    --preserve_init_if_better --aux_warmup 0 --aux_ramp 1 "${common[@]}"
done

init="$OUT/resunet/seed42/best_model.pth"
for model in resunet_dsp resunet_spc_region; do
  "$PYTHON_BIN" train.py --model_type "$model" --epochs 4 --min_epochs 3 \
    --patience 2 --lr 5e-5 --seed 42 --init_checkpoint "$init" \
    --preserve_init_if_better --aux_warmup 0 --aux_ramp 1 "${common[@]}"
done

for model in unetpp deeplabv3p; do
  "$PYTHON_BIN" train.py --model_type "$model" --epochs 8 --min_epochs 6 \
    --patience 2 --lr 3e-4 --seed 42 "${common[@]}"
done
