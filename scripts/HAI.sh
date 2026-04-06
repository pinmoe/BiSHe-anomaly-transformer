#!/bin/bash
# HAI dataset: 59 sensors, industrial control system
# Baseline (Gaussian prior) vs DGR Prior comparison

# --- Baseline (already done, uncomment to re-run) ---
# python main.py \
#   --anormly_ratio 1.0 --num_epochs 10 --batch_size 256 \
#   --mode train --dataset HAI --data_path data/HAI \
#   --input_c 59 --output_c 59 \
#   --model_save_path checkpoints/AT_HAI_baseline \
#   --use_dgr_prior false

# python main.py \
#   --anormly_ratio 1.0 --num_epochs 10 --batch_size 256 \
#   --mode test --dataset HAI --data_path data/HAI \
#   --input_c 59 --output_c 59 \
#   --model_save_path checkpoints/AT_HAI_baseline \
#   --use_dgr_prior false

# --- DGR Prior ---
python main.py \
  --anormly_ratio 1.0 --num_epochs 10 --batch_size 256 \
  --mode train --dataset HAI --data_path data/HAI \
  --input_c 59 --output_c 59 \
  --model_save_path checkpoints/AT_HAI_dgr \
  --use_dgr_prior true

python main.py \
  --anormly_ratio 1.0 --num_epochs 10 --batch_size 256 \
  --mode test --dataset HAI --data_path data/HAI \
  --input_c 59 --output_c 59 \
  --model_save_path checkpoints/AT_HAI_dgr \
  --use_dgr_prior true
