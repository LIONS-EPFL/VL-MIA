#!/usr/bin/bash
lr=1e-2 
epochs=60 
batch_size=8 
dataname=eminst

PRETRAINED_PATH="" # path to pre-trained checkpoint
CONFIG="./ft.yaml"
OUTPUT_DIR="./checkpoints_ft/eminst/"

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=1,2,3,7
python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=4 --use_env \
 main_finetune.py --data_config "$CONFIG" --batch_size $batch_size \
 --epochs $epochs --warmup_epochs 1 --lr $lr --blr 1e-2 --weight_decay 0.02 \
 --llama_path "$LLAMA_PATH" \   # llama path
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH" 
