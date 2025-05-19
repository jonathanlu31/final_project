#! /bin/bash

cd $(dirname "$0")

CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model jonluj/qwen8b_secreason --max_model_len 2048 > vllm_log.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --config_file ../accelerate_configs/grpo_deepspeed_zero3.yaml \
    --num_processes 3 ../../scripts/run_grpo.py \
    --config grpo.yaml
