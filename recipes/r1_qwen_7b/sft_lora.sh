#! /bin/bash

set -x

cd $(dirname "$0")

accelerate launch --config_file ../accelerate_configs/deepspeed_zero3.yaml ../../scripts/run_sft.py --config sft_lora.yaml
