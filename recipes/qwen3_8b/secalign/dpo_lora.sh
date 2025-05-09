#! /bin/bash

set -x

cd $(dirname "$0")

accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml ../../../scripts/run_dpo.py --config dpo_lora.yaml
