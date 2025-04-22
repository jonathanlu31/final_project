# Final Project

## Installation Instructions

```bash
conda create -n reasoning python=3.12
pip install -e .
pip install --no-build-isolation flash-attn
wandb login
huggingface-cli login
```

## Usage

Use the scripts in the recipes folders

```bash
bash recipes/r1_qwen_7B/sft_lora.sh
```

<del>
Start training run
```bash
cd scripts
accelerate launch --config_file ../recipes/accelerate_configs/deepspeed_zero3.yaml run_grpo.py
```

Chat with a checkpoint
```bash
cd scripts
python chat.py --model <model_path>
```
</del>

## Project Structure

Mostly modeled off of alignment-handbook but the structure is messy right now
