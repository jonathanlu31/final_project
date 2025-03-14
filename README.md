# Final Project

## Installation Instructions

```bash
conda create -n reasoning python=3.12
pip install -r requirements.txt
wandb login
huggingface-cli login
```

## Usage

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

## Project Structure

Mostly modeled off of alignment-handbook but the structure is messy right now
