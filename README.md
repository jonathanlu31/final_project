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

### Using AgentDojo
```bash
python -m agentdojo.scripts.benchmark --model local --attack important_instructions
```
Use this to start a run without a self contained VLLM server. VLLM server should be started with
```bash
uv run vllm serve Qwen/Qwen2.5-7B-Instruct/ --tool-call-parser hermes --enable-auto-tool-choice
```

To run with support for automatic starting and stopping of VLLM server use the following script:
`AgentDojo/run_vllm.sh Qwen/Qwen2.5-7B-Instruct/`
</del>

## Project Structure

Mostly modeled off of alignment-handbook but the structure is messy right now
