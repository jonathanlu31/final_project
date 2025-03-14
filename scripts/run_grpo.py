import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from accelerate import PartialState
from datetime import datetime

from utils import (
    get_redteam,
    correctness_reward_func,
    strict_format_reward_func,
    loose_format_reward_func,
)

def main():
    with PartialState().local_main_process_first():
        dataset = get_redteam(shuffle=True)

    ## config

    model_name = "/scratch/public_models/huggingface/Qwen/Qwen2.5-7B-Instruct"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"qwen_rules_{timestamp}"
    output_dir = f"/data/jonathan/models/reasoning/{run_name}"

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=3,
        save_steps=100,
        max_grad_norm=0.5,
        report_to="wandb",
        log_on_each_node=False,
        use_vllm=True,
        vllm_max_model_len=2048,
        vllm_gpu_memory_utilization=0.5
    )

    tokenizer = AutoTokenizer.from_pretrained("tokenizers/qwen2.5-7b-instruct", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    ).to("cuda")


    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            strict_format_reward_func,
            loose_format_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()