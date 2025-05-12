import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from accelerate import PartialState
from datetime import datetime
from reasoning import (
    get_redteam,
    get_ifeval,
    reward_instruction_following,
    correctness_reward_func,
    strict_format_reward_func,
    loose_format_reward_func,
    format_for_grpo,
    pir_reward_func
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="redteam", choices=["redteam", "ifeval", "pir"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    if args.dataset == "redteam":
        with PartialState().local_main_process_first():
            dataset = get_redteam(shuffle=True)
        reward_funcs = [
            strict_format_reward_func,
            loose_format_reward_func,
            correctness_reward_func,
        ]
        run_prefix = "qwen_redteam"
    elif args.dataset == "ifeval":
        with PartialState().local_main_process_first():
            dataset = get_ifeval(shuffle=True)
        reward_funcs = [reward_instruction_following]
        run_prefix = "qwen_ifeval"
    elif args.dataset == "pir":
        import datasets
        from datasets import load_dataset
        
        def set_benign(example, flag):
            example["benign"] = flag
            return example

        benign = load_dataset("jonluj/pir_full", "grpo_benign", split="train")
        injected = load_dataset("jonluj/pir_full", "grpo_injected", split="train")
        
        benign = benign.map(lambda x: set_benign(x, True))
        injected = injected.map(lambda x: set_benign(x, False))
        dataset = datasets.concatenate_datasets([benign, injected])
        
        dataset = dataset.map(
            format_for_grpo,
            remove_columns=dataset.column_names,
        )
        
        reward_funcs = [
            strict_format_reward_func,
            loose_format_reward_func,
            pir_reward_func
        ]
        
        run_prefix = "qwen_pir"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_prefix}_{timestamp}"
    output_dir = f"./outputs/{run_name}"

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
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        num_generations=24,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=3,
        save_steps=100,
        max_grad_norm=1.0,
        report_to="wandb",
        log_on_each_node=False,
        use_vllm=True,
        # use_liger_loss=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
        use_cache=False,
    ).to("cuda")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()

