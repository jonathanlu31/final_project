import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
from trl import SFTTrainer
from datetime import datetime
from datasets import load_dataset
from accelerate import PartialState
import logging
import sys


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def formatting_func(example): # specific for medical for now
    return f"{example['Question'].strip()}\n{example['Response'].strip()}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="medical", choices=["medical"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    set_seed(42)

    if args.dataset == "medical":
        with PartialState().local_main_process_first():
                dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT")
        run_prefix = "sft_qwen_medical"
        train_dataset = dataset["train"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_prefix}_{timestamp}"
    output_dir = f"./outputs/{run_name}"

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        device_map=None,
    ).to("cuda")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=False,
        fp16=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="epoch",
        max_grad_norm=1.0,
        report_to="wandb",
        log_on_each_node=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=formatting_func,
        max_seq_length=2048,
        packing=True,
    )
    trainer.train()
    logger.info("Training completed")

if __name__ == "__main__":
    main()

