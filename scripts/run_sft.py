import argparse
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime

import torch
import yaml
from peft import LoraConfig, get_peft_model
from reasoning import DataArguments, ModelArguments, SFTConfig, get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def main(args):
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse_yaml_file(args.config)

    set_seed(training_args.seed)
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel("INFO")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Setup dataset
    ###############

    train_data, eval_data = get_dataset(data_args, training_args.seed)

    # Setup model name and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = getattr(training_args, "run_prefix", f"sft_{data_args.dataset_name}")
    run_name = f"{run_prefix}_{timestamp}"
    training_args.run_name = run_name
    output_dir = os.path.join(training_args.output_dir, run_name)

    ###############
    # Setup tokenizer
    ###############

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load model
    ###############

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_load_params = {
        "torch_dtype": torch_dtype,
        "attn_implementation": model_args.attn_implementation,
        "trust_remote_code": model_args.trust_remote_code,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=model_args.device_map,
        **model_load_params,
    )

    if model_args.peft_enabled:
        logger.info("Initializing LoRA for training")

        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.bias,
            task_type="CAUSAL_LM",
            target_modules=model_args.target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        formatting_func=lambda x: x["formatted_text"],
    )

    trainer.train()

    logger.info(f"Training completed. Saving model to {output_dir}")
    trainer.save_model(output_dir)

    config_dict = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }

    config_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logger.info(f"Model and config saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args)
