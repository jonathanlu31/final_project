import argparse
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime

import torch
import yaml
from peft import LoraConfig, get_peft_model
from reasoning import (
    DataArguments,
    DPOConfig,
    ModelArguments,
    get_dataset,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import DPOTrainer

logger = logging.getLogger(__name__)


def main(args):
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
    parsed: tuple[ModelArguments, DataArguments, DPOConfig] = parser.parse_yaml_file(
        args.config
    )
    model_args, data_args, training_args = parsed

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
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}"
    )
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")
    logger.info(f"Training/evaluation arguments {training_args}")

    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Setup model name and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = getattr(training_args, "run_prefix", "secalign")
    run_name = f"{run_prefix}_{timestamp}"
    training_args.run_name = run_name
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)

    ###############
    # Setup tokenizer
    ###############

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path, padding_side="left", use_fast=True
    )

    ###############
    # Setup dataset
    ###############

    train_data, eval_data = get_dataset(data_args, training_args.seed, tokenizer)

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
        use_cache=False,
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
        if training_args.local_rank == 0:
            model.print_trainable_parameters()
            model.save_pretrained(os.path.join(training_args.output_dir, "lora_init"))

    if training_args.local_rank == 0:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Saved tokenizer to {training_args.output_dir}")

    ref_model = model

    if model_args.peft_enabled:
        ref_model = None

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )

    trainer.train()

    logger.info(f"Training completed. Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    config_dict = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }

    config_path = os.path.join(training_args.output_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logger.info(f"Model and config saved to {training_args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args)
