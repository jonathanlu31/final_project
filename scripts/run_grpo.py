import argparse
import os
from dataclasses import asdict
from datetime import datetime

import datasets
import torch
import yaml
from accelerate import PartialState
from datasets import load_dataset
from reasoning import (
    DataArguments,
    GRPOConfig,
    ModelArguments,
    correctness_reward_func,
    format_for_grpo,
    get_ifeval,
    get_redteam,
    loose_format_reward_func,
    pir_reward_func,
    reward_instruction_following,
    strict_format_reward_func,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import GRPOTrainer


def main(args):
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOConfig))
    parsed: tuple[ModelArguments, DataArguments, GRPOConfig] = parser.parse_yaml_file(
        args.config
    )
    model_args, data_args, training_args = parsed

    set_seed(training_args.seed)
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    if "redteam" in data_args.dataset_path:
        with PartialState().local_main_process_first():
            dataset = get_redteam(shuffle=True)
        reward_funcs = [
            strict_format_reward_func,
            loose_format_reward_func,
            correctness_reward_func,
        ]
    elif "ifeval" in data_args.dataset_path:
        with PartialState().local_main_process_first():
            dataset = get_ifeval(shuffle=True)
        reward_funcs = [reward_instruction_following]
    elif "pir" in data_args.dataset_path:

        def set_benign(example, flag):
            example["benign"] = flag
            return example

        benign = load_dataset("jonluj/pir_full", "grpo_benign", split="train")
        injected = load_dataset("jonluj/pir_full", "grpo_injected", split="train")

        benign = benign.map(
            lambda x: set_benign(x, True), cache_file_name=None, keep_in_memory=True
        )
        injected = injected.map(
            lambda x: set_benign(x, False), cache_file_name=None, keep_in_memory=True
        )
        benign = benign.map(
            format_for_grpo,
            remove_columns=benign.column_names,
            cache_file_name=None,
            keep_in_memory=True,
        )
        injected = injected.map(
            format_for_grpo,
            remove_columns=injected.column_names,
            cache_file_name=None,
            keep_in_memory=True,
        )
        dataset = datasets.concatenate_datasets([benign, injected])

        reward_funcs = [
            strict_format_reward_func,
            loose_format_reward_func,
            pir_reward_func,
        ]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = getattr(training_args, "run_prefix", "grpo")
    run_name = f"{run_prefix}_{timestamp}"
    training_args.run_name = run_name
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

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

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    config_dict = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }

    config_path = os.path.join(training_args.output_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args)
