import argparse
import torch
import os
import yaml
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def format_pir_data(example):
    messages = example['messages']
    formatted_text = ""
    
    for message in messages:
        role = message['role']
        content = message.get('content', '')
        
        if role == "system":
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return {"formatted_text": formatted_text}

def format_medical_data(example):
    return {"formatted_text": f"{example['Question'].strip()}\n{example['Response'].strip()}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['training']['seed'])
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    
    if dataset_name == "pir":
        with PartialState().local_main_process_first():
            dataset = load_dataset(dataset_config['path'], dataset_config.get('subset', None))
            split_dataset = dataset[dataset_config.get('split', 'train')].train_test_split(
                test_size=dataset_config.get('test_size', 0.05), 
                seed=config['training']['seed']
            )
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]
        
        # Process the PIR dataset
        train_data = train_data.map(format_pir_data, remove_columns=train_data.column_names)
        eval_data = eval_data.map(format_pir_data, remove_columns=eval_data.column_names)
        
    elif dataset_name == "medical":
        with PartialState().local_main_process_first():
            dataset = load_dataset(dataset_config['path'], dataset_config.get('subset', None))
            split_dataset = dataset[dataset_config.get('split', 'train')].train_test_split(
                test_size=dataset_config.get('test_size', 0.05), 
                seed=config['training']['seed']
            )
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]
        
        # Process the medical dataset
        train_data = train_data.map(format_medical_data, remove_columns=train_data.column_names)
        eval_data = eval_data.map(format_medical_data, remove_columns=eval_data.column_names)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Setup model name and output directory
    model_config = config['model']
    model_name = model_config['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = config['training'].get('run_prefix', f"sft_{dataset_name}")
    run_name = f"{run_prefix}_{timestamp}"
    output_dir = os.path.join(config['training'].get('output_dir', './outputs'), run_name)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side=model_config.get('padding_side', 'left'),
        trust_remote_code=model_config.get('trust_remote_code', False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_dtype = getattr(torch, model_config.get('dtype', 'float16'))
    model_load_params = {
        'torch_dtype': model_dtype,
        'attn_implementation': model_config.get('attn_implementation', 'sdpa'),
        'trust_remote_code': model_config.get('trust_remote_code', False),
    }
    
    if model_config.get('load_in_8bit', False):
        model_load_params['load_in_8bit'] = True
    elif model_config.get('load_in_4bit', False):
        model_load_params['load_in_4bit'] = True
        model_load_params['quantization_config'] = {
            'bnb_4bit_compute_dtype': model_dtype,
            'bnb_4bit_quant_type': model_config.get('bnb_4bit_quant_type', 'nf4'),
            'bnb_4bit_use_double_quant': model_config.get('bnb_4bit_use_double_quant', True),
        }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=model_config.get('device_map', 'auto'),
        **model_load_params
    )

    if config.get('lora', {}).get('enabled', False):
        lora_config = config['lora']
        logger.info("Initializing LoRA for training")
        
        if model_config.get('load_in_8bit', False) or model_config.get('load_in_4bit', False):
            model = prepare_model_for_kbit_training(model)
            
        peft_config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 16),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            bias=lora_config.get('bias', 'none'),
            task_type=lora_config.get('task_type', 'CAUSAL_LM'),
            target_modules=lora_config.get('target_modules', None),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    training_config = config['training']
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=float(training_config.get('learning_rate', 5e-6)),
        weight_decay=float(training_config.get('weight_decay', 0.1)),
        warmup_ratio=float(training_config.get('warmup_ratio', 0.1)),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        logging_steps=int(training_config.get('logging_steps', 5)),
        bf16=training_config.get('bf16', False),
        fp16=training_config.get('fp16', False),
        per_device_train_batch_size=int(training_config.get('per_device_train_batch_size', 2)),
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 2)),
        num_train_epochs=int(training_config.get('num_train_epochs', 3)),
        save_steps=int(training_config.get('save_steps', 100)),
        save_total_limit=int(training_config.get('save_total_limit', 2)),
        evaluation_strategy=training_config.get('evaluation_strategy', 'epoch'),
        max_grad_norm=float(training_config.get('max_grad_norm', 1.0)),
        report_to=training_config.get('report_to', 'wandb'),
        log_on_each_node=training_config.get('log_on_each_node', False),
    )
    
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
    
    with open(os.path.join(output_dir, 'training_config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    logger.info(f"Model and config saved to {output_dir}")

if __name__ == "__main__":
    main()
