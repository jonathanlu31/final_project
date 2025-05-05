from transformers import AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("base_model", type=str)
parser.add_argument("lora_adapter", type=str, help="Path to lora adapter or huggingface repo. The folder must have adapter_config.json and adapter_model.safetensors or adapter_model.bin")
parser.add_argument("output_dir", type=str)
args = parser.parse_args()


base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
model = PeftModel.from_pretrained(base_model, args.lora_adapter)
model = model.merge_and_unload()

model.save_pretrained(args.output_dir)
