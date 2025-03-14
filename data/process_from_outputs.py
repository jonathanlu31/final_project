"""Create prompts for GRPO based on output results from Qwen2.5 7B Instruct on RuLES Redteam. This is necessary because some of the test cases are multiturn so I prefill with a response from the original model and only grade the last response"""

from collections import defaultdict
import json
import os

from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../scripts/tokenizers/qwen2.5-7b-instruct")

def format_messages(messages):
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

def load_dataset():
    """Load all *.jsonl test files from the specified test_suite package."""
    dataset = defaultdict(dict)
    skipped_count = 0

    files = [
        f
        for f in os.listdir("redteam/Qwen2.5-7B-Instruct")
        if f.endswith(".jsonl")
    ]
    files = sorted(files)
    for file in files:
        full_name = os.path.splitext(file)[0]

        with open(f"redteam/Qwen2.5-7B-Instruct/{file}") as f:
            testcases = [json.loads(line) for line in f.readlines()]
            without_last_assistant = []
            for testcase in testcases:
                new_tc = testcase.copy()
                new_tc["messages"] = new_tc["messages"][:-1]
                # Skip testcases where the model input would be too long. This occurs for mainly base64 testcases and developer mode testcases
                if len(tokenizer.apply_chat_template(new_tc["messages"], add_generation_prompt=True)) > 1024:
                    skipped_count += 1
                    continue
                without_last_assistant.append(new_tc)

            dataset[full_name] = without_last_assistant
            
    print(skipped_count)
    return dataset

full_dataset = load_dataset()

hf_datasets = {
    task_name: Dataset.from_list(full_dataset[task_name]) for task_name in full_dataset
}

for name, dataset in hf_datasets.items():
    dataset.push_to_hub("jonluj/rules_redteam_qwen2.5-7b", private=False, config_name=name)
