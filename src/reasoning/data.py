from accelerate import PartialState
from datasets import Dataset, load_dataset

from reasoning.configs import DataArguments


def format_pir_data(example):
    messages = example["messages"]
    formatted_text = ""

    for message in messages:
        role = message["role"]
        content = message.get("content", "")

        if role == "system":
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    return {"formatted_text": formatted_text}


def format_medical_data(example):
    return {
        "formatted_text": f"{example['Question'].strip()}\n{example['Response'].strip()}"
    }


def get_dataset(data_args: DataArguments, seed: int) -> Dataset:
    dataset_name = data_args.dataset_name

    if dataset_name == "pir":
        with PartialState().local_main_process_first():
            dataset = load_dataset(data_args.dataset_path)
            split_dataset = dataset[data_args.split].train_test_split(
                test_size=data_args.test_size, seed=seed
            )
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]

        # Process the PIR dataset
        train_data = train_data.map(
            format_pir_data, remove_columns=train_data.column_names
        )
        eval_data = eval_data.map(
            format_pir_data, remove_columns=eval_data.column_names
        )

    elif dataset_name == "medical":
        with PartialState().local_main_process_first():
            dataset = load_dataset(data_args.dataset_path)
            split_dataset = dataset[data_args.split].train_test_split(
                test_size=data_args.test_size, seed=seed
            )
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]

        # Process the medical dataset
        train_data = train_data.map(
            format_medical_data, remove_columns=train_data.column_names
        )
        eval_data = eval_data.map(
            format_medical_data, remove_columns=eval_data.column_names
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_data, eval_data
