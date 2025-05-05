from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from reasoning.configs import DataArguments


def format_pir_data(
    tokenizer: AutoTokenizer, example: dict[str, str], tokenize: bool = False
):
    last_message = example["messages"][-1]
    assert last_message["role"] == "assistant"
    conversation = tokenizer.apply_chat_template(
        example["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    if "R1" in tokenizer.name_or_path:
        assistant_response = (
            last_message["reasoning_content"].rstrip()
            + "\n</think>\n\n"
            + last_message["content"].strip()
            + tokenizer.eos_token
        )
    elif "Qwen3" in tokenizer.name_or_path:
        assistant_response = (
            "<think>\n"
            + last_message["reasoning_content"].rstrip()
            + "\n</think>\n\n"
            + last_message["content"].strip()
            + tokenizer.eos_token
        )

    conversation += assistant_response
    if tokenize:
        return {**tokenizer(conversation, add_special_tokens=False)}
    return {"text": conversation}


def format_medical_data(example):
    return {
        "formatted_text": f"{example['Question'].strip()}\n{example['Response'].strip()}"
    }


def get_dataset(
    data_args: DataArguments, seed: int, tokenizer: AutoTokenizer
) -> Dataset:
    """Loads and processes the dataset"""
    dataset_name = data_args.dataset_name

    if dataset_name == "pir":
        with PartialState().local_main_process_first():
            dataset = load_dataset(data_args.dataset_path)[data_args.split]
            dataset = dataset.map(
                lambda x: format_pir_data(tokenizer, x, tokenize=True),
                remove_columns=dataset.column_names,
                num_proc=4,
            )
            if data_args.shuffle:
                dataset = dataset.shuffle(seed=seed)
            split_dataset = dataset.train_test_split(
                test_size=data_args.test_size, seed=seed
            )
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]

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
