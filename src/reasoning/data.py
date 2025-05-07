from accelerate import PartialState
from datasets import Dataset, concatenate_datasets, load_dataset
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


def get_dataset(
    data_args: DataArguments, seed: int, tokenizer: AutoTokenizer
) -> Dataset:
    """Loads and processes the dataset"""
    if "pir" in data_args.dataset_path:
        with PartialState().local_main_process_first():
            datasets = [
                load_dataset(data_args.dataset_path, config)[data_args.split]
                for config in data_args.dataset_subsets
            ]
            datasets = [
                ds.map(
                    lambda x: format_pir_data(tokenizer, x, tokenize=True),
                    remove_columns=ds.column_names,
                    num_proc=4,
                )
                for ds in datasets
            ]
            full_ds = concatenate_datasets(datasets)

            if data_args.shuffle:
                full_ds = full_ds.shuffle(seed=seed)
            split_dataset = full_ds.train_test_split(
                test_size=data_args.test_size, seed=seed
            )
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]
    else:
        raise ValueError(f"Unknown dataset: {data_args.dataset_path}")
    return train_data, eval_data
