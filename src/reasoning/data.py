from accelerate import PartialState
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from reasoning.configs import DataArguments


def tokenize_reasoning_assistant_response(
    tokenizer: AutoTokenizer, assistant_message: dict[str, str]
) -> dict[str, str]:
    response = (
        assistant_message["reasoning_content"].rstrip()
        + "\n</think>\n\n"
        + assistant_message["content"].strip()
        + tokenizer.eos_token
    )
    if "R1" in tokenizer.name_or_path:
        return response
    elif "Qwen3" in tokenizer.name_or_path:
        return "<think>\n" + response
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer.name_or_path}")


def tokenize_secalign_data(
    tokenizer: AutoTokenizer, example: dict[str, str]
) -> dict[str, str]:
    new_example = {}
    new_example["prompt"] = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )
    new_example["chosen"] = tokenize_reasoning_assistant_response(
        tokenizer, example["chosen"][0]
    )
    new_example["rejected"] = tokenize_reasoning_assistant_response(
        tokenizer, example["rejected"][0]
    )
    return new_example


def format_pir_data(
    tokenizer: AutoTokenizer, example: dict[str, str], tokenize: bool = False
):
    last_message = example["messages"][-1]
    assert last_message["role"] == "assistant"
    conversation = tokenizer.apply_chat_template(
        example["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    assistant_response = tokenize_reasoning_assistant_response(tokenizer, last_message)

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
    elif "secalign" in data_args.dataset_path:
        with PartialState().local_main_process_first():
            full_ds = load_dataset(data_args.dataset_path)[data_args.split]
            full_ds = full_ds.map(
                lambda x: tokenize_secalign_data(tokenizer, x),
                remove_columns=full_ds.column_names,
                num_proc=4,
            )
    else:
        raise ValueError(f"Unknown dataset: {data_args.dataset_path}")

    if data_args.shuffle:
        full_ds = full_ds.shuffle(seed=seed)
    split_dataset = full_ds.train_test_split(test_size=data_args.test_size, seed=seed)

    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]
    return train_data, eval_data
