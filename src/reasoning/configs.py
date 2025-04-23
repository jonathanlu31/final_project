from dataclasses import dataclass, field

import trl


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    torch_dtype: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The path to the tokenizer. Useful if you want to use a different tokenizer to the one stored in `model_name_or_path`."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code when loading a model."}
    )
    attn_implementation: str | None = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can use --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    device_map: str | None = field(
        default=None,
        metadata={
            "help": (
                "Which devices to put the model on. If None, will default to 'auto'."
            )
        },
    )
    peft_enabled: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: int | None = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: int | None = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: float | None = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    bias: str | None = field(
        default="none",
    )
    target_modules: list[str] | None = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    modules_to_save: list[str] | None = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str | None = field(
        default=None,
        metadata={"help": ("The name of the dataset to use.")},
    )
    dataset_path: str | None = field(
        default=None,
        metadata={"help": ("The path to the dataset to use.")},
    )
    split: str | None = ("train",)
    test_size: float | None = 0.05
    shuffle: bool = True


@dataclass
class SFTConfig(trl.SFTConfig):
    run_prefix: str | None = None
    wandb_project: str | None = None
