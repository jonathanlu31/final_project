#! /bin/bash

cd "$(dirname "$0")"

CHECKPOINT_DIR=${1} # something like checkpoint-1500
BASE_MODEL=${2}
FINAL_MODEL_NAME=${3}

CHECKPOINT_DIR="$(realpath $CHECKPOINT_DIR)"
REPO_PATH="$(realpath $CHECKPOINT_DIR/..)"

accelerate merge-weights $CHECKPOINT_DIR/pytorch_model_fsdp_0 $CHECKPOINT_DIR/adapter
mv $CHECKPOINT_DIR/adapter/model.safetensors $CHECKPOINT_DIR/adapter/adapter_model.safetensors
cp $REPO_PATH/lora_init/adapter_config.json $CHECKPOINT_DIR/adapter/adapter_config.json

mkdir -p $REPO_PATH/$FINAL_MODEL_NAME
cp $REPO_PATH/tokenizer.json $REPO_PATH/$FINAL_MODEL_NAME/tokenizer.json
cp $REPO_PATH/tokenizer_config.json $REPO_PATH/$FINAL_MODEL_NAME/tokenizer_config.json
cp $REPO_PATH/special_tokens_map.json $REPO_PATH/$FINAL_MODEL_NAME/special_tokens_map.json
python merge_adapter.py $BASE_MODEL $CHECKPOINT_DIR/adapter $REPO_PATH/$FINAL_MODEL_NAME
