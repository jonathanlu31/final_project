#! /bin/bash

set -x

cd "$(dirname "$0")"

CHECKPOINT_DIR=${1} # something like checkpoint-1500
BASE_MODEL=${2}
FINAL_MODEL_NAME=${3}

accelerate merge-weights $CHECKPOINT_DIR --output_dir $CHECKPOINT_DIR/adapter
mv $CHECKPOINT_DIR/adapter/model.safetensors $CHECKPOINT_DIR/adapter/adapter_model.safetensors
cp $CHECKPOINT_DIR/../lora_init/adapter_config.json $CHECKPOINT_DIR/adapter/adapter_config.json

cp $CHECKPOINT_DIR/../tokenizer.json $CHECKPOINT_DIR/../merged/tokenizer.json
cp $CHECKPOINT_DIR/../tokenizer_config.json $CHECKPOINT_DIR/../merged/tokenizer_config.json
cp $CHECKPOINT_DIR/../special_tokens_map.json $CHECKPOINT_DIR/../merged/special_tokens_map.json
python merge_adapter.py $BASE_MODEL $CHECKPOINT_DIR/adapter $CHECKPOINT_DIR/../$FINAL_MODEL_NAME
