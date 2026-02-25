#!/bin/bash


# --- Check arguments ---
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <num_trials> <target_hw> [optional: models]"
    echo "Example 1: $0 10000 \"llvm -num-cores 16\""
    echo "Example 2: $0 20000 \"nvidia/geforce-rtx-2080-ti\""
    echo "Example 3: $0 30000 \"nvidia/jetson-agx-orin-32gb\" \"bert,gpt2\""
    exit 1
fi
trials=$1
target_hw=$2
target_models=$3


# --- Check and prepare output directory ---
if [ -d "dataset" ]; then
    echo "Directory 'dataset' found. Checking for stray files..."

    # Check if there are any FILES in the root of dataset (ignoring directories)
    if [ -n "$(find dataset -maxdepth 1 -type f)" ]; then
        echo "Error: The 'dataset' directory contains stray files in its root."
        echo "Please remove these files manually or move them to a subdirectory before running to avoid data mixing."
        exit 1
    fi

    echo "Directory is clean. Running in APPEND mode."
else
    echo "Creating new 'dataset' directory..."
    mkdir -p dataset
fi


# --- Enable dataset construction mode ---
export TVMP_CONSTRUCT_DATASET=1


# --- Download ML models for the training dataset ---
# T5
if [ ! -f "./models/training/t5/model.onnx" ]; then
    wget https://github.com/onnx/models/raw/refs/heads/main/Natural_Language_Processing/skip/t5_Opset16_transformers/t5_Opset16.onnx
    mv t5_Opset16.onnx ./models/training/t5/model.onnx
fi

# DistilBERT
if [ ! -f "./models/training/distilbert/model.onnx" ]; then
    wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx
    mv model.onnx ./models/training/distilbert/model.onnx
fi

# Longformer
if [ ! -f "./models/training/longformer/model.onnx" ]; then
    wget https://github.com/onnx/models/raw/refs/heads/main/Natural_Language_Processing/longformer_Opset16_transformers/longformer_Opset16.onnx
    mv longformer_Opset16.onnx ./models/training/longformer/model.onnx
fi

# Segformer
if [ ! -f "./models/training/segformer/model.onnx" ]; then
    wget https://huggingface.co/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/model.onnx
    mv model.onnx ./models/training/segformer/model.onnx
fi

# DETR
if [ ! -f "./models/training/detr/model.onnx" ]; then
    wget https://huggingface.co/Xenova/detr-resnet-50/resolve/main/onnx/model.onnx
    mv model.onnx ./models/training/detr/model.onnx
fi


# --- Download ML models for the test dataset ---
# OPT
if [ ! -f "./models/test/opt/model.onnx" ]; then
    wget https://huggingface.co/Xenova/opt-125m/resolve/main/onnx/decoder_model.onnx
    mv decoder_model.onnx ./models/test/opt/model.onnx
fi

# GPT-2
if [ ! -f "./models/test/gpt2/decoder_model.onnx" ]; then
    wget https://huggingface.co/openai-community/gpt2-large/resolve/main/onnx/decoder_model.onnx
    mv decoder_model.onnx ./models/test/gpt2/decoder_model.onnx
fi
if [ ! -f "./models/test/gpt2/decoder_model.onnx_data" ]; then
    wget https://huggingface.co/openai-community/gpt2-large/resolve/main/onnx/decoder_model.onnx_data
    mv decoder_model.onnx_data ./models/test/gpt2/decoder_model.onnx_data
fi

# BERT
if [ ! -f "./models/test/bert/model.onnx" ]; then
    wget https://huggingface.co/dslim/bert-large-NER/resolve/main/onnx/model.onnx
    mv model.onnx ./models/test/bert/model.onnx
fi

# DINOv2-L
if [ ! -f "./models/test/dinov2_large/model.onnx" ]; then
    wget https://huggingface.co/Xenova/dinov2-large/resolve/main/onnx/model.onnx
    mv model.onnx ./models/test/dinov2_large/model.onnx
fi

# YOLOv4
if [ ! -f "./models/test/yolov4/model.onnx" ]; then
    wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/object_detection_segmentation/yolov4/model/yolov4.onnx
    mv yolov4.onnx ./models/test/yolov4/model.onnx
fi


# --- Construct the training dataset ---
echo "Processing training models..."
TOP_DIR="./models/training"

for dir in "$TOP_DIR"/*/; do
    model_name=$(basename "$dir")

    # If target_models is set, and the current model_name is NOT in the list, skip it
    if [ -n "$target_models" ] && [[ ",$target_models," != *",$model_name,"* ]]; then
        echo "Skipping $model_name (not in target list)"
        continue
    fi

    echo "Running $model_name on $target_hw"
    (
        cd "$dir"
        python3 main.py "$trials" "$target_hw"
    )

    if [ -z "$(find dataset -maxdepth 1 -type f)" ]; then
        echo "ERROR: No data generated for $model_name. Did main.py fail?"
    else
        target_dir="dataset/$model_name"
        mkdir -p "$target_dir"
        find dataset -maxdepth 1 -type f -exec mv {} "$target_dir/" \;
    fi
done


# --- Construct the test dataset ---
echo "Processing test models..."
TOP_DIR="./models/test"

for dir in "$TOP_DIR"/*/; do
    model_name=$(basename "$dir")

    # If target_models is set, and the current model_name is NOT in the list, skip it
    if [ -n "$target_models" ] && [[ ",$target_models," != *",$model_name,"* ]]; then
        echo "Skipping $model_name (not in target list)"
        continue
    fi

    echo "Running $model_name on $target_hw"
    (
        cd "$dir"
        python3 main.py "$trials" "$target_hw"
    )

    if [ -z "$(find dataset -maxdepth 1 -type f)" ]; then
        echo "ERROR: No data generated for $model_name. Did main.py fail?"
    else
        target_dir="dataset/$model_name"
        mkdir -p "$target_dir"
        find dataset -maxdepth 1 -type f -exec mv {} "$target_dir/" \;
    fi
done


# --- Disable dataset construction mode ---
export TVMP_CONSTRUCT_DATASET=0

echo "Dataset processing done. Results are in 'dataset'."
