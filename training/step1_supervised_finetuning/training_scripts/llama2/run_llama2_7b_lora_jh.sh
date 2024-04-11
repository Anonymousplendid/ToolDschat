#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
set -x 
set -e
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/data3/zhujh/model/chatgpt/rlhf_step1_llama2_7b_lora16
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT


export TORCH_EXTENSIONS_DIR=/data3/zhujh/model/usercache/t211c121
export HF_HOME=/data3/zhujh/model/hf

export PATH=/data3/zhujh/cuda/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/data3/zhujh/cuda/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/data3/zhujh/cuda/cuda-12.1

export OMP_NUM_THREADS=1
export http_proxy=http://zhujh:zhujh19973@192.168.161.36:19973 https_proxy=http://zhujh:zhujh19973@192.168.161.36:19973
export TRANSFORMERS_OFFLINE=1

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29512 main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --data_output_path /tmp/data_files_jh \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
   &> $OUTPUT/training.log
