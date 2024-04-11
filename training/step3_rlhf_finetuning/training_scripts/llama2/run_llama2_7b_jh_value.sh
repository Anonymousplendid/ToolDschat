#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
set -x
set -e

ACTOR_MODEL_PATH=/tmp/Qwen-7B-rl
CRITIC_MODEL_PATH=/tmp/Qwen-7B-rl
ACTOR_ZERO_STAGE=2
CRITIC_ZERO_STAGE=2
OUTPUT=/home/sist/teslazhu20/model/chatgpt/rlaif/rlhf_step3_qwen_7b_t2

export TORCH_EXTENSIONS_DIR=/home/sist/teslazhu20/model/usercache/qwen
export HF_HOME=/home/sist/teslazhu20/model/hfmodel

export PATH=/opt/cuda/11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/opt/cuda/11.7/lib64:/opt/cuda/11.7/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/opt/cuda/11.7

export PATH=/opt/gnu/gcc/11.2.0/bin:/opt/gnu/gcc/11.2.0/libexec/gcc/x86_64-pc-linux-gnu/11.2.0${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/opt/gnu/gcc/11.2.0/lib64:/opt/gnu/gcc/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0:/opt/gnu/gmp/6.2.1/lib:/opt/gnu/mpc/1.2.1/lib:/opt/gnu/mpfr/4.1.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export OMP_NUM_THREADS=1
export NCCL_NET=Socket
export http_proxy=http://zhujh:zhujh19973@127.0.0.1:19973 https_proxy=http://zhujh:zhujh19973@127.0.0.1:19973
# export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

Actor_Lr=9.65e-6
Critic_Lr=5e-6

Lora_Lr=1e-5
answerlen=512
promptlen=2048
loradim=32
only_optimize_lora=false
batchsize=2
actorsleep=10000
gradientaccumulation=16
klcoff=0.05
klclip=2
POSITIONAL=()
OUTPUT=${OUTPUT}_lora${Lora_Lr}_promptlen${promptlen}_answerlen${answerlen}_loradim${loradim}_batchsize_${batchsize}_asleep${actorsleep}
OUTPUT=${OUTPUT}_gas${gradientaccumulation}_kf${klcoff}_kc${klclip}

if [[ $only_optimize_lora == "true" ]]; then
    OUTPUT=${OUTPUT}_only_optimize_lora
    POSITIONAL+=("--only_optimize_lora")
fi

mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --data_output_path /tmp/data_files_jh \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size ${batchsize} \
   --per_device_training_batch_size ${batchsize} \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len $answerlen \
   --max_prompt_seq_len ${promptlen} \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps ${gradientaccumulation} \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --actor_dropout 0.0 \
   --num_warmup_steps 10 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --actor_lora_dim $loradim \
   --critic_lora_dim $loradim \
   --lora_alpha $loradim \
   --critic_lora_module_name "h." \
   --actor_lora_module_name "h." \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
   --print_answers \
   --enable_test_mode \
   --test_stop_step 10000 \
   --compute_fp32_loss \
   --actor_lora_learning_rate $Lora_Lr \
   --critic_lora_learning_rate $Lora_Lr \
   --norm-advantage \
   --reward-scaling \
   --kl_ctl ${klcoff} \
   --clip_reward_value 1 \
   --kl_appoximation \
   --kl_clip ${klclip} \
   --dtype bf16 \
   --actor_sleep ${actorsleep} \
   "${POSITIONAL[@]}" | tee $OUTPUT/training.log
