export WANDB_PROJECT=Qwen25-VL-7B-Video-GRPO
export WANDB_NAME=SUTD-filtered-f16-gpu4-numG-8-rewardRatio-10

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

mkdir -p /research/d1/gds/zhxing/projects_r1/experiments/Open-R1-Video/$WANDB_PROJECT/$WANDB_NAME

TRITON_CACHE_DIR=/research/d1/gds/zhxing/triton_cache CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    src/open_r1_video/grpo.py \
    --deepspeed /research/d1/gds/zhxing/projects_r1/Open-R1-Video/scripts/zero3_offload.json \
    --output_dir /research/d1/gds/zhxing/projects_r1/experiments/Open-R1-Video/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path /research/d1/gds/zhxing/projects_r1/models/Qwen2.5-VL-7B-Instruct \
    --dataset_name SUTD \
    --jsonl_path /research/d1/gds/zhxing/projects_r1/datasets/SUTD-R1/R2_train_rl.json \
    --max_prompt_length 4096 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 20 \
    --save_only_model true \
    --num_generations 8

# 如果您想训练 DVD 数据集，可以取消下面注释
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12352" \
#     src/open_r1_video/grpo.py \
#     --deepspeed scripts/zero3_offload.json \
#     --output_dir /research/d1/gds/zhxing/projects_r1/experiments/Open-R1-Video/$WANDB_PROJECT/$WANDB_NAME \
#     --model_name_or_path /research/d1/gds/zhxing/projects_r1/models/Qwen2.5-VL-7B-Instruct \
#     --dataset_name dvd \
#     --jsonl_path /research/d1/gds/zhxing/projects_r1/datasets/DVD-counting/train_dvd.jsonl \
#     --max_prompt_length 8192 \
#     --learning_rate 1e-6 \
#     --beta 0.1 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --logging_steps 1 \
#     --bf16 \
#     --torch_dtype bfloat16 \
#     --data_seed 42 \
#     --report_to wandb \
#     --gradient_checkpointing true \
#     --attn_implementation flash_attention_2 \
#     --num_train_epochs 1 \
#     --run_name $WANDB_NAME \
#     --save_steps 500 \
#     --save_only_model true \
#     --num_generations 8