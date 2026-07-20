#!/bin/bash

### NEED TO CHANGE ###
# Checkpoints are write-once/read-rarely (~1.8GB each, 30 epochs ~= 54GB), so
# park them on the roomy HDD and keep the NVMe for the hot training data.
save_checkpoint_path="/mnt/data/seer_checkpoints/"
root_dir="/home/khanh/data/libero_100"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
libero_path="/home/khanh/work/Seer/LIBERO"
wandb_entity="khanh-ng-18-havilab"
### NEED TO CHANGE ###
calvin_dataset_path="calvin/dataset/task_ABC_D"

node=1
node_num=1
# Single-GPU on GPU0. 2-GPU is NOT viable on this box: GPU1 (slot 03:00.0) drops
# off the PCIe bus under sustained dual-GPU load and wedges CUDA process-wide,
# needing a power cycle. Confirmed TWICE, incl. with NCCL_P2P_DISABLE=1 (that only
# delayed the crash from step 1 to ~step 25, so it's a GPU1 hardware fault, not the
# P2P software bug). Temp was NOT the cause (GPU1 ~69C, throttle is 95C).
# Upstream config is 8 GPUs x bs 10 x accum 8 = 640 samples/step. Keep that
# product on 1 GPU by scaling accumulation 8 -> 64.
export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 64 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 30 \
    --seed 42 \
    --batch_size 10 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type libero_pretrain \
    --root_dir ${root_dir} \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name libero_pretrain \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "pretrain" \
    --obs_pred \
    --sequence_length 11 \
    --action_pred_steps 3 \
    --future_steps 3 \
    --atten_goal 4 \
    --window_size 11 \
    --loss_image \
    --loss_action \
    --gripper_width \
    --atten_only_obs \
    --atten_goal_state \
    --mask_l_obs_ratio 0.5 \
    --warmup_epochs 1 \
    --libero_path ${libero_path} \
    --wandb_entity ${wandb_entity} \
    --report_to_wandb