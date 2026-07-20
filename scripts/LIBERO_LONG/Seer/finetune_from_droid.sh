#!/bin/bash
# LIBERO-LONG (libero_10) finetune starting from the curated-DROID pretrain.
# See docs/pretrain-droid-curated.md §5. Single 12 GB RTX 3060.

source /home/khanh/work/Seer/.venv/bin/activate

save_checkpoint_path="/mnt/data/seer_checkpoints/libero_ft_from_droid/"
root_dir="/mnt/khanh/libero_100"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
finetune_from_pretrained_ckpt="/mnt/data/seer_checkpoints/droid_curated/exp/droid_curated_1gpu/2760000.pth"
libero_path="/home/khanh/work/Seer/LIBERO"
calvin_dataset_path="calvin/dataset/task_ABC_D"   # unused for libero_finetune, arg is required

# glibc arena tuning - patch #3 in docs/pretrain-droid-curated.md (dataloader heap fragmentation)
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export WORKER_RECYCLE_EVERY=2000

# global batch 512 preserved: 1 GPU x bs 8 x accum 64 (upstream: 8 x 16 x 4)
node=1
node_num=1
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 64 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 4 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 8 \
    --precision amp_bfloat16 \
    --learning_rate 1e-3 \
    --save_checkpoint \
    --finetune_type libero_finetune \
    --root_dir ${root_dir} \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name libero_ft_from_droid_bf16 \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "finetune" \
    --obs_pred \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --loss_image \
    --loss_action \
    --reset_action_token \
    --reset_obs_token \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint 12 \
    --gripper_width \
    --warmup_epochs 3 \
    --libero_path ${libero_path} \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
    --report_to_wandb
