#!/bin/bash
# 1-GPU DROID pretrain on the curated LIBERO+CALVIN subset (20,886 eps), staged
# on NVMe. Derived from single_node_language_cluster.sh. Key deltas vs the
# 32-GPU reference: 1 GPU, CUDA_VISIBLE_DEVICES=0, batch/accum keep global batch
# 2048 (1 x 8 x 256), curated dataset_info, checkpoints to HDD, intra-epoch
# saves (each epoch is multi-day on 1 GPU, so save_every_iter matters).
### NEED TO CHANGE ###
save_checkpoint_path="/mnt/data/seer_checkpoints/droid_curated/"
root_dir="/home/khanh/data/droid"        # NVMe; contains droid_success/{episodes,meta_info.h5,shape_info.h5}
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
wandb_entity="khanh-ng-18-havilab"
# Resume from the last intra-epoch checkpoint of the crashed run (h5py-leak OOM at
# ~step 66k). 60000.pth has model+optimizer+scheduler; resumes as epoch 1.
# Set to "" for a fresh run.
resume_ckpt="/mnt/data/seer_checkpoints/droid_curated/exp/droid_curated_1gpu/1460000.pth"
### NEED TO CHANGE ###

mkdir -p "$save_checkpoint_path"
export CUDA_VISIBLE_DEVICES=0
# FIX for dataloader-worker RSS leak -> OOM: the loader alloc/frees ~2.7MB image
# buffers while holding the previous one (DataLoader pattern), which fragments the
# glibc heap unboundedly. Pinning the mmap threshold low forces those buffers to
# mmap (returned to the OS on free, no heap fragmentation). Verified: +11GB/4000
# getitems -> -0.5GB. Fork-inherited by all workers.
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
torchrun --nnodes=1 --nproc_per_node=1 --master_port=10244 train.py \
    ${resume_ckpt:+--resume_from_checkpoint ${resume_ckpt}} \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 256 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset "" \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 20000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 8 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type "droid" \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name droid_curated_1gpu \
    --save_checkpoint_path ${save_checkpoint_path} \
    --except_lang \
    --transformer_layers 24 \
    --phase "pretrain" \
    --obs_pred \
    --action_pred_steps 3 \
    --sequence_length 11 \
    --window_size 11 \
    --future_steps 3 \
    --loss_action \
    --loss_image \
    --atten_goal 4 \
    --atten_goal_state \
    --atten_only_obs \
    --real_dataset_names "" \
    --root_dir ${root_dir} \
    --dataset_info droid_success_libero_calvin_curated \
    --warmup_epochs 3 \
    --wandb_entity ${wandb_entity} \
    --report_to_wandb