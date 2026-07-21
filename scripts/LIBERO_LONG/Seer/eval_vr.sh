#!/bin/bash
# LIBERO-LONG (libero_10) eval on the vr box: 1x RTX 5070 laptop (sm_120), torch 2.7.1+cu128.
# Rendering is headless EGL (no osmesa libs on this box); eval_utils_libero.py uses setdefault
# so these exports take effect.
# Finetune checkpoints from shiba0 land in checkpoints/libero_ft_from_droid/ as <epoch>.pth.
# Usage: bash scripts/LIBERO_LONG/Seer/eval_vr.sh [ckpt_id ...]   # default: all *.pth in the dir

source .venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0

resume_from_checkpoint="checkpoints/libero_ft_from_droid/exp/libero_ft_from_droid_bf16"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
save_checkpoint_path="checkpoints/"
libero_path="/home/khanhnd61/work/LIBERO"

if [ "$#" -gt 0 ]; then
    pthlist=("$@")
else
    pthlist=($(ls ${resume_from_checkpoint}/*.pth 2>/dev/null | xargs -n1 basename | sed 's/\.pth$//' | sort -n))
fi
if [ ${#pthlist[@]} -eq 0 ]; then
    echo "no checkpoints found in ${resume_from_checkpoint}"
    exit 1
fi

dirname=$(basename "$resume_from_checkpoint")
LOG_DIR="checkpoints/eval_logs/${dirname}"
mkdir -p ${LOG_DIR}

node=1
node_num=1

for ckpt_id in "${pthlist[@]}"; do
    this_resume_from_checkpoint="${resume_from_checkpoint}/${ckpt_id}.pth"
    logfile="${LOG_DIR}/${ckpt_id}.log"

    python -m torch.distributed.run  --nnodes=${node} --nproc_per_node=${node_num} --master_port=10133 eval_libero.py \
        --traj_cons \
        --rgb_pad 10 \
        --gripper_pad 4 \
        --gradient_accumulation_steps 1 \
        --bf16_module "vision_encoder" \
        --vit_checkpoint_path ${vit_checkpoint_path} \
        --calvin_dataset "" \
        --workers 4 \
        --lr_scheduler cosine \
        --save_every_iter 50000 \
        --num_epochs 20 \
        --seed 42 \
        --batch_size 64 \
        --precision fp32 \
        --weight_decay 1e-4 \
        --num_resampler_query 6 \
        --run_name test \
        --transformer_layers 24 \
        --phase "evaluate" \
        --finetune_type "libero_10" \
        --save_checkpoint_path ${save_checkpoint_path} \
        --action_pred_steps 3 \
        --future_steps 3 \
        --sequence_length 7 \
        --obs_pred \
        --gripper_width \
        --eval_libero_ensembling \
        --libero_path ${libero_path} \
        --resume_from_checkpoint ${this_resume_from_checkpoint} | tee ${logfile}
done
