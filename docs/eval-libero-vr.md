# LIBERO-LONG Eval on the vr Box

Eval machine for the DROID-pretrain (shiba1) -> LIBERO finetune (shiba0) pipeline.
Hardware: 1x RTX 5070 Laptop 8 GB (Blackwell sm_120), CUDA 12.8 driver.

## Environment (already set up)

- Venv: `.venv` in the repo root (python 3.10, uv).
- torch 2.7.1+cu128 - NOT the doc's 2.2.0+cu121. sm_120 needs cu128 wheels; older
  torch fails with "no kernel image is available".
- mujoco pinned to 2.3.7 - mujoco 3.x breaks robosuite 1.4.0 (`mj_fullM` signature).
- transformers 4.40.2, numpy 1.23.1, timm 0.9.16 per Seer requirements.
- LIBERO cloned at `/home/khanhnd61/work/LIBERO`, installed editable in setuptools
  compat mode (`--config-settings editable_mode=compat`; PEP-660 mode maps nothing
  because top-level `libero/` has no `__init__.py`).
- Rendering: headless EGL via the NVIDIA driver (no osmesa libs on this box, no sudo).
  `eval_vr.sh` exports `MUJOCO_GL=egl`.
- `checkpoints/vit_mae/mae_pretrain_vit_base.pth` and `checkpoints/clip/ViT-B-32.pt`
  downloaded and verified.

## Repo patches for this box (safe on shiba machines too)

- `utils/eval_utils_libero.py`: `MUJOCO_GL` / `PYOPENGL_PLATFORM` now `setdefault`
  instead of hardcoded osmesa, and init-state `torch.load(..., weights_only=False)`.
- `eval_libero.py`, `models/seer_model.py`: `torch.load(..., weights_only=False)`.
  Needed because torch >= 2.6 defaults to `weights_only=True`, which rejects the
  numpy arrays in LIBERO `.pruned_init` files and the argparse.Namespace in the MAE
  checkpoint. The kwarg exists since torch 1.13, so shiba's torch 2.2 is unaffected.

## Running the eval

1. Copy finetune checkpoints from shiba0 into
   `checkpoints/libero_ft_from_droid/exp/libero_ft_from_droid_bf16/` (train.py nests
   `exp/<run_name>/` under save_checkpoint_path; intra-epoch saves are `<step>.pth`,
   end-of-epoch saves `<epoch>.pth`).
2. `bash scripts/LIBERO_LONG/Seer/eval_vr.sh` - evaluates every `*.pth` in that dir,
   or pass ids: `bash scripts/LIBERO_LONG/Seer/eval_vr.sh 18 19`.
3. Logs land in `checkpoints/eval_logs/libero_ft_from_droid/<id>.log`; per-task
   success rates are printed at the end of each log.

Notes:
- Single GPU, 200 episodes (10 tasks x 20) run sequentially, up to 600 steps each.
  Expect hours per checkpoint - start with the last epoch or two, not all eight.
- The finetune ckpt stores only trainable params ("module."-prefixed, strict=False
  load), so the MAE ViT and CLIP files are required at eval time.
