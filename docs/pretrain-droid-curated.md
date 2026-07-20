# DROID Curated Pretraining - Summary & LIBERO Finetune Handoff

**Purpose:** hand off a DROID-pretrained Seer checkpoint so the LIBERO-LONG finetune can be run
on a different machine. Everything needed to reproduce, transfer, and finetune is below.

Written 2026-07-20. Pretraining machine: 1× RTX 3090 24GB, 62 GB RAM, Ubuntu.

---

## 1. TL;DR - what you're getting

A Seer checkpoint pretrained on a **curated 20,886-episode DROID subset**, selected to match
the LIBERO + CALVIN downstream task distribution.

| | |
|---|---|
| **Checkpoint** | `2760000.pth` (773 MB) - use the **newest** `<step>.pth` at transfer time |
| **Location** | `/mnt/data/seer_checkpoints/droid_curated/exp/droid_curated_1gpu/` |
| **Trained to** | epoch 5/20, global step ~2.77M, train loss **0.0856** |
| **Trainable params** | 67.69 M (frozen encoders NOT included - see §5) |
| **Base repo** | `InternRobotics/Seer`, with 4 required patches (§4) |
| **wandb** | project `seer`, entity `khanh-ng-18-havilab`, runs `m53vhjjx -> trglkys7 -> fn8u0p4b -> bmwt1tmc` |

> **This is a partial, scaled-down pretrain - a weak-but-real signal, not a paper reproduction.**
> See §7 for the honest limitations before interpreting any downstream number.

---

## 2. The checkpoint

### Format
```
torch.load("2760000.pth") ->
  {
    "epoch": 4,                    # 0-indexed epoch the ckpt was saved in
    "model_state_dict": {...},     # 400 tensors, ALL prefixed "module." (DDP)
    "optimizer_state_dict": {...},
    "lr_scheduler_state_dict": {...},
  }
```
- Filename = **global step**, saved every 20,000 steps (~2.7 h).
- `model_state_dict` contains **only trainable params (67.69 M)** - `get_checkpoint()` strips
  everything with `requires_grad=False`.
- `transformer_backbone_position_embedding` has shape `(1, 11, 1, 384)` (sequence_length 11).

### Transfer
```bash
# on the pretraining machine, pick the newest checkpoint
ls -t /mnt/data/seer_checkpoints/droid_curated/exp/droid_curated_1gpu/*.pth | head -1

# copy it (773 MB) to the finetune machine
rsync -avP /mnt/data/seer_checkpoints/droid_curated/exp/droid_curated_1gpu/2760000.pth \
      <user>@<host>:/path/to/checkpoints/droid_pretrain/
```
Only the single `.pth` is needed - not the optimizer state, though it's harmless to keep.

---

## 3. What was pretrained

### 3.1 Data - curated DROID subset

Source: DROID `droid_success` converted to Seer per-step format (`episodes/<id>/steps/<NNNN>/`
with `image_primary.jpg`, `image_wrist.jpg`, `image_3.jpg`, `other.h5`).

Curation, from the 37,160-episode **language-annotated** pool (`droid_success_languaged_0803`):

1. Extracted the language instruction for all 37,160 episodes (all had clean English text).
2. Scored each episode by **CLIP ViT-B/32 text similarity** against two target sets, kept
   separate so the larger set couldn't dominate:
   - **LIBERO**: 113 unique `(:language ...)` strings from all `bddl_files/` suites.
   - **CALVIN**: 34 canonical task instructions derived from `utils/enrich_lang_annotations.json`.
3. Final score = `0.6 * norm(max(clip_libero, clip_calvin)) + 0.4 * norm(lexical)`, with a
   lexical boost for CALVIN-signature tokens (`block, slider, drawer, rotate, push, pull, lift,
   stack, led, lightbulb, switch, button, slide, cabinet, turn`).
4. Took the top-scoring episodes until a 300 GB budget filled.

**Result: 20,886 episodes / 4,915,088 steps / ~300 GB**, 21% CALVIN-leaning, spread across all
10 DROID collection labs (IRIS, AUTOLab, ILIAD, CLVR, IPRL, RAIL, PennPAL, REAL, GuptaLab, RAD).
Median episode 187 steps.

- Index file: **`data_info/droid_success_libero_calvin_curated.json`** - format `[[episode_id, num_steps], ...]`
- Top matches are near-identical to LIBERO/CALVIN tasks ("Close the bottom drawer of the cabinet",
  "Turn on the stove", "Open the drawer", "Put the bowl on the plate").
- Dropped: laundry/cloth, plush toys, and other off-domain manipulations.
- Vocabulary-limited: CALVIN's `rotate the block` (23) and `lift the colored block` (25) are
  rare - DROID simply doesn't contain many of those.

Data was staged on **NVMe** for training. On the HDD the loader is I/O-bound (GPU 0% util);
on NVMe it is compute-bound (GPU ~100%). If you re-run pretraining, stage to SSD/NVMe.

### 3.2 Pretraining config

`scripts/REAL/single_node_1gpu_curated.sh` (derived from `single_node_language_cluster.sh`).
Key deltas vs the 32-GPU reference run:

| Arg | Reference (4 nodes × 8 GPU) | **This run (1 GPU)** |
|---|---|---|
| GPUs | 32 | **1** |
| `batch_size` | 32 | **8** (16 OOMs on 24 GB) |
| `gradient_accumulation_steps` | 2 | **256** |
| **global batch** | 2048 | **2048** (preserved) |
| `dataset_info` | `droid_success_languaged_0803` | **`droid_success_libero_calvin_curated`** |
| `save_every_iter` | 20000 | 20000 (re-enabled, see §4) |

Unchanged: `lr 1e-4` cosine, `warmup_epochs 3`, `weight_decay 1e-4`, `precision fp32`,
`sequence_length 11`, `window_size 11`, `action_pred_steps 3`, `future_steps 3`,
`atten_goal 4 --atten_goal_state --atten_only_obs`, `--except_lang`, `--obs_pred`,
`--loss_action --loss_image`, `transformer_layers 24`, `num_resampler_query 6`, `seed 42`.

Throughput: **2.02 it/s**, ~13.4 GB VRAM, ~3.4 days/epoch.

---

## 4. REQUIRED repo patches

The upstream repo does **not** run DROID training correctly out of the box. All four of these
were needed; **apply them on the finetune machine too** (the last three matter for any long run).

| # | File | Problem | Fix |
|---|---|---|---|
| 1 | `utils/data_utils.py` (~L1264, ~L2986) | `self.client = Client(self.conf_path)` called **unconditionally** in `BaseDroidDataset` / `RealDataset`. `Client` is SenseTime's internal Ceph client (`petrel_client`), unavailable off-cluster -> `NameError` at dataset init. | Guard with `if data_in_ceph:` (matches how `CalvinDataset` already does it). |
| 2 | `utils/train_utils.py` (~L253) | The intra-epoch `save_every_iter` checkpoint block was **entirely commented out**, so with multi-day epochs the first checkpoint is days away. | Uncomment/re-enable. Also **remove the local `import os`** inside `train_one_epoch_calvin` - it shadows the module-level import (`UnboundLocalError`). |
| 3 | launch script | DataLoader leaked ~127 MB/min -> **OOM-killed a worker after ~9 h**. Cause: glibc heap fragmentation from ~2.7 MB image buffers allocated while the previous batch is still held. | `export MALLOC_MMAP_THRESHOLD_=131072` and `MALLOC_TRIM_THRESHOLD_=131072` before `torchrun` (fork-inherited by workers). Verified: +11 GB/4000 getitems -> −0.5 GB. Costs ~6% throughput. |
| 4 | `utils/data_utils.py` + `utils/train_utils.py` | Residual ~20 MB/min worker leak -> OOM at ~26 h. **Root cause: `persistent_workers=True`** - workers live for the entire run, and our epochs are ~3.4 days, so they never recycle and accumulate forever. | Set **`persistent_workers=False`**, and recycle workers mid-epoch: helper `_batch_iter_with_worker_recycling(loader, epoch, num_batches, recycle_every)` wraps the train loop, recreating the loader iterator every `WORKER_RECYCLE_EVERY` (default **2000**) batches. Verified: worker private-dirty sawtooths 6 GB -> 1.4 GB per recycle and stays bounded; system memory flat at 5 GiB over 59 h (was climbing to 53 GiB -> OOM). |

**Operational gotcha:** after killing a run, confirm `nvidia-smi --query-compute-apps` is empty
before relaunching - a straggler worker can hold ~13 GB VRAM and CUDA-OOM the new process.

---

## 5. Running the LIBERO finetune from this checkpoint

### 5.1 Prerequisites on the finetune machine

1. Seer repo + venv (python 3.10, torch 2.2.0+cu121, transformers 4.40.2), LIBERO installed.
2. **MAE ViT-B/16 checkpoint** - `mae_pretrain_vit_base.pth` into `checkpoints/vit_mae/`.
   ```bash
   wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -P checkpoints/vit_mae/
   ```
   > The DROID checkpoint contains **only trainable params** - the frozen MAE vision encoder and
   > CLIP text encoder are **not** in it and must be present separately. CLIP ViT-B/32
   > auto-downloads on first run.
3. **LIBERO-10 data** converted to Seer per-step format, plus its index
   `data_info/libero_10_converted.json` (format `[[episode_dir, num_steps], ...]`).
   `BaseLiberoDataset` hard-asserts `./data_info/<dataset_info>.json` exists.
   Convert with `utils/convert_libero_per_step.py` (paths are hardcoded near the bottom - edit
   `dataset_name`, `src_dir`, `tgt_dir`), or copy the converted data across.
4. Apply patches **#1, #2, #4** from §4 (and #3 if training long).

### 5.2 Edit `scripts/LIBERO_LONG/Seer/finetune.sh`

Upstream ships placeholders and `node_num=8`. Change:

```bash
save_checkpoint_path="/your/path/libero_finetune_from_droid/"
root_dir="/parent/dir/of/libero_10_converted"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
finetune_from_pretrained_ckpt="/path/to/droid_pretrain/2760000.pth"   # <-- the DROID checkpoint
libero_path="/path/to/LIBERO"

node=1
node_num=<YOUR_GPU_COUNT>            # upstream default 8
```

**Keep the global batch at 512** (upstream: 8 GPUs × bs 16 × accum 4):

| GPUs | `batch_size` | `gradient_accumulation_steps` |
|---|---|---|
| 8 | 16 | 4 (upstream) |
| 4 | 16 | 8 |
| 2 | 16 | 16 |
| 1 | 16 | 32 (or bs 8 / accum 64 if 16 OOMs) |

Everything else in `finetune.sh` stays as shipped: `lr 1e-3`, `num_epochs 40`,
`sequence_length 7`, `window_size 10`, `--gripper_width`, `--reset_action_token`,
`--reset_obs_token`, `save_checkpoint_seq 1`, `start_save_checkpoint 25`, `warmup_epochs 5`.

Then:
```bash
bash scripts/LIBERO_LONG/Seer/finetune.sh
```

### 5.3 Eval

`scripts/LIBERO_LONG/Seer/eval.sh` loops checkpoints 30–39; point `resume_from_checkpoint` at the
finetune output dir. It also contains hardcoded `/home/tianyang/...` paths that must be fixed.

---

## 6. Compatibility notes - read before trusting the load

The finetune loads the pretrained weights with **`load_state_dict(..., strict=False)`**
(`train.py` ~L181). **Mismatched or missing keys are silently skipped** - nothing errors.
Check the `loading pretrained weights : ...` line it prints, and verify the loss actually starts
lower than a from-scratch run.

Known DROID-pretrain -> LIBERO-finetune differences:

| Item | DROID pretrain | LIBERO finetune | Effect |
|---|---|---|---|
| `sequence_length` | 11 | 7 | **Handled** - the loader truncates `transformer_backbone_position_embedding` to `[:, :7]` automatically. |
| `action_pred_token`, `obs_tokens` | trained | `--reset_action_token`, `--reset_obs_token` | **Deliberately deleted** before load - re-initialized. Expected. |
| `--gripper_width` | **not set** | **set** | ⚠️ **Silent semantic mismatch.** `gripper_state_encoder` is `nn.Linear(2, 384)` in *both*, so weights load without a shape error - but DROID fed it a **2-class one-hot** (binary open/closed) while LIBERO feeds **continuous gripper width**. The layer transfers by shape, not by meaning, and must re-adapt during finetuning. Harmless but worth knowing; if results disappoint, try `--reset_action_decoder` or training this layer with a higher LR. |
| `atten_goal` / `atten_goal_state` / `atten_only_obs` | set | not set | Changes attention masking, not parameter shapes. |
| Action space | DROID delta wrist/TCP pose (`action_delta_wrist_pose`) | LIBERO action space | Action decoder is re-adapted during finetuning; `--reset_action_decoder` is available if transfer looks harmful. |
| DDP prefix | `module.` | `module.` | Matches - `train.py` always wraps in DDP. |

---

## 7. Honest limitations

Be careful how you frame any downstream number from this checkpoint:

1. **Partial training.** Reached **epoch 5 of 20** (~24% of the configured schedule). The
   reference DROID run is 20–30 epochs on 32 GPUs; the plan's "minimum viable" stop is epoch 10.
   Completing all 20 epochs on 1 GPU would take **~51 more days**.
2. **Curated subset, not full DROID.** 20,886 of 76,500 success episodes (27%), and deliberately
   biased toward LIBERO/CALVIN-like tasks. This *helps* downstream transfer but means it is **not**
   the paper's data condition.
3. **1 GPU vs 32.** Global batch (2048) was preserved via gradient accumulation, so the
   optimization math matches - only wall-clock and total steps seen differ.
4. **Discontinuous training history.** The run was restarted 3× (two OOM crashes from the leaks in
   §4, one deliberate restart to apply the fix). Each resume advances to the next epoch and skips
   the current epoch's remainder, and RNG state is not checkpointed. Weights/optimizer/scheduler
   carry over correctly, but the loss curve is split across 4 wandb runs.
5. **Train loss only.** No held-out validation was run during pretraining. `loss 0.0856` is a
   training loss and says nothing directly about downstream success.

**For a meaningful result you need a baseline.** Finetune LIBERO-10 from at least one of:
- the **LIBERO-90-pretrained** checkpoint (the plan's week-1 path), and/or
- **from scratch** (no pretrain).

Then compare LIBERO-LONG success rates. Reference points: Seer paper's LIBERO-LONG number, and
SmolVLA ≈ 68. A DROID-pretrain win should show up as a higher success rate than the
scratch/LIBERO-90 baseline under an otherwise identical finetune config.

---

## 8. Reproducing the curation (optional)

Scratch scripts used to build the subset (adapt paths):

| Script | Purpose |
|---|---|
| `scan_droid_lang.py` | Extract language for all 37,160 languaged episodes -> `droid_lang_cache.json` |
| `score_and_select_v2.py` | CLIP + lexical scoring vs LIBERO/CALVIN -> writes `data_info/droid_success_libero_calvin_curated.json` |
| `stage_droid.sh` | Single-stream sorted `tar` copy HDD->NVMe (**do not parallelize** - 4 streams were 4× *slower* on one spinning disk) |
| `leak_full_getitem.py`, `leak_dataloader.py`, `leak_trim_test.py` | Memory-leak probes used to diagnose §4 items 3–4 |

LIBERO target instructions come from `LIBERO/libero/libero/bddl_files/*/*.bddl` (`(:language ...)`);
CALVIN targets are derived from the 34 keys of `utils/enrich_lang_annotations.json`.
