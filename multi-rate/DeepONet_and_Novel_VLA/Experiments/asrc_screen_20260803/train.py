#!/usr/bin/env python
"""
train.py  (DeepONet campaign)
=============================
One trainer for the three head-to-head models, all under an IDENTICAL recipe so
the comparison is fair:

    M1  --head flow     --variant baseline               (flow-matching)
    M3  --head deeponet --variant baseline               (DeepONet)
    M4  --head deeponet --variant ph --lambda_ph 0.02    (DeepONet + PH)

Recipe (LIBERO-Spatial, batch 48, ~7.5 epochs = ~400k frames seen)
------------------------------------------------------------------
Stage 1 (head warm-up, backbone frozen) : ~1,650 steps, head lr 1e-4
Stage 2 (full-backbone fine-tune)       : ~6,650 steps, head lr 1e-4,
                                          backbone lr 1e-5 (500-step warmup),
                                          bf16 + gradient checkpointing
EMA(0.999) over all trainable (non-dead) params; checkpoints save EMA weights.

Logging mirrors the flow-matching runs (log_step.csv / log_epoch.csv) with the
same column names (flow_matching_loss column holds the MSE regression loss for
the DeepONet models) so the existing plotting utilities work for all three.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    """Seed python/numpy/torch for reproducible multi-seed comparison."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

DEV = "cuda"
REPO_DATA = "lerobot/libero_spatial_image"


# ----------------------------------------------------------------------------- policy
def build_policy(head, variant, base_ckpt, lambda_ph, ph_k, deeponet_p,
                 deeponet_blocks=3, deeponet_queries=8, deeponet_fourier=16,
                 deeponet_head="deeponet", deeponet_pool_norm=False,
                 deeponet_trunk_bandlimit=False, rate_consistency_weight=0.0,
                 consistency_rates=(5, 10, 25, 40, 50)):
    ph_enabled = (variant == "ph")
    if head == "flow":
        from modeling_smolvla_ph import SmolVLAPHPolicy
        pol = SmolVLAPHPolicy.from_pretrained(
            base_ckpt, ph_enabled=ph_enabled, lambda_ph=lambda_ph, ph_k=ph_k)
    elif head == "deeponet":
        from modeling_smolvla_deeponet_v2 import SmolVLADeepONetPolicy
        # POD: load with standard deeponet weights first, then swap the head
        # (avoids safetensors shared-buffer issues with POD buffers at load time).
        load_head = "deeponet" if deeponet_head == "pod" else deeponet_head
        pol = SmolVLADeepONetPolicy.from_pretrained(
            base_ckpt, ph_enabled=ph_enabled, lambda_ph=lambda_ph, ph_k=ph_k,
            deeponet_p=deeponet_p, deeponet_blocks=deeponet_blocks,
            deeponet_queries=deeponet_queries, deeponet_fourier=deeponet_fourier,
            deeponet_head=load_head, deeponet_pool_norm=deeponet_pool_norm,
            deeponet_trunk_bandlimit=deeponet_trunk_bandlimit,
            rate_consistency_weight=rate_consistency_weight,
            consistency_rates=consistency_rates)
        if deeponet_head == "pod":  # POD_SWAP
            import os, torch, sys
            sys.path.insert(0, os.path.expanduser("~/Desktop/Ayush PH test/contrib_postjul15"))
            from pod_trunk import PODHead
            pod = torch.load(os.environ["POD_CKPT"], map_location="cpu", weights_only=False)
            ctx = pol.model.vlm_with_expert.config.text_config.hidden_size
            pol.model.deeponet = PODHead(
                context_dim=ctx, mean=pod["mean"], basis=pod["basis"],
                d_model=512, n_queries=deeponet_queries, n_blocks=deeponet_blocks)
            print(f"[train] swapped in PODHead p={pod['basis'].shape[0]} params={pol.model.deeponet.num_params()/1e6:.2f}M", flush=True)
    else:
        raise ValueError(head)
    return pol.to(DEV)


def adapt_features(policy, meta):
    from modeling_smolvla_ph import adapt_policy_features_to_dataset
    return adapt_policy_features_to_dataset(policy, meta)


# ----------------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average over all currently-and-future trainable params.

    Built over every param that is NOT permanently dead, so it is valid across
    the stage-1 -> stage-2 transition (frozen params simply equal their own EMA).
    """

    def __init__(self, policy, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.names = []
        is_dead = getattr(policy, "_is_dead", lambda n: False)
        for n, p in policy.named_parameters():
            if is_dead(n):
                continue
            self.shadow[n] = p.detach().float().clone()
            self.names.append(n)
        self._backup = None

    @torch.no_grad()
    def update(self, policy):
        d = self.decay
        for n, p in policy.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def store_and_copy(self, policy):
        """Back up live weights and load EMA weights (for eval/checkpoint)."""
        self._backup = {}
        for n, p in policy.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def restore(self, policy):
        if self._backup is None:
            return
        for n, p in policy.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = None

    def state(self):
        return {"decay": self.decay, "shadow": self.shadow}


# ----------------------------------------------------------------------------- helpers
def to_device(b, dev=DEV):
    if torch.is_tensor(b):
        return b.to(dev, non_blocking=True)
    if isinstance(b, dict):
        return {k: to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(to_device(v, dev) for v in b)
    return b


def cyclic_loader(dataset, batch_size, num_workers):
    epoch = 0
    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=True, persistent_workers=False)
        for batch in loader:
            yield batch, epoch
        epoch += 1


class CSVLogger:
    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(
                {k: row.get(k, "") for k in self.fields})


def vram_gb():
    return torch.cuda.max_memory_allocated() / 1e9


# ----------------------------------------------------------------------------- stage
def run_stage(stage_name, scfg, policy, ema, data_iter, preprocessor, out_dir,
              step_logger, epoch_logger, epoch_steps, ckpt_every,
              global_step_start, grad_clip=10.0):
    is_stage2 = "backbone_lr" in scfg
    head_lr = scfg["head_lr"]
    backbone_lr = scfg.get("backbone_lr", 0.0)
    warmup = scfg.get("warmup", 0)
    n_steps = scfg["steps"]
    if n_steps <= 0:
        print(f"=== {stage_name} skipped (0 steps) ===", flush=True)
        return global_step_start

    if is_stage2:
        policy.unfreeze_all()
        if scfg.get("grad_ckpt"):
            policy.enable_gradient_checkpointing()
    else:
        policy.freeze_backbone()

    groups = policy.param_groups(backbone_lr=backbone_lr, head_lr=head_lr)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-6)
    bb_idx = next((i for i, g in enumerate(optimizer.param_groups)
                   if g.get("name") == "backbone"), None)

    tc = policy.trainable_param_count()
    print(f"\n=== {stage_name} | steps={n_steps} batch={scfg['batch']} "
          f"trainable: head={tc['head']/1e6:.2f}M backbone={tc['backbone']/1e6:.1f}M ===",
          flush=True)

    policy.train()
    epoch_buf = []
    t_last = time.perf_counter()
    for s in range(n_steps):
        gstep = global_step_start + s
        if is_stage2 and bb_idx is not None:
            ramp = min(1.0, (s + 1) / max(1, warmup))
            optimizer.param_groups[bb_idx]["lr"] = backbone_lr * ramp

        batch, ep = next(data_iter)
        batch = to_device(preprocessor(batch))

        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ld = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            (p for p in policy.parameters() if p.requires_grad), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(policy)

        lr_head = next(g["lr"] for g in optimizer.param_groups if g["name"] == "head")
        lr_bb = optimizer.param_groups[bb_idx]["lr"] if bb_idx is not None else 0.0
        step_time = time.perf_counter() - t_last
        t_last = time.perf_counter()

        row = dict(global_step=gstep, stage=stage_name, dataset_epoch=ep,
                   flow_matching_loss=ld["flow_matching_loss"], l1_loss=ld["l1_loss"],
                   ph_loss=ld["ph_loss"], rate_consistency_loss=ld.get("rate_consistency_loss", 0.0),
                   total_loss=ld["total_loss"],
                   lr_head=lr_head, lr_backbone=lr_bb, vram_gb=round(vram_gb(), 3),
                   grad_norm=round(float(gnorm), 4), step_time_s=round(step_time, 4))
        step_logger.log(row)
        epoch_buf.append(row)

        if (s + 1) % epoch_steps == 0:
            n = len(epoch_buf)
            agg = {k: sum(r[k] for r in epoch_buf) / n
                   for k in ("flow_matching_loss", "l1_loss", "ph_loss",
                             "rate_consistency_loss", "total_loss", "vram_gb")}
            epoch_logger.log(dict(epoch_global_step=gstep, stage=stage_name,
                                  logging_epoch=(gstep // epoch_steps), **agg))
            print(f"[{stage_name}] step {gstep:6d} | mse={agg['flow_matching_loss']:.4f} "
                   f"L1={agg['l1_loss']:.4f} PH={agg['ph_loss']:.4f} "
                   f"RATE={agg['rate_consistency_loss']:.4f} "
                   f"total={agg['total_loss']:.4f} | VRAM={agg['vram_gb']:.1f}GB "
                  f"| {step_time:.3f}s/it", flush=True)
            epoch_buf = []

        if (s + 1) % ckpt_every == 0 or (s + 1) == n_steps:
            save_checkpoint(policy, ema, preprocessor, out_dir, gstep + 1)

    return global_step_start + n_steps


def save_checkpoint(policy, ema, preprocessor, out_dir, gstep):
    cdir = Path(out_dir) / "checkpoints" / str(gstep)
    cdir.mkdir(parents=True, exist_ok=True)
    if ema is not None:
        ema.store_and_copy(policy)      # save EMA weights as the checkpoint
    policy.save_pretrained(cdir)
    if ema is not None:
        ema.restore(policy)
    try:
        preprocessor.save_pretrained(cdir)
    except Exception as e:
        print(f"[ckpt] processor save skipped ({e})", flush=True)
    (Path(out_dir) / "checkpoints" / "LATEST.txt").write_text(str(gstep))
    print(f"[ckpt] saved -> {cdir}  (EMA weights)", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["flow", "deeponet"], required=True)
    ap.add_argument("--variant", choices=["baseline", "ph"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_ckpt", default="lerobot/smolvla_base")
    ap.add_argument("--dataset", default=REPO_DATA)
    ap.add_argument("--lambda_ph", type=float, default=0.02)
    ap.add_argument("--ph_k", type=int, default=8)
    ap.add_argument("--deeponet_p", type=int, default=256)
    ap.add_argument("--deeponet_blocks", type=int, default=3)
    ap.add_argument("--deeponet_queries", type=int, default=8)
    ap.add_argument("--deeponet_fourier", type=int, default=16)
    ap.add_argument("--trunk_bandlimit", action="store_true",
                    help="Linear (Nyquist-respecting) Fourier band spacing in the trunk "
                         "instead of the geometric 2^k ladder. Same n_fourier => IDENTICAL "
                         "parameter count, so the only variable is whether bands alias "
                         "against the action chunk sample rate.")
    ap.add_argument("--deeponet_head", default="deeponet", choices=["deeponet", "tempo", "reg", "pod", "cross_gated_operator", "adaptive_hybrid_operator", "relational_hybrid_operator", "spectral_operator", "ti", "asrc"])
    ap.add_argument("--tempo_min_speed", type=float, default=0.5)
    ap.add_argument("--tempo_max_speed", type=float, default=2.0)
    ap.add_argument("--rate_consistency_weight", type=float, default=0.0)
    ap.add_argument("--consistency_rates", default="5,10,25,40,50")
    ap.add_argument("--pool_channel_norm", action="store_true",
                    help="Per-channel norm across the token axis before CrossAttnPool (targets "
                         "Camera Viewpoints / Sensor Noise robustness; composes with deeponet/reg "
                         "heads; NOT a per-token LayerNorm -- ln_kv inside CrossBlock already does "
                         "that and a second one is a verified no-op, see deeponet_head_v2.py)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--stage1_steps", type=int, default=1650)
    ap.add_argument("--stage2_steps", type=int, default=6650)
    ap.add_argument("--stage1_batch", type=int, default=48)
    ap.add_argument("--stage2_batch", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 disables)")
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--epoch_steps", type=int, default=200)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true", help="lerobot photometric+affine image aug (robustness)")
    args = ap.parse_args()

    consistency_rates = tuple(int(x) for x in args.consistency_rates.split(",") if x.strip())
    if any(r <= 0 for r in consistency_rates):
        raise SystemExit("[FATAL] consistency rates must be positive")
    if args.deeponet_head == "asrc" and (not args.trunk_bandlimit or args.rate_consistency_weight <= 0):
        raise SystemExit("[FATAL] ASRC requires --trunk_bandlimit and --rate_consistency_weight > 0")
    if args.deeponet_head == "ti" and args.rate_consistency_weight != 0:
        raise SystemExit("[FATAL] TI control must use --rate_consistency_weight 0")
    if args.deeponet_head == "tempo" and not (0 < args.tempo_min_speed <= 1 <= args.tempo_max_speed):
        raise SystemExit("[FATAL] TempoVLA-style speed range must satisfy 0 < min <= 1 <= max")


    # guard: train.py takes --deeponet_head and IGNORES $DEEPONET_HEAD. A stale export
    # previously trained the wrong architecture silently while the eval honoured the var.
    import os as _osg
    _envh = _osg.environ.get("DEEPONET_HEAD")
    if _envh and _envh != args.deeponet_head:
        raise SystemExit(
            f"[FATAL] DEEPONET_HEAD={_envh} but --deeponet_head={args.deeponet_head}; "
            "train.py reads the CLI flag only — fix the launcher.")
    _envn = _osg.environ.get("DEEPONET_POOL_CHANNEL_NORM")
    if _envn is not None and bool(int(_envn)) != args.pool_channel_norm:
        raise SystemExit(
            f"[FATAL] DEEPONET_POOL_CHANNEL_NORM={_envn} but --pool_channel_norm={args.pool_channel_norm}; "
            "train.py reads the CLI flag only — fix the launcher.")
    _envb = _osg.environ.get("DEEPONET_TRUNK_BANDLIMIT")
    if _envb is not None and bool(int(_envb)) != args.trunk_bandlimit:
        raise SystemExit(
            f"[FATAL] DEEPONET_TRUNK_BANDLIMIT={_envb} but --trunk_bandlimit={args.trunk_bandlimit}; "
            "train.py reads the CLI flag only — fix the launcher.")
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] head={args.head} variant={args.variant} seed={args.seed} out={out_dir}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    policy = build_policy(args.head, args.variant, args.base_ckpt,
                          args.lambda_ph, args.ph_k, args.deeponet_p,
                          args.deeponet_blocks, args.deeponet_queries, args.deeponet_fourier,
                          args.deeponet_head, args.pool_channel_norm, args.trunk_bandlimit,
                          args.rate_consistency_weight, consistency_rates)

    meta = LeRobotDatasetMetadata(args.dataset)
    adapt_features(policy, meta)
    dt = resolve_delta_timestamps(policy.config, meta)
    if args.deeponet_head == "tempo":
        source_steps = int(np.ceil(args.tempo_max_speed * policy.config.chunk_size))
        dt["action"] = [i / meta.fps for i in range(source_steps)]
        policy.configure_tempo_training(args.tempo_min_speed, args.tempo_max_speed)
        print(f"[tempo] VSTA source_steps={source_steps} output_steps={policy.config.chunk_size} "
              f"speed=[{args.tempo_min_speed},{args.tempo_max_speed}]", flush=True)
    image_transforms = None
    if args.augment:
        import os as _os
        from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig
        _cfg = ImageTransformsConfig(enable=True)
        if _os.environ.get('AUG_AFFINE_ONLY'):
            _cfg.tfs = {'affine': _cfg.tfs['affine']}
            _cfg.max_num_transforms = 1
        elif _os.environ.get('AUG_NO_AFFINE'):
            _cfg.tfs.pop('affine', None)
            _cfg.max_num_transforms = min(_cfg.max_num_transforms, len(_cfg.tfs))
        image_transforms = ImageTransforms(_cfg)
        print('[train] AUGMENT ON tfs=%s' % list(_cfg.tfs.keys()), flush=True)
    dataset = LeRobotDataset(args.dataset, delta_timestamps=dt, image_transforms=image_transforms)
    norm_stats = torch.load(args.stats_path) if args.stats_path else meta.stats
    configure_stats = getattr(policy, "configure_action_stats", None)
    if configure_stats is not None:
        configure_stats(norm_stats["action"])
    preprocessor, _post = make_smolvla_pre_post_processors(policy.config, dataset_stats=norm_stats)
    print(f"[train] dataset frames={dataset.num_frames} episodes={dataset.num_episodes} "
          f"-> 1 epoch = {dataset.num_frames/args.stage2_batch:.0f} steps @ batch {args.stage2_batch}",
          flush=True)
    total_frames = args.stage1_steps * args.stage1_batch + args.stage2_steps * args.stage2_batch
    print(f"[train] planned frames seen = {total_frames} "
          f"= {total_frames/dataset.num_frames:.2f} epochs", flush=True)

    ema = EMA(policy, decay=args.ema) if args.ema and args.ema > 0 else None

    cfg = {
        "stage1": dict(steps=args.stage1_steps, batch=args.stage1_batch, head_lr=args.head_lr),
        "stage2": dict(steps=args.stage2_steps, batch=args.stage2_batch, head_lr=args.head_lr,
                       backbone_lr=args.backbone_lr, warmup=args.warmup, grad_ckpt=True),
    }

    step_fields = ["global_step", "stage", "dataset_epoch", "flow_matching_loss", "l1_loss",
                   "ph_loss", "rate_consistency_loss", "total_loss", "lr_head", "lr_backbone", "vram_gb",
                   "grad_norm", "step_time_s"]
    epoch_fields = ["epoch_global_step", "stage", "logging_epoch", "flow_matching_loss",
                    "l1_loss", "ph_loss", "rate_consistency_loss", "total_loss", "vram_gb"]
    step_logger = CSVLogger(out_dir / "log_step.csv", step_fields)
    epoch_logger = CSVLogger(out_dir / "log_epoch.csv", epoch_fields)

    (out_dir / "run_config.json").write_text(json.dumps(
        {"args": vars(args), "stage_configs": cfg,
         "dataset_frames": dataset.num_frames,
         "planned_epochs": total_frames / dataset.num_frames}, indent=2, default=str))

    t0 = time.perf_counter()
    gstep = 0
    di = cyclic_loader(dataset, cfg["stage1"]["batch"], args.num_workers)
    gstep = run_stage("stage1", cfg["stage1"], policy, ema, di, preprocessor, out_dir,
                      step_logger, epoch_logger, args.epoch_steps, args.ckpt_every, gstep)
    di = cyclic_loader(dataset, cfg["stage2"]["batch"], args.num_workers)
    gstep = run_stage("stage2", cfg["stage2"], policy, ema, di, preprocessor, out_dir,
                      step_logger, epoch_logger, args.epoch_steps, args.ckpt_every, gstep)

    dt_min = (time.perf_counter() - t0) / 60
    print(f"\n[train] DONE head={args.head} variant={args.variant} steps={gstep} "
          f"wall={dt_min:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
