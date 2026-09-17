"""
modeling_smolvla_deeponet_v2.py
===============================
SmolVLA + DeepONet head v2. Identical to v1 except the head reads the FULL prefix
token sequence via cross-attention (CrossAttnPool inside DeepONetHeadV2) instead
of a single mean-pooled context vector — fixing the spatial-localization
bottleneck. Class name SmolVLADeepONetPolicy is kept so train/evaluate import
unchanged (they point at this module in v2/).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
)

from deeponet_head_v2 import DeepONetHeadV2
from ph_loss import ph_surrogate_loss
from modeling_smolvla_ph import adapt_policy_features_to_dataset  # noqa: F401

BACKBONE_PREFIX = "model.vlm_with_expert.vlm."
DEAD_PREFIXES = (
    "model.vlm_with_expert.lm_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)


class VLADeepONetV2(VLAFlowMatching):
    def __init__(self, config, p=256, d_model=512, n_queries=8, n_blocks=3, n_fourier=16,
                 head_type="deeponet", channel_norm=False, trunk_bandlimit=False,
                 rate_consistency_weight=0.0, consistency_rates=(5, 10, 25, 40, 50)):
        super().__init__(config)
        context_dim = self.vlm_with_expert.config.text_config.hidden_size
        self.tempo_enabled = head_type == "tempo"
        self._tempo_speed = None
        self.register_buffer("action_mean", torch.empty(0), persistent=False)
        self.register_buffer("action_std", torch.empty(0), persistent=False)
        if head_type in ("ti", "asrc"):
            from rate_integrated_deeponet import RateIntegratedDeepONetHead
            self.deeponet = RateIntegratedDeepONetHead(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, variant=head_type, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
                consistency_rates=consistency_rates)
        elif head_type == "tempo":
            self.deeponet = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
                channel_norm=channel_norm, trunk_bandlimit=trunk_bandlimit,
                speed_condition=True,
            )
        elif head_type == "pod":
            import os, sys
            sys.path.insert(0, os.path.expanduser("~/Desktop/Ayush PH test/contrib_postjul15"))
            from pod_trunk import PODHead
            pod = torch.load(os.environ["POD_CKPT"], map_location="cpu", weights_only=False)
            self.deeponet = PODHead(
                context_dim=context_dim, mean=pod["mean"], basis=pod["basis"],
                d_model=d_model, n_queries=n_queries, n_blocks=n_blocks)
        elif head_type == "spectral_operator":
            from multi_operator_head import SpectralGateDeepONetHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = SpectralGateDeepONetHead(base)
        elif head_type == "timewarp_operator":
            from multi_operator_head import TimeWarpDeepONetHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = TimeWarpDeepONetHead(base)
        elif head_type == "overlap_operator":
            import os
            from multi_operator_head import OverlapConditionedDeepONetHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = OverlapConditionedDeepONetHead(
                base, replan=int(os.environ.get("OVERLAP_REPLAN", "5")))
        elif head_type == "multioperator_recurrent":
            import os
            from multi_operator_head import RecurrentMultiOperatorHead, MultiOperatorHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = RecurrentMultiOperatorHead(
                MultiOperatorHead(
                    base, n_modes=int(os.environ.get("MULTIOP_MODES", "3"))))
        elif head_type == "multioperator_contextmarkov":
            import os
            from multi_operator_head import ContextualMarkovMultiOperatorHead, MultiOperatorHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = ContextualMarkovMultiOperatorHead(
                MultiOperatorHead(
                    base, n_modes=int(os.environ.get("MULTIOP_MODES", "3"))))
        elif head_type == "multioperator_markov":
            import os
            from multi_operator_head import MarkovMultiOperatorHead, MultiOperatorHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = MarkovMultiOperatorHead(
                MultiOperatorHead(
                    base, n_modes=int(os.environ.get("MULTIOP_MODES", "3"))))
        elif head_type == "relational_hybrid_operator":
            import os
            from regression_head import RegressionHeadV2
            from multi_operator_head import RelationalHybridOperatorHead
            regression = RegressionHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks,
            )
            operator = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = RelationalHybridOperatorHead(
                regression, operator,
                max_delta=float(os.environ.get("OPERATOR_MAX_DELTA", "0.25")),
            )
        elif head_type == "adaptive_hybrid_operator":
            import os
            from regression_head import RegressionHeadV2
            from multi_operator_head import AdaptiveHybridOperatorHead
            regression = RegressionHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks,
            )
            operator = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = AdaptiveHybridOperatorHead(
                regression, operator,
                max_delta=float(os.environ.get("OPERATOR_MAX_DELTA", "0.25")),
            )
        elif head_type == "cross_gated_operator":
            from multi_operator_head import CrossGatedDeepONetHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = CrossGatedDeepONetHead(base)
        elif head_type == "bounded_residual_operator":
            import os
            from regression_head import RegressionHeadV2
            from multi_operator_head import BoundedResidualOperatorHead
            regression = RegressionHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks,
            )
            operator = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = BoundedResidualOperatorHead(
                regression, operator,
                max_delta=float(os.environ.get("OPERATOR_MAX_DELTA", "0.25")),
            )
        elif head_type == "multioperator_residual":
            import os
            from multi_operator_head import MultiOperatorHead, ResidualMultiOperatorHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            multi_base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = ResidualMultiOperatorHead(
                base,
                MultiOperatorHead(
                    multi_base, n_modes=int(os.environ.get("MULTIOP_MODES", "3"))),
                alpha=float(os.environ.get("MULTIOP_RESIDUAL_ALPHA", "0.25")),
            )
        elif head_type == "multioperator_temporal":
            import os
            from multi_operator_head import TemporalMultiOperatorHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = TemporalMultiOperatorHead(
                base, n_modes=int(os.environ.get("MULTIOP_MODES", "3")))
        elif head_type == "multioperator_energy":
            import os
            from multi_operator_head import EnergyRoutedMultiOperatorHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = EnergyRoutedMultiOperatorHead(
                base, n_modes=int(os.environ.get("MULTIOP_MODES", "3")))
        elif head_type == "multioperator":
            import os
            from multi_operator_head import MultiOperatorHead
            base = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
            self.deeponet = MultiOperatorHead(
                base, n_modes=int(os.environ.get("MULTIOP_MODES", "3")))
        elif head_type == "reg":
            from regression_head import RegressionHeadV2
            self.deeponet = RegressionHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, channel_norm=channel_norm)
        else:
            self.deeponet = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
                channel_norm=channel_norm, trunk_bandlimit=trunk_bandlimit,
            )
        self._ph_cache = None
        self.rate_consistency_weight = float(rate_consistency_weight)

    def encode_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        """Run the VLM over the prefix only; return (prefix_out fp32, pad_mask)."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        (prefix_out, _), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks, position_ids=position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None],
            use_cache=False, fill_kv_cache=True,
        )
        return prefix_out.to(torch.float32), prefix_pad_masks

    def configure_action_stats(self, mean, std):
        device = next(self.parameters()).device
        mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(std, dtype=torch.float32, device=device)
        self.action_mean = mean.clone()
        self.action_std = std.clone().clamp_min(1e-6)
        configure = getattr(self.deeponet, "configure_action_stats", None)
        if configure is not None:
            configure(mean, std)

    def denormalize_actions(self, actions):
        if self.action_mean.numel() == 0:
            raise RuntimeError("action stats are required for TempoVLA-style VSTA")
        out = actions.clone()
        d = self.action_mean.numel()
        out[..., :d] = actions[..., :d] * self.action_std + self.action_mean
        return out

    def normalize_actions(self, actions):
        out = actions.clone()
        d = self.action_mean.numel()
        out[..., :d] = (actions[..., :d] - self.action_mean) / self.action_std
        return out

    def set_execution_speed(self, speed):
        value = torch.as_tensor(speed, dtype=torch.float32)
        if value.numel() != 1 or value.item() <= 0:
            raise ValueError("execution speed must be one positive scalar")
        self._tempo_speed = value.reshape(1, 1)

    def set_training_speed(self, speed):
        if speed.ndim != 2 or speed.shape[1] != 1 or (speed <= 0).any():
            raise ValueError("training speed must be (batch,1) and positive")
        self._tempo_speed = speed

    def _speed_for(self, batch_size, device):
        if not self.tempo_enabled:
            return None
        speed = self._tempo_speed
        if speed is None:
            return torch.ones(batch_size, 1, dtype=torch.float32, device=device)
        if speed.shape[0] == 1:
            return speed.to(device).expand(batch_size, 1)
        if speed.shape == (batch_size, 1):
            return speed.to(device)
        raise ValueError("training speed must be shaped (batch,1)")

    def predict_chunk(self, images, img_masks, lang_tokens, lang_masks, state) -> Tensor:
        prefix_out, pad_mask = self.encode_prefix(images, img_masks, lang_tokens, lang_masks, state)
        with torch.autocast("cuda", enabled=False):
            pred = self.deeponet(prefix_out.float(), pad_mask,
                                speed=self._speed_for(prefix_out.shape[0], prefix_out.device))
        return pred

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions,
                noise=None, time=None) -> Tensor:
        pred = self.predict_chunk(images, img_masks, lang_tokens, lang_masks, state)
        actions = actions[:, :pred.shape[1]].to(pred.dtype)
        self._ph_cache = {"pred": pred, "actions": actions}
        return F.mse_loss(pred, actions, reduction="none")

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       noise=None, **kwargs) -> Tensor:
        return self.predict_chunk(images, img_masks, lang_tokens, lang_masks, state)


class SmolVLADeepONetPolicy(SmolVLAPolicy):
    def __init__(self, config, ph_enabled=False, lambda_ph=0.0, ph_k=8, ph_p=2.0,
                 deeponet_p=256, deeponet_blocks=3, deeponet_queries=8, deeponet_fourier=16,
                 deeponet_head="deeponet", deeponet_pool_norm=False,
                 deeponet_trunk_bandlimit=False, rate_consistency_weight=0.0,
                 consistency_rates=(5, 10, 25, 40, 50), **kwargs):
        super().__init__(config, **kwargs)
        old = self.model
        self.model = VLADeepONetV2(config, p=deeponet_p, n_blocks=deeponet_blocks,
                                   n_queries=deeponet_queries, n_fourier=deeponet_fourier,
                                   head_type=deeponet_head, channel_norm=deeponet_pool_norm,
                                   trunk_bandlimit=deeponet_trunk_bandlimit,
                                   rate_consistency_weight=rate_consistency_weight,
                                   consistency_rates=consistency_rates)
        self.model.load_state_dict(old.state_dict(), strict=False)
        del old
        self.configure_ph(ph_enabled, lambda_ph, ph_k, ph_p)
        self._freeze_dead()

    def reset(self):
        super().reset()
        reset_router = getattr(getattr(self.model, "deeponet", None), "reset_router", None)
        if reset_router is not None:
            reset_router()

    def configure_ph(self, enabled=False, lambda_ph=0.0, k=8, p=2.0):
        self.ph_enabled = bool(enabled); self.lambda_ph = float(lambda_ph)
        self.ph_k = int(k); self.ph_p = float(p); return self

    def configure_action_stats(self, stats):
        self.model.configure_action_stats(stats["mean"], stats["std"])
        return self

    def configure_tempo_training(self, low=0.5, high=2.0):
        if not self.model.tempo_enabled:
            raise ValueError("TempoVLA-style VSTA requires --deeponet_head tempo")
        if not (0 < low <= 1 <= high):
            raise ValueError("speed range must satisfy 0 < low <= 1 <= high")
        self.tempo_speed_range = (float(low), float(high))
        return self

    def set_execution_speed(self, speed):
        self.model.set_execution_speed(speed)
        return self

    @staticmethod
    def _is_backbone(name): return name.startswith(BACKBONE_PREFIX)
    @staticmethod
    def _is_dead(name): return any(name.startswith(p) for p in DEAD_PREFIXES)

    def _freeze_dead(self):
        for n, p in self.named_parameters():
            if self._is_dead(n): p.requires_grad = False

    def forward(self, batch, noise=None, time=None, reduction="mean"):
        tempo_training = self.model.tempo_enabled and self.training
        if tempo_training:
            from tempo_vsta import retime_delta_actions, sample_speed
            if "action_is_pad" not in batch:
                raise RuntimeError("TempoVLA-style VSTA requires action_is_pad")
            low, high = getattr(self, "tempo_speed_range", (0.5, 2.0))
            batch = dict(batch)
            speeds = sample_speed(batch["action"].shape[0], low, high, batch["action"].device)
            raw_actions = self.model.denormalize_actions(batch["action"])
            raw_actions, action_is_pad = retime_delta_actions(
                raw_actions, batch["action_is_pad"], speeds, self.config.chunk_size,
                self.config.action_feature.shape[0] - 1)
            batch["action"] = self.model.normalize_actions(raw_actions)
            batch["action_is_pad"] = action_is_pad
            self.model.set_training_speed(speeds)
        try:
            loss, loss_dict = super().forward(batch, noise=noise, time=time, reduction=reduction)
        finally:
            if tempo_training:
                self.model._tempo_speed = None
        cache = getattr(self.model, "_ph_cache", None)
        ph_val = loss.new_zeros(()); l1_val = loss.new_zeros(()); rate_val = loss.new_zeros(())
        if cache is not None:
            d = self.config.action_feature.shape[0]
            pred_a = cache["pred"][:, :, :d]; tgt_a = cache["actions"][:, :, :d]
            l1_val = (pred_a - tgt_a).abs().mean()
            if self.ph_enabled and self.lambda_ph > 0:
                ph_val = ph_surrogate_loss(pred_a, tgt_a, k=self.ph_k, p=self.ph_p, reduction="mean")
        take_rate = getattr(self.model.deeponet, "take_consistency_loss", None)
        if take_rate is not None:
            cached_rate = take_rate()
            if cached_rate is not None:
                rate_val = cached_rate
        total = loss + self.lambda_ph * ph_val if (self.ph_enabled and self.lambda_ph > 0) else loss
        total = total + self.model.rate_consistency_weight * rate_val
        loss_dict["flow_matching_loss"] = float(loss.detach())
        loss_dict["l1_loss"] = float(l1_val.detach())
        loss_dict["ph_loss"] = float(ph_val.detach())
        loss_dict["rate_consistency_loss"] = float(rate_val.detach())
        loss_dict["total_loss"] = float(total.detach())
        self.model._ph_cache = None
        return total, loss_dict

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            p.requires_grad = False if self._is_dead(n) else (not self._is_backbone(n))

    def unfreeze_all(self):
        for n, p in self.named_parameters():
            p.requires_grad = not self._is_dead(n)

    def param_groups(self, backbone_lr, head_lr):
        backbone, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad or self._is_dead(n): continue
            (backbone if self._is_backbone(n) else head).append(p)
        groups = []
        if head: groups.append({"params": head, "lr": head_lr, "name": "head"})
        if backbone: groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
        return groups

    def trainable_param_count(self):
        bb = sum(p.numel() for n, p in self.named_parameters()
                 if p.requires_grad and self._is_backbone(n) and not self._is_dead(n))
        hd = sum(p.numel() for n, p in self.named_parameters()
                 if p.requires_grad and not self._is_backbone(n) and not self._is_dead(n))
        return {"backbone": bb, "head": hd, "total": bb + hd}

    def enable_gradient_checkpointing(self):
        kw = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        try:
            self.model.vlm_with_expert.vlm.gradient_checkpointing_enable(**kw)
        except TypeError:
            self.model.vlm_with_expert.vlm.gradient_checkpointing_enable()


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = SmolVLADeepONetPolicy.from_pretrained("lerobot/smolvla_base", ph_enabled=True, lambda_ph=0.02).to(dev)
    p.freeze_backbone()
    tc = p.trainable_param_count()
    print(f"[v2] stage1 head={tc['head']/1e6:.2f}M backbone={tc['backbone']/1e6:.1f}M")
    print(f"[v2] deeponet head params = {p.model.deeponet.num_params()/1e6:.2f}M")
