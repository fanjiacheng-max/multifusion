"""
End-to-end IRENE-style four-modality CMR age regressor.

This keeps the simple_late_gated_v2 raw-data/backbone/deep-supervision
scaffolding, but replaces gate + concat MLP fusion with the existing
cmr_irene_v7 CMRIrene token fusion module.
"""
from __future__ import annotations

import os
import sys
import time
import types
import importlib.util
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_fusion.cmr_irene_v7.models.configs import CONFIGS
from multi_fusion.cmr_irene_v7.models.modeling_irene import CMRIrene

BASE = Path("/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "ViTa"))

DINO_PATH = (
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/"
    "zskj-hub/models--facebook--webssl-dino300m-full2b-224"
)

MODALITIES = ["SA", "LA", "T1", "AO"]


# ---------------------------------------------------------------------------
# Backbone wrappers（复用 extract_embeddings.py 的加载逻辑，补充中间层输出）
# ---------------------------------------------------------------------------

class ViTaBackbone(nn.Module):
    """ViTa encoder wrapper.

    forward returns:
      pooled feature [B,256] for compatibility,
      token sequence [B,N,1024] for IRENE fusion,
      aux pooled features {-4:[B,256], -2:[B,256]}.
    """

    def __init__(self, ckpt_path: str, img_shape: Tuple[int, int, int, int] = (6, 50, 128, 128)):
        super().__init__()
        if "datasets" not in sys.modules:
            dummy = types.ModuleType("datasets")
            dummy.__spec__ = importlib.util.spec_from_loader("datasets", loader=None)
            sys.modules["datasets"] = dummy
        import src.models.vita_downstream as vita_downstream

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = {k.replace("regressor.", "head."): v for k, v in state.items()}
        has_attn_pool = any("attn_pool" in k for k in state)
        variant = "strong" if has_attn_pool else "base"

        old_img_shape = vita_downstream._ENCODER_HPARAMS["img_shape"]
        try:
            vita_downstream._ENCODER_HPARAMS["img_shape"] = torch.tensor(img_shape)
            model = vita_downstream.build_vita_downstream_model(
                task_type="regression",
                ckpt_path=None,
                variant=variant,
            )
        finally:
            vita_downstream._ENCODER_HPARAMS["img_shape"] = old_img_shape

        model_state = model.state_dict()
        compatible_state = {
            k: v for k, v in state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped = sorted(set(state) - set(compatible_state))
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        if skipped:
            print(f"[ViTaBackbone] skipped incompatible checkpoint keys for img_shape={img_shape}: {skipped}")
        if unexpected:
            print(f"[ViTaBackbone] unexpected checkpoint keys: {unexpected}")

        self.encoder     = model.encoder            # ImagingMaskedEncoder
        self.dec_embed   = model.dec_embed           # Linear(1024, 256)
        self.dec_pos_embed = model.dec_pos_embed     # [1, 1+N, 256]
        self.n_blocks    = len(self.encoder.encoder)  # 6
        self.img_shape   = tuple(img_shape)

    def set_trainable(self, flag: bool):
        for p in self.parameters():
            p.requires_grad_(flag)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        if x.dim() == 6:
            x = x.squeeze(1)
        enc = self.encoder
        h = enc.patch_embed(x)
        if enc.use_enc_pe:
            pos = enc.enc_pos_embed
            h   = h + pos[:, 1:, :]
            cls = enc.cls_token + pos[:, :1, :]
        else:
            cls = enc.cls_token
        B = h.shape[0]
        h = torch.cat([cls.expand(B, -1, -1), h], dim=1)   # [B, 1+N, 1024]

        aux_idx = {self.n_blocks - 4: -4, self.n_blocks - 2: -2}
        aux_raw: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(enc.encoder):
            h = blk(h)
            if i in aux_idx:
                aux_raw[aux_idx[i]] = h
        h = enc.encoder_norm(h)
        token_seq = h[:, 1:, :]  # IRENE adds its own modality CLS tokens.

        def project(feat):
            proj = self.dec_embed(feat) + self.dec_pos_embed[:, : feat.shape[1], :]
            return proj.mean(dim=1)   # [B, 256] global pool

        final_feat = project(h)
        aux_feats  = {k: project(v) for k, v in aux_raw.items()}
        return final_feat, token_seq, aux_feats


class DinoBackbone(nn.Module):
    """DINOv2 wrapper.

    forward returns:
      CLS feature [B,1024] for compatibility,
      patch token sequence [B,N,1024] for IRENE fusion,
      aux CLS features {-4:[B,1024], -2:[B,1024]}.
    """

    def __init__(self, ckpt_path: str):
        super().__init__()
        from safetensors.torch import load_file
        from transformers import Dinov2Config, Dinov2Model

        config = Dinov2Config.from_json_file(os.path.join(DINO_PATH, "config.json"))
        backbone = Dinov2Model(config)
        backbone.load_state_dict(load_file(os.path.join(DINO_PATH, "model.safetensors")), strict=True)

        ft_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        bb_state = {k.replace("backbone.", ""): v for k, v in ft_state.items() if k.startswith("backbone.")}
        if bb_state:
            backbone.load_state_dict(bb_state, strict=True)

        self.backbone = backbone   # Dinov2Model, 24 层

    def set_trainable(self, flag: bool):
        for p in self.parameters():
            p.requires_grad_(flag)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        final_feat = out.last_hidden_state[:, 0]         # [B, 1024]
        token_seq = out.last_hidden_state[:, 1:, :]      # [B, N, 1024]
        hs = out.hidden_states                            # tuple of 25, index 0..24
        aux_feats = {
            -4: hs[len(hs) - 4][:, 0],   # hidden_states[21] CLS
            -2: hs[len(hs) - 2][:, 0],   # hidden_states[23] CLS
        }
        return final_feat, token_seq, aux_feats


# ---------------------------------------------------------------------------
# Token bridge / Aux heads
# ---------------------------------------------------------------------------

class TokenReducer(nn.Module):
    """Convert arbitrary backbone tokens to CMRIrene's [B, 256, 1024] input."""

    def __init__(self, in_dim: int, out_dim: int = 1024, n_tokens: int = 256):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self._pool_weight_cache: Dict[int, torch.Tensor] = {}

    def _pool_weight(self, n_in: int, device, dtype) -> torch.Tensor:
        # Cached [n_tokens, n_in] averaging matrix, equivalent to adaptive_avg_pool1d
        # but done via matmul. F.adaptive_avg_pool1d's CUDA backward kernel hits a
        # PyTorch internal assert ("Couldn't reduce launch bounds to accomodate
        # sharedMemPerBlock limit") on H200 for large reduction ratios (e.g. SAX's
        # 15360 -> 256 tokens), so we avoid that kernel entirely.
        w = self._pool_weight_cache.get(n_in)
        if w is None:
            n_out = self.n_tokens
            idx = torch.arange(n_out + 1, dtype=torch.float64) * n_in / n_out
            starts = idx[:-1].floor().long()
            ends = torch.maximum(idx[1:].ceil().long(), starts + 1).clamp(max=n_in)
            w = torch.zeros(n_out, n_in, dtype=torch.float32)
            for i in range(n_out):
                s, e = starts[i].item(), ends[i].item()
                w[i, s:e] = 1.0 / (e - s)
            self._pool_weight_cache[n_in] = w
        return w.to(device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected [B,N,C] tokens, got shape {tuple(tokens.shape)}")
        n_in = tokens.shape[1]
        if n_in != self.n_tokens:
            if n_in % self.n_tokens == 0:
                k = n_in // self.n_tokens
                tokens = tokens.reshape(tokens.shape[0], self.n_tokens, k, tokens.shape[2]).mean(dim=2)
            else:
                weight = self._pool_weight(n_in, tokens.device, tokens.dtype)
                tokens = torch.einsum("on,bnc->boc", weight, tokens)
        return self.norm(self.proj(tokens))


class AuxAgeHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x): return self.head(x)


def apply_modality_dropout(batch_size: int, p: float, training: bool, device) -> torch.Tensor:
    base_mask = torch.ones(batch_size, 4, device=device)
    if not training or p <= 0:
        return base_mask
    keep_prob = 1.0 - p
    mask = torch.bernoulli(torch.full((batch_size, 4), keep_prob, device=device))
    zero_rows = mask.sum(dim=1) == 0
    if zero_rows.any():
        idx = torch.randint(0, 4, (int(zero_rows.sum().item()),), device=device)
        mask[zero_rows, idx] = 1.0
    return mask


class EndToEndMultiModalAgeRegressor(nn.Module):
    def __init__(
        self,
        sa_ckpt: str, la_ckpt: str, t1_ckpt: str, ao_ckpt: str,
        irene_token_dim: int = 1024,
        irene_tokens_per_modality: int = 256,
        modality_dropout_p: float = 0.1,
        use_deep_supervision: bool = True,
        use_lia: bool = True,
        lambda_lia: float = 0.1,
        lia_temperature: float = 0.1,
        sa_img_shape: Tuple[int, int, int, int] = (6, 50, 128, 128),
        la_img_shape: Tuple[int, int, int, int] = (1, 50, 128, 128),
    ):
        super().__init__()
        self.modality_dropout_p   = modality_dropout_p
        self.use_deep_supervision = use_deep_supervision

        self.backbones = nn.ModuleDict({
            "SA": ViTaBackbone(sa_ckpt, img_shape=sa_img_shape),
            "LA": ViTaBackbone(la_ckpt, img_shape=la_img_shape),
            "T1": DinoBackbone(t1_ckpt),
            "AO": DinoBackbone(ao_ckpt),
        })
        feat_dims = {"SA": 256, "LA": 256, "T1": 1024, "AO": 1024}
        token_dims = {"SA": 1024, "LA": 1024, "T1": 1024, "AO": 1024}

        self.token_reducers = nn.ModuleDict({
            m: TokenReducer(
                token_dims[m],
                out_dim=irene_token_dim,
                n_tokens=irene_tokens_per_modality,
            )
            for m in MODALITIES
        })

        irene_config = copy.deepcopy(CONFIGS["CMR_IRENE"])
        irene_config.token_dim = irene_token_dim
        irene_config.use_lia = use_lia
        irene_config.lambda_lia = lambda_lia
        irene_config.lia_temperature = lia_temperature
        self.irene = CMRIrene(irene_config)

        if use_deep_supervision:
            aux_hidden = {"SA": 128, "LA": 128, "T1": 256, "AO": 256}
            self.aux_heads = nn.ModuleDict({
                f"{m}_{layer}": AuxAgeHead(feat_dims[m], aux_hidden[m])
                for m in MODALITIES for layer in (-4, -2)
            })

    def set_backbones_trainable(self, flag: bool):
        for bb in self.backbones.values():
            bb.set_trainable(flag)

    def enable_profile(self, flag: bool = True):
        """诊断用：打开后 forward() 会把各阶段耗时记录到 self.last_timing。默认关闭，
        生产训练（train.py）不调用这个方法，行为和性能都不受影响。"""
        self._profile = flag
        self.last_timing = {}

    def _sync(self, device):
        if getattr(self, "_profile", False) and device.type == "cuda":
            torch.cuda.synchronize()

    def forward(self, batch: Dict[str, torch.Tensor], target: torch.Tensor = None) -> Dict:
        device = batch["SA"].device
        B = batch["SA"].shape[0]
        profile = getattr(self, "_profile", False)
        timing = {}

        dropout_mask = apply_modality_dropout(B, self.modality_dropout_p, self.training, device)

        token_seqs, aux_feats = {}, {}
        for m in MODALITIES:
            self._sync(device); t0 = time.perf_counter()
            _, token_seqs[m], aux_feats[m] = self.backbones[m](batch[m])
            self._sync(device); t1 = time.perf_counter()
            if profile:
                timing[f"{m}_backbone_forward"] = t1 - t0

        self._sync(device); t0 = time.perf_counter()
        tokens_list = []
        for i, m in enumerate(MODALITIES):
            token = self.token_reducers[m](token_seqs[m])  # [B,256,1024]
            token = token * dropout_mask[:, i].view(B, 1, 1)
            tokens_list.append(token)

        irene_out = self.irene(tokens_list, dropout_mask.to(torch.bool), target=target)
        irene_loss = None
        irene_loss_dict = {}
        if target is not None:
            irene_loss, age_pred, irene_loss_dict = irene_out
        else:
            age_pred = irene_out
        self._sync(device); t1 = time.perf_counter()
        if profile:
            timing["irene_fusion_forward"] = t1 - t0

        aux_preds = {}
        if self.use_deep_supervision:
            self._sync(device); t0 = time.perf_counter()
            for m in MODALITIES:
                for layer in (-4, -2):
                    key = f"{m}_{layer}"
                    aux_preds[key] = self.aux_heads[key](aux_feats[m][layer])
            self._sync(device); t1 = time.perf_counter()
            if profile:
                timing["aux_heads_forward"] = t1 - t0

        if profile:
            self.last_timing = timing

        return {
            "age_pred": age_pred,
            "irene_loss": irene_loss,
            "irene_loss_dict": irene_loss_dict,
            "aux_preds": aux_preds,
            "dropout_mask": dropout_mask,
        }
