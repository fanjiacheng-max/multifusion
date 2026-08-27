import math

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, Softmax


class Attention(nn.Module):
    """
    mm=True  : 4-modal cross-attention block
               每个模态独立 QKV 投影；self-attn + 3个 cross-attn 平均，
               (self + cross_avg) / 2 后送 out_proj。
               per-sample modality mask：source 缺失时 cross contribution = 0。

    mm=False : 普通 self-attention（共享权重，在 concat 后的 [B, sum(N_m), D] 操作）
    """

    def __init__(self, config, vis, mm=False):
        super().__init__()
        self.vis = vis
        self.mm  = mm
        M = config.n_modalities
        D = config.hidden_size
        self.num_heads = config.transformer["num_heads"]
        self.head_size = D // self.num_heads
        self.all_head_size = self.num_heads * self.head_size

        if mm:
            self.query = nn.ModuleList([Linear(D, D) for _ in range(M)])
            self.key   = nn.ModuleList([Linear(D, D) for _ in range(M)])
            self.value = nn.ModuleList([Linear(D, D) for _ in range(M)])
            self.out   = nn.ModuleList([Linear(D, D) for _ in range(M)])
        else:
            self.query = Linear(D, D)
            self.key   = Linear(D, D)
            self.value = Linear(D, D)
            self.out   = Linear(D, D)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tr(self, x):
        """[B, N, D] → [B, heads, N, head_size]"""
        B, N, _ = x.shape
        x = x.view(B, N, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)

    def _attn(self, Q, K, V, attention_mask=None):
        """Scaled dot-product attention → [B, N, D]"""
        s = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=s.device, dtype=torch.bool)
            key_mask = attention_mask[:, None, None, :]
            s = s.masked_fill(~key_mask, torch.finfo(s.dtype).min)
        p = self.softmax(s)
        w = p if self.vis else None
        p = self.attn_dropout(p)
        c = torch.matmul(p, V)
        c = c.permute(0, 2, 1, 3).contiguous()
        c = c.view(c.shape[0], c.shape[1], self.all_head_size)
        return c, w

    # ------------------------------------------------------------------
    # mm forward: 4-modal cross-attention
    # ------------------------------------------------------------------

    def _forward_mm(self, tokens_list, modality_mask):
        """
        tokens_list  : list of M × [B, N, D]
        modality_mask: [B, M] bool
        返回: list of M × [B, N, D], attn_weights
        """
        M   = len(tokens_list)
        B   = tokens_list[0].shape[0]
        device = tokens_list[0].device
        if modality_mask is None:
            modality_mask = torch.ones(B, M, dtype=torch.bool, device=device)
        else:
            modality_mask = modality_mask.to(device=device, dtype=torch.bool)

        outputs = []
        weights = None

        for i in range(M):
            X_i = tokens_list[i]              # [B, N, D]

            # Q shared between self-attn and all cross-attns for target i
            Q_i = self._tr(self.query[i](X_i))
            K_i = self._tr(self.key[i](X_i))
            V_i = self._tr(self.value[i](X_i))

            self_ctx, w = self._attn(Q_i, K_i, V_i)   # [B, N, D]
            if i == 0:
                weights = w

            # Cross-attention with available source modalities only.
            cross_sum = torch.zeros_like(self_ctx)
            source_count = torch.zeros(B, 1, 1, device=device, dtype=self_ctx.dtype)
            for j in range(M):
                if j == i:
                    continue
                K_j = self._tr(self.key[j](tokens_list[j]))
                V_j = self._tr(self.value[j](tokens_list[j]))
                cross_ctx, _ = self._attn(Q_i, K_j, V_j)
                source_present = modality_mask[:, j].to(self_ctx.dtype).view(B, 1, 1)
                cross_sum = cross_sum + cross_ctx * source_present
                source_count = source_count + source_present

            cross_avg = cross_sum / source_count.clamp_min(1.0)

            has_cross = source_count > 0
            combined = torch.where(has_cross, (self_ctx + cross_avg) / 2, self_ctx)

            out_i = self.proj_dropout(self.out[i](combined))
            target_present = modality_mask[:, i].to(out_i.dtype).view(B, 1, 1)
            out_i = out_i * target_present
            outputs.append(out_i)

        return outputs, weights

    # ------------------------------------------------------------------
    # Standard self-attention (mm=False, concatenated token sequence in)
    # ------------------------------------------------------------------

    def _forward_self(self, x, attention_mask=None):
        """x: [B, N_total, D] → [B, N_total, D]"""
        Q = self._tr(self.query(x))
        K = self._tr(self.key(x))
        V = self._tr(self.value(x))
        ctx, w = self._attn(Q, K, V, attention_mask=attention_mask)
        out = self.proj_dropout(self.out(ctx))
        return out, w

    # ------------------------------------------------------------------

    def forward(self, hidden_states, modality_mask=None):
        if self.mm:
            return self._forward_mm(hidden_states, modality_mask)
        return self._forward_self(hidden_states, modality_mask)
