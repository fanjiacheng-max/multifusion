import torch.nn as nn
from torch.nn import LayerNorm

from .attention import Attention
from .mlp import Mlp


class Block(nn.Module):
    """
    mm=True  : 每模态独立 Pre-LN + 4-modal cross-attention + 独立 FFN
    mm=False : 共享 Pre-LN + self-attention + 共享 FFN
               (对 concat 后的 [B, sum(N_m), D] token 序列处理)
    """

    def __init__(self, config, vis, mm=False):
        super().__init__()
        self.mm = mm
        M = config.n_modalities
        D = config.hidden_size

        self.attn = Attention(config, vis, mm=mm)

        if mm:
            self.attention_norms = nn.ModuleList([LayerNorm(D, eps=1e-6) for _ in range(M)])
            self.ffn_norms       = nn.ModuleList([LayerNorm(D, eps=1e-6) for _ in range(M)])
            self.ffns            = nn.ModuleList([Mlp(config)             for _ in range(M)])
        else:
            self.attention_norm = LayerNorm(D, eps=1e-6)
            self.ffn_norm       = LayerNorm(D, eps=1e-6)
            self.ffn            = Mlp(config)

    def forward(self, tokens_list, modality_mask=None):
        if self.mm:
            return self._forward_mm(tokens_list, modality_mask)
        return self._forward_self(tokens_list, modality_mask)

    def _apply_modality_mask(self, tokens_list, modality_mask):
        if modality_mask is None:
            return tokens_list
        return [
            x * modality_mask[:, i].to(x.dtype).view(x.shape[0], 1, 1)
            for i, x in enumerate(tokens_list)
        ]

    def _apply_token_mask(self, x, token_mask):
        if token_mask is None:
            return x
        return x * token_mask.to(x.dtype).unsqueeze(-1)

    def _forward_mm(self, tokens_list, modality_mask):
        M = len(tokens_list)
        # Pre-LN → cross-modal attention → residual
        normed    = [self.attention_norms[i](tokens_list[i]) for i in range(M)]
        attn_outs, _ = self.attn(normed, modality_mask)
        post_attn = [tokens_list[i] + attn_outs[i] for i in range(M)]
        post_attn = self._apply_modality_mask(post_attn, modality_mask)

        # Pre-LN → FFN → residual
        out = [
            post_attn[i] + self.ffns[i](self.ffn_norms[i](post_attn[i]))
            for i in range(M)
        ]
        return self._apply_modality_mask(out, modality_mask)

    def _forward_self(self, x, token_mask=None):
        # Pre-LN → self-attn → residual
        h = x
        attn_out, _ = self.attn(self.attention_norm(x), token_mask)
        x = self._apply_token_mask(h + attn_out, token_mask)

        # Pre-LN → FFN → residual
        x = self._apply_token_mask(x + self.ffn(self.ffn_norm(x)), token_mask)
        return x
