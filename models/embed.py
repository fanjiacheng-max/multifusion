import torch
import torch.nn as nn
from torch.nn import Dropout, Linear


class Embeddings(nn.Module):
    """
    把 4 个模态的 pre-extracted tokens 投影到 hidden_size，
    各加 cls_token 和 position embedding。

    输入: list of 4 × [B, 256, 1024] float32
    输出: list of 4 × [B, 257, 768]
    """

    def __init__(self, config):
        super().__init__()
        D = config.hidden_size   # 768
        C = config.token_dim     # 1024
        N = 256                  # tokens per modality
        M = config.n_modalities  # 4

        self.proj      = nn.ModuleList([Linear(C, D) for _ in range(M)])
        self.cls_token = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, D)) for _ in range(M)]
        )
        self.pos_embed = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, N + 1, D)) for _ in range(M)]
        )
        self.dropout = Dropout(config.transformer["dropout_rate"])

        for m in range(M):
            nn.init.trunc_normal_(self.pos_embed[m], std=0.02)
            nn.init.trunc_normal_(self.cls_token[m], std=0.02)

    def forward(self, tokens_list):
        outputs = []
        for i, tokens in enumerate(tokens_list):
            B = tokens.shape[0]
            x   = self.proj[i](tokens)                      # [B, 256, 768]
            cls = self.cls_token[i].expand(B, -1, -1)       # [B,   1, 768]
            x   = torch.cat([cls, x], dim=1)                # [B, 257, 768]
            x   = x + self.pos_embed[i]
            x   = self.dropout(x)
            outputs.append(x)
        return outputs
