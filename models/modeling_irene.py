import torch
import torch.nn as nn

from .embed import Embeddings
from .encoder import Encoder
from .local_alignment import LocalAlignmentLoss
from . import configs


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder    = Encoder(config, vis=False)

    def forward(self, tokens_list, modality_mask, return_embedded=False):
        embedded = self.embeddings(tokens_list)          # list of 4 × [B, 257, 768]
        encoded, token_mask = self.encoder(embedded, modality_mask) # [B, 4*257, 768]
        if return_embedded:
            return encoded, token_mask, embedded
        return encoded, token_mask


class CMRIrene(nn.Module):
    """
    四模态 CMR 年龄预测。

    输入:
      tokens_list  : list of 4 × [B, 256, 1024]  (pre-extracted tokens, float32)
      modality_mask: [B, 4] bool  (False = absent / dropped)
      target       : [B] float32  (z-score normalized age, optional)

    输出 (target=None): [B] pred_age_normalized
    输出 (target!=None): (loss, pred)
    """

    MODALITY_NAMES = ['SA', 'LA', 'T1', 'Aortic']

    def __init__(self, config):
        super().__init__()
        D = config.hidden_size   # 768

        self.transformer = Transformer(config)
        self.use_lia = bool(getattr(config, "use_lia", False))
        self.lambda_lia = float(getattr(config, "lambda_lia", 0.0))
        if self.use_lia:
            self.lia_loss = LocalAlignmentLoss(
                temperature=float(getattr(config, "lia_temperature", 0.1)),
                exclude_cls=bool(getattr(config, "lia_exclude_cls", True)),
            )
        else:
            self.lia_loss = None

        self.head = nn.Sequential(
            nn.Linear(D, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.loss_fn = nn.HuberLoss()

    def forward(self, tokens_list, modality_mask, target=None):
        need_lia = target is not None and self.use_lia and self.lambda_lia > 0.0
        if need_lia:
            encoded, token_mask, embedded = self.transformer(
                tokens_list, modality_mask, return_embedded=True
            )
            loss_lia, lia_valid_subjects = self.lia_loss(embedded, modality_mask)
        else:
            encoded, token_mask = self.transformer(tokens_list, modality_mask)
            loss_lia = encoded.sum() * 0.0
            lia_valid_subjects = torch.zeros((), dtype=torch.long, device=encoded.device)

        # encoded: [B, 4*257, 768], matching original IRENE's fused sequence.
        if token_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            weights = token_mask.to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1)
            pooled = pooled / weights.sum(dim=1).clamp_min(1.0)

        pred = self.head(pooled).squeeze(-1)        # [B]

        if target is not None:
            loss_reg = self.loss_fn(pred, target)
            loss = loss_reg + self.lambda_lia * loss_lia
            loss_dict = {
                'loss_reg': loss_reg.detach(),
                'loss_lia': loss_lia.detach(),
                'lia_valid_subjects': lia_valid_subjects.detach(),
            }
            return loss, pred, loss_dict
        return pred


CONFIGS = {k: v for k, v in configs.CONFIGS.items()}
