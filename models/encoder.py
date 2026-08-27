import torch
import torch.nn as nn
from torch.nn import LayerNorm

from .block import Block


class Encoder(nn.Module):
    """
    前 mm_layers 层: 4-modal cross-attention (mm=True)
    剩余层:         concat modalities as tokens, then shared self-attention
    """

    def __init__(self, config, vis):
        super().__init__()
        D = config.hidden_size

        self.layers = nn.ModuleList()
        for i in range(config.transformer["num_layers"]):
            self.layers.append(Block(config, vis, mm=(i < config.mm_layers)))
        self.mm_layers = config.mm_layers

        self.encoder_norm = LayerNorm(D, eps=1e-6)

    def _make_token_mask(self, modality_mask, seq_lens):
        if modality_mask is None:
            return None
        return torch.cat([
            modality_mask[:, i].unsqueeze(1).expand(-1, seq_len)
            for i, seq_len in enumerate(seq_lens)
        ], dim=1)

    def forward(self, tokens_list, modality_mask):
        token_mask = None
        hidden_states = tokens_list

        for i, layer in enumerate(self.layers):
            if i < self.mm_layers:
                hidden_states = layer(hidden_states, modality_mask)
                continue

            if isinstance(hidden_states, list):
                seq_lens = [x.shape[1] for x in hidden_states]
                token_mask = self._make_token_mask(modality_mask, seq_lens)
                hidden_states = torch.cat(hidden_states, dim=1)

            hidden_states = layer(hidden_states, token_mask)

        if isinstance(hidden_states, list):
            seq_lens = [x.shape[1] for x in hidden_states]
            token_mask = self._make_token_mask(modality_mask, seq_lens)
            hidden_states = torch.cat(hidden_states, dim=1)

        encoded = self.encoder_norm(hidden_states)
        if token_mask is not None:
            encoded = encoded * token_mask.to(encoded.dtype).unsqueeze(-1)
        return encoded, token_mask
