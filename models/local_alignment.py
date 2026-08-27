import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAlignmentLoss(nn.Module):
    """
    One-way local alignment for clean per-modality tokens.

    For each subject and each present modality m, x_m queries the concatenated
    tokens from all other present modalities. The final loss is averaged per
    valid subject, then across valid subjects.
    """

    def __init__(self, temperature=0.1, exclude_cls=True):
        super().__init__()
        self.temperature = temperature
        self.exclude_cls = exclude_cls

    def _tokens_for_lia(self, x):
        if self.exclude_cls:
            return x[:, 1:]
        return x

    def _one_way_loss(self, query, context):
        query = F.normalize(query, dim=-1)
        context = F.normalize(context, dim=-1)

        atten_sim = torch.matmul(query, context.transpose(0, 1))
        atten_scores = F.softmax(atten_sim / self.temperature, dim=-1)
        attended_context = torch.matmul(atten_scores, context)
        attended_context = F.normalize(attended_context, dim=-1)

        token_sim = torch.matmul(query, attended_context.transpose(0, 1))
        token_sim = token_sim / self.temperature

        num_query_tokens = token_sim.shape[0]
        targets = torch.arange(num_query_tokens, device=query.device)

        loss_q2c = F.cross_entropy(token_sim, targets)
        loss_c2q = F.cross_entropy(token_sim.transpose(0, 1), targets)
        return (loss_q2c + loss_c2q) / 2.0

    def forward(self, tokens_list, modality_mask):
        if modality_mask is None:
            B = tokens_list[0].shape[0]
            M = len(tokens_list)
            modality_mask = torch.ones(
                B, M, dtype=torch.bool, device=tokens_list[0].device
            )
        else:
            modality_mask = modality_mask.to(device=tokens_list[0].device, dtype=torch.bool)

        clean_tokens = [self._tokens_for_lia(x) for x in tokens_list]
        B, _ = modality_mask.shape

        subject_losses = []
        for b in range(B):
            present = torch.where(modality_mask[b])[0].tolist()
            if len(present) < 2:
                continue

            query_losses = []
            for m in present:
                context_indices = [n for n in present if n != m]
                query = clean_tokens[m][b]
                context = torch.cat([clean_tokens[n][b] for n in context_indices], dim=0)
                query_losses.append(self._one_way_loss(query, context))

            subject_losses.append(torch.stack(query_losses).mean())

        if not subject_losses:
            zero = clean_tokens[0].sum() * 0.0
            valid_subjects = torch.zeros((), dtype=torch.long, device=clean_tokens[0].device)
            return zero, valid_subjects

        loss = torch.stack(subject_losses).mean()
        valid_subjects = torch.tensor(
            len(subject_losses), dtype=torch.long, device=clean_tokens[0].device
        )
        return loss, valid_subjects
