"""
CMRTokenDataset
================
从 token_embeddings/ 预加载四模态 pre-extracted tokens 到内存。

每个 .npy 文件形状: (256, 1024) float16
模态顺序: SA / LA(4ch) / T1(ShMOLLI) / Aortic

Modality dropout (训练时):
  - 每个模态以 p=modality_dropout_p 独立被 drop
  - 若全部被 drop，随机保留一个原本可用的模态
"""

import csv
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

BASE = Path("/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp")

TOKEN_ROOTS = {
    'sa':     BASE / 'token_embeddings' / 'sa',
    'la':     BASE / 'token_embeddings' / 'la_4ch',
    't1':     BASE / 'token_embeddings' / 't1',
    'aortic': BASE / 'token_embeddings' / 'aortic',
}
MOD_KEYS = ['sa', 'la', 't1', 'aortic']

_ZERO = torch.zeros(256, 1024, dtype=torch.float32)


def _load_labels(csv_path: str, col: str):
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            eid = row.get('eid', '').strip()
            val = row.get(col, '').strip()
            if eid and val:
                try:
                    out[eid] = float(val)
                except ValueError:
                    pass
    return out


class CMRTokenDataset(Dataset):
    def __init__(
        self,
        split_json: str,
        outcome_csv: str,
        splits: List[str],           # ['train'] or ['train', 'val']
        target_col: str = 'age_at_scan',
        modality_dropout_p: float = 0.0,   # >0 only for training
        preload: bool = True,
    ):
        self.dropout_p = modality_dropout_p

        sp     = json.load(open(split_json))
        labels = _load_labels(outcome_csv, target_col)

        # Build subject list
        sids = []
        for split in splits:
            sids.extend(sp.get(split, []))
        sids = [str(s['subject_id'] if isinstance(s, dict) else s) for s in sids]

        # Path.iterdir() returns Path objects → use .stem (not .replace)
        avail = {
            m: set(p.stem for p in TOKEN_ROOTS[m].iterdir() if p.suffix == '.npy')
            for m in MOD_KEYS
        }

        self.records = []   # (sid, target, base_mask: list[bool])
        for sid in sids:
            if sid not in labels:
                continue
            mask = [sid in avail[m] for m in MOD_KEYS]
            if not any(mask):
                continue
            self.records.append((sid, labels[sid], mask))

        print(f"[CMRTokenDataset] splits={splits}  n={len(self.records)}  "
              f"dropout_p={modality_dropout_p}", flush=True)

        # Preload into memory (only available modalities stored)
        self.cache: Optional[dict] = None
        if preload:
            self.cache = {}
            for sid, _, mask in tqdm(self.records, desc='preloading tokens'):
                self.cache[sid] = {}
                for i, m in enumerate(MOD_KEYS):
                    if mask[i]:
                        arr = np.load(TOKEN_ROOTS[m] / f'{sid}.npy')
                        # keep float16 in memory (~126GB vs ~252GB for float32)
                        self.cache[sid][m] = torch.from_numpy(arr)  # float16

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sid, target, base_mask = self.records[idx]
        mask = list(base_mask)

        # Modality dropout
        if self.dropout_p > 0:
            drop     = [random.random() < self.dropout_p for _ in range(4)]
            new_mask = [m and not d for m, d in zip(mask, drop)]
            if any(new_mask):
                mask = new_mask
            else:
                # all dropped — restore one random available modality
                valid  = [i for i, m in enumerate(mask) if m]
                restore = random.choice(valid)
                new_mask[restore] = True
                mask = new_mask

        tokens_list = []
        for i, m in enumerate(MOD_KEYS):
            if mask[i]:
                if self.cache is not None:
                    t = self.cache[sid].get(m)
                else:
                    p = TOKEN_ROOTS[m] / f'{sid}.npy'
                    t = torch.from_numpy(np.load(p)) if p.exists() else None
                if t is not None:
                    tokens_list.append(t.float())  # float16 → float32
                    continue
            tokens_list.append(_ZERO.clone())
            mask[i] = False

        return {
            'tokens': tokens_list,                           # list of 4 × [256, 1024]
            'mask':   torch.tensor(mask, dtype=torch.bool),  # [4]
            'target': torch.tensor(target, dtype=torch.float32),
            'eid':    sid,
        }


def collate_fn(batch):
    """Custom collate: list-of-tensors → list of 4 stacked tensors."""
    tokens = [
        torch.stack([item['tokens'][i] for item in batch], dim=0)
        for i in range(4)
    ]  # list of 4 × [B, 256, 1024]
    return {
        'tokens': tokens,
        'mask':   torch.stack([item['mask']   for item in batch]),  # [B, 4]
        'target': torch.stack([item['target'] for item in batch]),  # [B]
        'eid':    [item['eid'] for item in batch],
    }
