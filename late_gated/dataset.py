"""
四模态原始数据 Dataset（端到端训练用，非预提取 token）
=======================================================
预处理逻辑与各单模态训练/extract_embeddings.py 完全一致；
训练时加 RandomHorizontalFlip（与单模态训练脚本的增强方式一致）。

Split 来源：visit1-only 各模态独立 split 文件（排除双访问受试者）：
  data/metadata/visit1only_sax_split.json      -> SA
  data/metadata/visit1only_lax4ch_split.json   -> LA
  data/metadata/visit1only_shmolli_split.json  -> T1
  data/metadata/visit1only_ao_split.json       -> AO
四份文件里同一 subject 的 split 归属（train/val/test）完全一致，
Dataset 取"某个 split 下四模态同时存在"的交集作为样本，并直接使用
每条记录自带的 npz_path（避免多次随访 NPZ 目录 glob 时取错文件的历史 bug）。

数据集无缺失模态（四模态交集）。
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

BASE = Path("/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp")

MODALITIES = ["SA", "LA", "T1", "AO"]

SPLIT_JSONS = {
    "SA": BASE / "data/metadata/visit1only_sax_split.json",
    "LA": BASE / "data/metadata/visit1only_lax4ch_split.json",
    "T1": BASE / "data/metadata/visit1only_shmolli_split.json",
    "AO": BASE / "data/metadata/visit1only_ao_split.json",
}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CROP_SIZE = 128
AO_RESIZE_SIZE = 256
AO_CROP_SIZE = 224
DEFAULT_VITA_Z = 6
DEFAULT_VITA_T = 50


def _crop_spatial(vol: torch.Tensor, is_train: bool) -> torch.Tensor:
    """
    vol: [Z, T, H, W]. Train uses one random spatial crop for the whole volume;
    val/test uses center crop. Z/T are left unchanged here.
    """
    H, W = vol.shape[2], vol.shape[3]
    ph = max(0, CROP_SIZE - H)
    pw = max(0, CROP_SIZE - W)
    if ph or pw:
        vol = F.pad(vol, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.0)
        H, W = vol.shape[2], vol.shape[3]

    if is_train:
        top = random.randint(0, H - CROP_SIZE)
        left = random.randint(0, W - CROP_SIZE)
    else:
        top = (H - CROP_SIZE) // 2
        left = (W - CROP_SIZE) // 2

    return vol[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]


def _resize_crop_2d(t: torch.Tensor, is_train: bool) -> torch.Tensor:
    """t: [C, H, W]. Resize to 256, then random/center crop to 224."""
    t = F.interpolate(
        t.unsqueeze(0), size=(AO_RESIZE_SIZE, AO_RESIZE_SIZE),
        mode="bilinear", align_corners=False,
    ).squeeze(0)

    if is_train:
        top = random.randint(0, AO_RESIZE_SIZE - AO_CROP_SIZE)
        left = random.randint(0, AO_RESIZE_SIZE - AO_CROP_SIZE)
    else:
        top = (AO_RESIZE_SIZE - AO_CROP_SIZE) // 2
        left = (AO_RESIZE_SIZE - AO_CROP_SIZE) // 2

    return t[:, top:top + AO_CROP_SIZE, left:left + AO_CROP_SIZE]


def _preprocess_vita(
    volume: np.ndarray,
    random_crop: bool = False,
    target_z: int = DEFAULT_VITA_Z,
    target_t: int = DEFAULT_VITA_T,
) -> torch.Tensor:
    """(H,W,Z,T) -> [1, target_z, target_t, 128, 128]."""
    if target_z <= 0 or target_t <= 0:
        raise ValueError(f"target_z and target_t must be positive, got {target_z=}, {target_t=}")

    vol = torch.from_numpy(volume.astype(np.float32)).permute(2, 3, 0, 1)
    Z, T, H, W = vol.shape
    vol = _crop_spatial(vol, is_train=random_crop)
    if Z >= target_z:
        vol = vol[(Z - target_z) // 2:(Z - target_z) // 2 + target_z]
    else:
        pad = torch.zeros(target_z, T, CROP_SIZE, CROP_SIZE)
        pad[(target_z - Z) // 2:(target_z - Z) // 2 + Z] = vol
        vol = pad
    T = vol.shape[1]
    if T > target_t:
        frame_idx = torch.linspace(0, T - 1, steps=target_t).round().long()
        vol = vol[:, frame_idx]
    elif T < target_t:
        vol = torch.cat(
            [vol, torch.zeros(target_z, target_t - T, CROP_SIZE, CROP_SIZE)],
            dim=1,
        )
    mn, mx = vol.amin(), vol.amax()
    if (mx - mn) > 1e-6:
        vol = (vol - mn) / (mx - mn)
    return vol.unsqueeze(0)   # [1, target_z, target_t, 128, 128]


def _preprocess_shmolli(volume: np.ndarray) -> torch.Tensor:
    Z = volume.shape[2]
    best_z = max(range(Z), key=lambda z: volume[:, :, z, :].mean())
    frames = volume[:, :, best_z, [0, 3, 6]].astype(np.float32).transpose(2, 0, 1)
    mn, mx = frames.min(), frames.max()
    if mx - mn > 1e-6:
        frames = (frames - mn) / (mx - mn)
    t = torch.from_numpy(frames).unsqueeze(0)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


def _preprocess_aortic(
    volume: np.ndarray,
    crop_256: bool = False,
    random_crop: bool = False,
) -> torch.Tensor:
    frames = volume[:, :, 0, [0, 33, 66]].astype(np.float32).transpose(2, 0, 1)
    mn, mx = frames.min(), frames.max()
    if mx - mn > 1e-6:
        frames = (frames - mn) / (mx - mn)
    t = torch.from_numpy(frames)
    if crop_256:
        t = _resize_crop_2d(t, is_train=random_crop)
    else:
        t = F.interpolate(t.unsqueeze(0), size=(224, 224),
                          mode="bilinear", align_corners=False).squeeze(0)
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


_PREPROCESS = {"SA": _preprocess_vita, "LA": _preprocess_vita,
               "T1": _preprocess_shmolli, "AO": _preprocess_aortic}


def _load_labels(csv_path: str, col: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            eid = row.get("eid", "").strip()
            val = row.get(col, "").strip()
            if eid and val:
                try: out[eid] = float(val)
                except ValueError: pass
    return out


def _load_modality_split(path: Path, split: str) -> Dict[str, str]:
    """返回 {subject_id: npz_path} for the given split ('train'/'val'/'test')"""
    with open(path) as f:
        sp = json.load(f)
    out = {}
    for rec in sp.get(split, []):
        sid = str(rec["subject_id"])
        out[sid] = rec["npz_path"]
    return out


class MultiModalRawDataset(Dataset):
    """
    返回:
      batch["SA"] : [1, 6, 50, 128, 128]
      batch["LA"] : [1, 6, 50, 128, 128]
      batch["T1"] : [3, 224, 224]
      batch["AO"] : [3, 224, 224]
      batch["age"]: scalar
      batch["eid"]: str
    """

    def __init__(
        self,
        outcome_csv: str,
        split:       str  = "train",
        target_col:  str  = "age_at_scan",
        augment:     bool = True,
        sa_la_random_crop: bool = False,
        ao_random_crop: bool = False,
        sa_target_z: int = DEFAULT_VITA_Z,
        sa_target_t: int = DEFAULT_VITA_T,
        la_target_z: int = DEFAULT_VITA_Z,
        la_target_t: int = DEFAULT_VITA_T,
        max_retries: int  = 3,
    ):
        self.augment     = augment and (split == "train")
        self.sa_la_random_crop = sa_la_random_crop and self.augment
        self.ao_crop_256 = ao_random_crop
        self.ao_random_crop = ao_random_crop and self.augment
        self.vita_shapes = {
            "SA": (sa_target_z, sa_target_t),
            "LA": (la_target_z, la_target_t),
        }
        self.max_retries = max_retries
        self.flip        = v2.RandomHorizontalFlip(p=0.5)

        labels = _load_labels(outcome_csv, target_col)

        # 各模态: {sid: npz_path}，均取自 visit1only 各模态独立 split 文件
        npz_paths = {
            m: _load_modality_split(SPLIT_JSONS[m], split)
            for m in MODALITIES
        }
        for m in MODALITIES:
            print(f"  {m}: {len(npz_paths[m])} subjects in split={split!r}")

        # 四模态交集
        common_sids = set(npz_paths["SA"])
        for m in MODALITIES[1:]:
            common_sids &= set(npz_paths[m])

        self.records: List[dict] = []
        for sid in common_sids:
            if sid not in labels:
                continue
            self.records.append({
                "sid": sid,
                "target": labels[sid],
                "npz_paths": {m: npz_paths[m][sid] for m in MODALITIES},
            })

        targets = [r["target"] for r in self.records]
        self.target_mean = float(np.mean(targets)) if targets else 0.0
        self.target_std  = float(np.std(targets))  if targets else 1.0
        print(f"[MultiModalRawDataset] split={split!r}  n={len(self.records)}"
              f"  mean={self.target_mean:.2f}  std={self.target_std:.2f}")

    def __len__(self): return len(self.records)

    def _load_npz(self, modality: str, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as d:
            vol = d["volume"]
        if modality in ("SA", "LA"):
            target_z, target_t = self.vita_shapes[modality]
            return _preprocess_vita(
                vol,
                random_crop=self.sa_la_random_crop,
                target_z=target_z,
                target_t=target_t,
            )
        if modality == "AO":
            return _preprocess_aortic(
                vol, crop_256=self.ao_crop_256,
                random_crop=self.ao_random_crop,
            )
        return _PREPROCESS[modality](vol)

    def _load_one(self, idx: int):
        rec = self.records[idx]
        sid = rec["sid"]
        try:
            out = {}
            for m in MODALITIES:
                t = self._load_npz(m, rec["npz_paths"][m])
                if t is None: return None
                if self.augment:
                    t = self.flip(t)
                out[m] = t
            out["age"] = torch.tensor(rec["target"], dtype=torch.float32)
            out["eid"] = sid
            return out
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict:
        for _ in range(self.max_retries + 1):
            item = self._load_one(idx)
            if item is not None: return item
            idx = random.randint(0, len(self.records) - 1)
        raise RuntimeError(f"Failed to load after {self.max_retries} retries")
