"""
eval.py — IRENE v7 test 集评估脚本

用法:
  cd /mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp
  python -m multi_fusion.cmr_irene_v7.eval \
      --ckpt outputs/cmr_irene_v7/checkpoints/best_model.pth \
      --split test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multi_fusion.cmr_irene_v7.data.cmr_dataset import CMRTokenDataset, collate_fn
from multi_fusion.cmr_irene_v7.models.configs import CONFIGS
from multi_fusion.cmr_irene_v7.models.modeling_irene import CMRIrene

BASE        = Path("/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp")
SPLIT_JSON  = str(BASE / "data/metadata/unified_split.json")
OUTCOME_CSV = str(BASE / "UKB_processed/age_at_scan_wide.csv")


def compute_metrics(targets, preds):
    t = np.array(targets, dtype=np.float64)
    p = np.array(preds,   dtype=np.float64)
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    r2   = 1.0 - np.sum((t - p) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-9)
    corr = float(np.corrcoef(t, p)[0, 1]) if len(t) > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": float(r2), "pearson_r": corr, "n": len(t)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--split",      default="test")
    parser.add_argument("--out_dir",    default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir   = Path(args.out_dir) if args.out_dir else ckpt_path.parents[1] / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    startup = json.load(open(ckpt_path.parents[1] / "logs" / "startup_metrics.json"))
    target_mean = startup["target_mean"]
    target_std  = startup["target_std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] split={args.split}  device={device}")

    ds = CMRTokenDataset(
        SPLIT_JSON, OUTCOME_CSV,
        splits=[args.split],
        modality_dropout_p=0.0,
        preload=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn,
                        pin_memory=(device.type == "cuda"))

    config = CONFIGS["CMR_IRENE"]
    config.use_lia = startup.get("use_lia", True)
    config.lambda_lia = startup.get("lambda_lia", 0.1)
    config.lia_temperature = startup.get("lia_temperature", 0.1)
    model = CMRIrene(config).to(device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[Eval] {len(ds)} samples")

    all_t, all_p, all_eids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            tokens = [t.to(device) for t in batch["tokens"]]
            mask   = batch["mask"].to(device)
            tgt    = batch["target"]

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                pred_norm = model(tokens, mask, target=None)

            pred_age = pred_norm.float().cpu() * target_std + target_mean
            all_t.extend(tgt.tolist())
            all_p.extend(pred_age.tolist())
            all_eids.extend(batch["eid"])

    metrics = compute_metrics(all_t, all_p)
    print(f"\n{'='*55}")
    print(f"  {args.split.upper()}  N={metrics['n']}  "
          f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
          f"R²={metrics['r2']:.4f}  r={metrics['pearson_r']:.4f}")
    print(f"{'='*55}")

    json.dump(metrics, open(out_dir / "test_metrics.json", "w"), indent=2)
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eid", "target", "pred", "abs_error"])
        for eid, t, p in zip(all_eids, all_t, all_p):
            w.writerow([eid, f"{t:.4f}", f"{p:.4f}", f"{abs(p-t):.4f}"])

    print(f"[Eval] Saved to {out_dir}")


if __name__ == "__main__":
    main()
