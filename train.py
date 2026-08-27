"""
train.py — 四模态 CMR 年龄预测训练脚本 (IRENE-based v7)

用法:
  cd /mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp
  python -m multi_fusion.cmr_irene_v7.train \
      --out_dir outputs/cmr_irene_v7 \
      [--epochs 30] [--batch_size 64] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from multi_fusion.cmr_irene_v7.data.cmr_dataset import CMRTokenDataset, collate_fn
from multi_fusion.cmr_irene_v7.models.configs import CONFIGS
from multi_fusion.cmr_irene_v7.models.modeling_irene import CMRIrene

BASE = Path("/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp")
SPLIT_JSON  = str(BASE / "data/metadata/unified_split.json")
OUTCOME_CSV = str(BASE / "UKB_processed/age_at_scan_wide.csv")


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def compute_metrics(targets, preds):
    t = np.array(targets, dtype=np.float64)
    p = np.array(preds,   dtype=np.float64)
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    r2   = 1.0 - np.sum((t - p) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-9)
    corr = float(np.corrcoef(t, p)[0, 1]) if len(t) > 1 else 0.0
    return {'mae': mae, 'rmse': rmse, 'r2': float(r2), 'pearson_r': corr, 'n': len(t)}


# -----------------------------------------------------------------------
# One epoch
# -----------------------------------------------------------------------

def run_epoch(model, loader, optimizer, scaler, device,
              target_mean, target_std, is_train):
    model.train(is_train)
    all_t, all_p = [], []
    total_loss = 0.0
    total_reg_loss = 0.0
    total_lia_loss = 0.0
    total_lia_valid_subjects = 0

    for batch in loader:
        tokens = [t.to(device) for t in batch['tokens']]  # 4 × [B, 256, 1024]
        mask   = batch['mask'].to(device)                  # [B, 4]
        tgt    = batch['target'].to(device)                # [B]

        tgt_norm = (tgt - target_mean) / target_std

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                output = model(tokens, mask, target=tgt_norm)
                if len(output) == 3:
                    loss, pred_norm, loss_dict = output
                else:
                    loss, pred_norm = output
                    loss_dict = {}

        if is_train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        B = tgt.shape[0]
        lia_valid_subjects = int(
            loss_dict.get(
                'lia_valid_subjects',
                loss.new_zeros((), dtype=torch.long),
            ).item()
        )
        total_loss += loss.item() * B
        total_reg_loss += float(loss_dict.get('loss_reg', loss).item()) * B
        total_lia_loss += float(loss_dict.get('loss_lia', loss.new_zeros(())).item()) * lia_valid_subjects
        total_lia_valid_subjects += lia_valid_subjects
        pred_age = pred_norm.detach().float() * target_std + target_mean
        all_t.extend(tgt.cpu().tolist())
        all_p.extend(pred_age.cpu().tolist())

    metrics = compute_metrics(all_t, all_p)
    metrics['loss_reg'] = total_reg_loss / len(all_t)
    metrics['loss_lia'] = total_lia_loss / max(total_lia_valid_subjects, 1)
    metrics['loss'] = metrics['loss_reg'] + model.lambda_lia * metrics['loss_lia']
    metrics['lia_valid_subjects'] = total_lia_valid_subjects
    return metrics


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',    default='outputs/cmr_irene_v7')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--modality_dropout_p', type=float, default=0.1)
    parser.add_argument('--lambda_lia', type=float, default=0.1)
    parser.add_argument('--lia_temperature', type=float, default=0.1)
    parser.add_argument('--disable_lia', action='store_true')
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    ckpt_dir = out_dir / 'checkpoints'
    log_dir  = out_dir / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train] device={device}  out={out_dir}')

    # ---- Datasets (preload train+val into memory) ----
    ds_train = CMRTokenDataset(
        SPLIT_JSON, OUTCOME_CSV,
        splits=['train'],
        modality_dropout_p=args.modality_dropout_p,
        preload=True,
    )
    ds_val = CMRTokenDataset(
        SPLIT_JSON, OUTCOME_CSV,
        splits=['val'],
        modality_dropout_p=0.0,
        preload=True,
    )

    target_mean = float(np.mean([r[1] for r in ds_train.records]))
    target_std  = float(np.std( [r[1] for r in ds_train.records]))
    print(f'[train] age mean={target_mean:.2f}  std={target_std:.2f}')
    print(f'[train] modality_dropout_p={args.modality_dropout_p}')
    print(
        f'[train] lia={not args.disable_lia}  '
        f'lambda_lia={args.lambda_lia}  temperature={args.lia_temperature}'
    )
    json.dump({
        'target_mean': target_mean,
        'target_std': target_std,
        'modality_dropout_p': args.modality_dropout_p,
        'use_lia': not args.disable_lia,
        'lambda_lia': args.lambda_lia,
        'lia_temperature': args.lia_temperature,
    }, open(log_dir / 'startup_metrics.json', 'w'))

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(ds_train, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(ds_val,   shuffle=False, **loader_kw)

    # ---- Model ----
    config = CONFIGS['CMR_IRENE']
    config.use_lia = not args.disable_lia
    config.lambda_lia = args.lambda_lia
    config.lia_temperature = args.lia_temperature
    model  = CMRIrene(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[train] trainable params: {n_params/1e6:.1f}M')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # Linear warmup then cosine annealing
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=(device.type == 'cuda'))

    best_val_mae = float('inf')
    log_path = log_dir / 'epoch_metrics.jsonl'

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, scaler,
                            device, target_mean, target_std, is_train=True)
        val_m   = run_epoch(model, val_loader,   None,      scaler,
                            device, target_mean, target_std, is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:3d}/{args.epochs}  '
            f'train MAE={train_m["mae"]:.3f} R²={train_m["r2"]:.3f} '
            f'LIA={train_m["loss_lia"]:.4f}  |  '
            f'val MAE={val_m["mae"]:.3f} R²={val_m["r2"]:.3f} '
            f'LIA={val_m["loss_lia"]:.4f}  |  '
            f'{elapsed:.0f}s'
        )
        with open(log_path, 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'train': train_m, 'val': val_m}) + '\n')

        if val_m['mae'] < best_val_mae:
            best_val_mae = val_m['mae']
            torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')
            print(f'  ✓ best model saved  (val MAE={best_val_mae:.4f})')

    torch.save(model.state_dict(), ckpt_dir / 'last_model.pth')
    print(f'\n[train] done. best val MAE={best_val_mae:.4f}')


if __name__ == '__main__':
    main()
