"""
End-to-end IRENE fusion 训练脚本
================================
true-frozen then soft fine-tuning：
  - frozen 阶段四个 backbone requires_grad=False，且不进入 optimizer
  - 超过 frozen 阶段后解冻 backbone，并重建 optimizer
  - token bridge / IRENE fusion / aux head 使用 lr_head
  - backbone 使用较小的 lr_backbone
  - WORLD_SIZE>1 时自动启用 DDP

用法:
  python -m multi_fusion.cmr_irene_v7.late_gated.train_end2end \
      --out_dir outputs/cmr_irene_v7_end2end
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from multi_fusion.cmr_irene_v7.late_gated.dataset import MultiModalRawDataset, MODALITIES
from multi_fusion.cmr_irene_v7.late_gated.model_end2end import EndToEndMultiModalAgeRegressor
from multimode_train.train_four_modal_mult import compute_metrics, append_jsonl

BASE        = Path("/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp")
OUTCOME_CSV = str(BASE / "UKB_processed/age_at_scan_wide.csv")

SA_CKPT = str(BASE / "outputs/clean/sax/age_at_scan/checkpoints/best_model.pth")
LA_CKPT = str(BASE / "outputs/clean/lax4ch/age_at_scan/checkpoints/best_model.pth")
T1_CKPT = str(BASE / "outputs/clean_dinov2/shmolli/checkpoints/best_model.pth")
AO_CKPT = str(BASE / "outputs/clean_dinov2/aortic/checkpoints/best_model.pth")


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def gather_targets_preds(targets, preds):
    if not (dist.is_available() and dist.is_initialized()):
        return targets, preds

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, (targets, preds))
    if not is_main_process():
        return [], []

    all_t, all_p = [], []
    for t, p in gathered:
        all_t.extend(t)
        all_p.extend(p)
    return all_t, all_p


class DistributedEvalSampler(torch.utils.data.Sampler):
    """Sequential distributed sampler without padding or duplicated eval samples."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.indices = list(range(self.rank, len(self.dataset), self.world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def resolve_amp(mode: str, device: torch.device):
    if mode == "off" or device.type != "cuda":
        return False, None, None
    if mode == "bf16":
        return True, torch.bfloat16, None
    if mode == "fp16":
        return True, torch.float16, torch.amp.GradScaler("cuda")
    if mode == "auto":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, None
        return True, torch.float16, torch.amp.GradScaler("cuda")
    raise ValueError(f"Unsupported AMP mode: {mode}")


def run_epoch(model, loader, optimizer, device, target_mean, target_std,
              is_train, lambda_aux, use_aux, amp_enabled=False,
              amp_dtype=None, scaler=None, main_process=True):
    model.train(is_train)
    aux_loss_fn = nn.SmoothL1Loss()
    all_t, all_p = [], []
    total_irene, total_reg, total_lia, total_aux = 0.0, 0.0, 0.0, 0.0
    total_lia_valid_subjects = 0

    for batch in tqdm(loader, desc="train" if is_train else "val", leave=False,
                      disable=not main_process):
        inputs = {m: batch[m].to(device) for m in MODALITIES}
        tgt    = batch["age"].to(device)
        tgt_n  = (tgt - target_mean) / target_std

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                out = model(inputs, target=tgt_n)
                irene_loss = out["irene_loss"]
                loss_dict = out.get("irene_loss_dict", {})
                reg_loss = loss_dict.get("loss_reg", irene_loss)
                lia_loss = loss_dict.get("loss_lia", irene_loss.new_zeros(()))

                aux_loss = torch.tensor(0.0, device=device)
                if use_aux and out["aux_preds"]:
                    for v in out["aux_preds"].values():
                        aux_loss = aux_loss + aux_loss_fn(v.squeeze(-1), tgt_n)
                    aux_loss = aux_loss / len(out["aux_preds"])

                loss = irene_loss + lambda_aux * aux_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        lia_valid_subjects = int(
            loss_dict.get(
                "lia_valid_subjects",
                irene_loss.new_zeros((), dtype=torch.long),
            ).item()
        )
        total_irene += irene_loss.item() * len(tgt)
        total_reg += float(reg_loss.item()) * len(tgt)
        total_lia += float(lia_loss.item()) * lia_valid_subjects
        total_lia_valid_subjects += lia_valid_subjects
        total_aux += float(aux_loss) * len(tgt)
        pred_age = out["age_pred"].detach().squeeze(-1) * target_std + target_mean
        all_t.extend(tgt.cpu().tolist())
        all_p.extend(pred_age.cpu().tolist())

    gathered_t, gathered_p = gather_targets_preds(all_t, all_p)
    if main_process:
        local_n = max(len(all_t), 1)
        loss_payload = torch.tensor(
            [
                total_irene,
                total_reg,
                total_lia,
                total_lia_valid_subjects,
                total_aux,
                local_n,
            ],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_payload, op=dist.ReduceOp.SUM)
        metrics = compute_metrics(gathered_t, gathered_p)
        n_total = max(loss_payload[5].item(), 1.0)
        lia_n = max(loss_payload[3].item(), 1.0)
        metrics["irene_loss"] = float(loss_payload[0].item() / n_total)
        metrics["loss_reg"] = float(loss_payload[1].item() / n_total)
        metrics["loss_lia"] = float(loss_payload[2].item() / lia_n)
        metrics["lia_valid_subjects"] = int(loss_payload[3].item())
        metrics["aux_loss"] = float(loss_payload[4].item() / n_total)
        metrics["loss"] = metrics["irene_loss"] + lambda_aux * metrics["aux_loss"]
        return metrics

    loss_payload = torch.tensor(
        [
            total_irene,
            total_reg,
            total_lia,
            total_lia_valid_subjects,
            total_aux,
            max(len(all_t), 1),
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_payload, op=dist.ReduceOp.SUM)
    return {}


def build_optimizer(model, lr_head, lr_backbone, weight_decay, include_backbone):
    model = unwrap_model(model)
    backbone_params = list(model.backbones.parameters())
    other_params = (
        list(model.token_reducers.parameters())
        + list(model.irene.parameters())
        + (list(model.aux_heads.parameters()) if model.use_deep_supervision else [])
    )
    groups = [{"params": other_params, "lr": lr_head, "name": "head"}]
    if include_backbone:
        groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
    return optim.AdamW(groups, weight_decay=weight_decay)


def scheduled_backbone_lr(epoch, zero_epochs, lr_backbone):
    return 0.0 if epoch <= zero_epochs else lr_backbone


def scheduled_stage(epoch, frozen_epochs, zero_epochs):
    if epoch <= frozen_epochs:
        return "true_frozen"
    if epoch <= zero_epochs:
        return "backbone_lr0_warmup"
    return "soft_finetune"


def should_include_backbone(epoch, frozen_epochs):
    return epoch > frozen_epochs


def set_backbones_trainable(model, flag):
    unwrap_model(model).set_backbones_trainable(flag)


def optimizer_has_backbone(optimizer):
    return any(group.get("name") == "backbone" for group in optimizer.param_groups)


def set_backbone_lr(optimizer, lr):
    found = False
    for group in optimizer.param_groups:
        if group.get("name") == "backbone":
            group["lr"] = lr
            found = True
    if not found and len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr"] = lr


def get_group_lr(optimizer, name):
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    fallback_idx = {"head": 0, "backbone": 1}.get(name)
    if fallback_idx is not None and fallback_idx < len(optimizer.param_groups):
        return float(optimizer.param_groups[fallback_idx]["lr"])
    return None


def format_lr(lr):
    return "none" if lr is None else f"{lr:.2e}"


def save_resume_state(
    path,
    epoch,
    model,
    optimizer,
    scaler,
    best_val_mae,
    best_epoch,
    epochs_without_improvement,
    args,
):
    state = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "args": vars(args),
    }
    torch.save(state, path)


def load_resume_state(state, model, optimizer, scaler):
    unwrap_model(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return (
        int(state["epoch"]) + 1,
        float(state.get("best_val_mae", float("inf"))),
        int(state.get("best_epoch", 0)),
        int(state.get("epochs_without_improvement", 0)),
    )


def main():
    distributed, local_rank, rank, world_size = setup_distributed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",       default="outputs/cmr_irene_v7_end2end_sa6t50_la1_single_stage_bb1e5_md0p1_noaocrop")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--frozen_epochs", type=int,   default=0,
                        help="True-freeze backbones for the first N epochs: "
                             "requires_grad=False and excluded from optimizer.")
    parser.add_argument("--batch_size",    type=int,   default=8,
                        help="Per-GPU batch size when using DDP.")
    parser.add_argument("--lr_head",       type=float, default=1e-4)
    parser.add_argument("--lr_backbone",   type=float, default=1e-5)
    parser.add_argument("--backbone_lr_zero_epochs", type=int, default=0,
                        help="Keep backbone optimizer lr at 0 for the first N epochs; "
                             "keeps DDP/optimizer structure unchanged.")
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--modality_dropout_p", type=float, default=0.1)
    parser.add_argument("--lambda_lia", type=float, default=0.1)
    parser.add_argument("--lia_temperature", type=float, default=0.1)
    parser.add_argument("--disable_lia", action="store_true")
    parser.add_argument("--lambda_aux_frozen",   type=float, default=0.05)
    parser.add_argument("--lambda_aux_unfrozen", type=float, default=0.05)
    parser.add_argument("--no_deep_sup",   action="store_true")
    parser.add_argument("--sa_la_random_crop", dest="sa_la_random_crop",
                        action="store_true", default=True)
    parser.add_argument("--no_sa_la_random_crop", "--no-sa_la_random_crop",
                        dest="sa_la_random_crop", action="store_false")
    parser.add_argument("--ao_random_crop", dest="ao_random_crop",
                        action="store_true", default=False)
    parser.add_argument("--no_ao_random_crop", "--no-ao_random_crop",
                        dest="ao_random_crop", action="store_false")
    parser.add_argument("--sa_target_z", type=int, default=6)
    parser.add_argument("--sa_target_t", type=int, default=50)
    parser.add_argument("--la_target_z", type=int, default=1)
    parser.add_argument("--la_target_t", type=int, default=50)
    parser.add_argument("--amp", choices=("auto", "bf16", "fp16", "off"), default="auto")
    parser.add_argument("--early_stop_patience", type=int, default=8,
                        help="Stop after this many val epochs without MAE improvement; <=0 disables.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="Minimum val MAE improvement required to reset early-stop patience.")
    parser.add_argument("--resume", default=None,
                        help="Path to resume_state.pth for exact training resume.")
    parser.add_argument("--workers",       type=int, default=4)
    args = parser.parse_args()
    if args.frozen_epochs < 0:
        raise ValueError("--frozen_epochs must be >= 0")
    if args.backbone_lr_zero_epochs < 0:
        raise ValueError("--backbone_lr_zero_epochs must be >= 0")

    sa_img_shape = (args.sa_target_z, args.sa_target_t, 128, 128)
    la_img_shape = (args.la_target_z, args.la_target_t, 128, 128)

    out_dir  = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = out_dir / "logs";        log_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype, scaler = resolve_amp(args.amp, device)
    amp_dtype_name = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none"
    main_process = is_main_process()
    if main_process:
        print(f"[Train] device={device}  out={out_dir}  amp={args.amp}/{amp_dtype_name}  "
              f"distributed={distributed} world_size={world_size} per_gpu_batch={args.batch_size}")
        if args.resume:
            print(f"[Train] resume={args.resume}")

    ds_train = MultiModalRawDataset(
        OUTCOME_CSV, split="train", augment=True,
        sa_la_random_crop=args.sa_la_random_crop,
        ao_random_crop=args.ao_random_crop,
        sa_target_z=args.sa_target_z,
        sa_target_t=args.sa_target_t,
        la_target_z=args.la_target_z,
        la_target_t=args.la_target_t,
    )
    ds_val = MultiModalRawDataset(
        OUTCOME_CSV, split="val", augment=False,
        ao_random_crop=args.ao_random_crop,
        sa_target_z=args.sa_target_z,
        sa_target_t=args.sa_target_t,
        la_target_z=args.la_target_z,
        la_target_t=args.la_target_t,
    )
    target_mean, target_std = ds_train.target_mean, ds_train.target_std

    use_aux = not args.no_deep_sup
    if args.frozen_epochs > 0:
        training_strategy = "true_frozen_then_soft_finetune_irene_fusion"
    elif args.backbone_lr_zero_epochs > 0:
        training_strategy = "backbone_lr0_warmup_then_soft_finetune_irene_fusion"
    else:
        training_strategy = "single_stage_soft_finetune_irene_fusion"

    if main_process:
        json.dump({
            "target_mean": target_mean,
            "target_std": target_std,
            "sa_la_random_crop": args.sa_la_random_crop,
            "ao_random_crop": args.ao_random_crop,
            "sa_target_z": args.sa_target_z,
            "sa_target_t": args.sa_target_t,
            "la_target_z": args.la_target_z,
            "la_target_t": args.la_target_t,
            "sa_img_shape": list(sa_img_shape),
            "la_img_shape": list(la_img_shape),
            "amp": args.amp,
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype_name,
            "distributed": distributed,
            "world_size": world_size,
            "ddp_find_unused_parameters": distributed,
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "training_strategy": training_strategy,
            "frozen_epochs": args.frozen_epochs,
            "backbones_trainable_from_epoch": args.frozen_epochs + 1,
            "backbone_optimizer_includes_from_epoch": args.frozen_epochs + 1,
            "backbone_lr_zero_epochs": args.backbone_lr_zero_epochs,
            "backbone_lr_after_zero": args.lr_backbone,
            "backbone_weight_updates_from_epoch": (
                max(args.frozen_epochs, args.backbone_lr_zero_epochs) + 1
                if args.lr_backbone > 0 else None
            ),
            "use_deep_supervision": use_aux,
            "use_lia": not args.disable_lia,
            "lambda_lia": args.lambda_lia,
            "lia_temperature": args.lia_temperature,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "resume": args.resume,
        }, open(log_dir / "startup_metrics.json", "w"), indent=2)

    lkw = dict(batch_size=args.batch_size, num_workers=args.workers,
               pin_memory=(device.type == "cuda"), persistent_workers=(args.workers > 0))
    train_sampler = DistributedSampler(ds_train, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedEvalSampler(ds_val) if distributed else None
    train_loader = DataLoader(
        ds_train,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=distributed,
        **lkw,
    )
    val_loader = DataLoader(
        ds_val,
        shuffle=False,
        sampler=val_sampler,
        **lkw,
    )

    model = EndToEndMultiModalAgeRegressor(
        sa_ckpt=SA_CKPT, la_ckpt=LA_CKPT, t1_ckpt=T1_CKPT, ao_ckpt=AO_CKPT,
        modality_dropout_p=args.modality_dropout_p,
        use_deep_supervision=use_aux,
        use_lia=not args.disable_lia,
        lambda_lia=args.lambda_lia,
        lia_temperature=args.lia_temperature,
        sa_img_shape=sa_img_shape,
        la_img_shape=la_img_shape,
    ).to(device)
    # DDP must see all backbone parameters at construction time so that hooks are
    # available after true-frozen epochs are over.
    model.set_backbones_trainable(True)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    resume_state = None
    # Build the initial optimizer to match the saved checkpoint structure.
    # The loop below rebuilds it at the next epoch boundary if the stage changes.
    optimizer_epoch = 1
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        optimizer_epoch = int(resume_state["epoch"])
    initial_include_backbone = should_include_backbone(optimizer_epoch, args.frozen_epochs)
    set_backbones_trainable(model, initial_include_backbone)
    initial_backbone_lr = scheduled_backbone_lr(
        optimizer_epoch,
        args.backbone_lr_zero_epochs,
        args.lr_backbone,
    )
    optimizer = build_optimizer(model, args.lr_head, initial_backbone_lr,
                                args.weight_decay,
                                include_backbone=initial_include_backbone)

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    log_path = log_dir / "epoch_metrics.jsonl"
    resume_path = ckpt_dir / "resume_state.pth"

    if resume_state is not None:
        start_epoch, best_val_mae, best_epoch, epochs_without_improvement = load_resume_state(
            resume_state, model, optimizer, scaler,
        )
        if main_process:
            print(f"[Train] Resumed from epoch {start_epoch} with best_val_mae={best_val_mae:.4f} "
                  f"best_epoch={best_epoch} no_improve={epochs_without_improvement}")
    if distributed:
        dist.barrier()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        stage = scheduled_stage(epoch, args.frozen_epochs, args.backbone_lr_zero_epochs)
        include_backbone = should_include_backbone(epoch, args.frozen_epochs)
        backbone_lr = scheduled_backbone_lr(
            epoch,
            args.backbone_lr_zero_epochs,
            args.lr_backbone,
        )
        if optimizer_has_backbone(optimizer) != include_backbone:
            set_backbones_trainable(model, include_backbone)
            optimizer = build_optimizer(
                model,
                args.lr_head,
                backbone_lr,
                args.weight_decay,
                include_backbone=include_backbone,
            )
            if main_process:
                state = "unfrozen" if include_backbone else "frozen"
                print(f"[Train] Rebuilt optimizer for epoch {epoch}: backbone={state}")
        else:
            set_backbones_trainable(model, include_backbone)
            set_backbone_lr(optimizer, backbone_lr)
        lambda_aux = (
            args.lambda_aux_frozen
            if stage in ("true_frozen", "backbone_lr0_warmup")
            else args.lambda_aux_unfrozen
        )

        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, device,
                       target_mean, target_std, True, lambda_aux,
                       use_aux, amp_enabled, amp_dtype, scaler,
                       main_process)
        va = run_epoch(model, val_loader, None, device,
                       target_mean, target_std, False, lambda_aux,
                       use_aux, amp_enabled, amp_dtype, None,
                       main_process)

        stop_training = False
        stop_reason = ""
        if main_process:
            head_lr = get_group_lr(optimizer, "head")
            actual_backbone_lr = get_group_lr(optimizer, "backbone")
            print(f"Epoch {epoch:3d}/{args.epochs} [{stage}]  "
                  f"train MAE={tr['mae']:.3f} LIA={tr['loss_lia']:.4f}  "
                  f"val MAE={va['mae']:.3f} LIA={va['loss_lia']:.4f}  "
                  f"lr_head={format_lr(head_lr)} lr_backbone={format_lr(actual_backbone_lr)}  "
                  f"{time.time()-t0:.0f}s")

            row = {
                "epoch": epoch,
                "stage": stage,
                "backbone_trainable": include_backbone,
                "optimizer_includes_backbone": optimizer_has_backbone(optimizer),
                "lr_head": head_lr,
                "lr_backbone": actual_backbone_lr,
                "train": tr,
                "val": va,
            }
            append_jsonl(log_path, row)

            if not torch.isfinite(torch.tensor(va["mae"])):
                print("[Train] NaN detected in val MAE, stopping.")
                stop_training = True
                stop_reason = "nan_val_mae"

            improved = va["mae"] < (best_val_mae - args.early_stop_min_delta)
            if not stop_training and improved:
                best_val_mae = va["mae"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(unwrap_model(model).state_dict(), ckpt_dir / "best_model.pth")
                print(f"  ✓ Best saved (val MAE={best_val_mae:.4f})")
            elif not stop_training:
                epochs_without_improvement += 1
                if args.early_stop_patience > 0:
                    print(f"  no val MAE improvement for {epochs_without_improvement}/"
                          f"{args.early_stop_patience} epochs "
                          f"(best epoch={best_epoch}, best MAE={best_val_mae:.4f})")
                    if epochs_without_improvement >= args.early_stop_patience:
                        stop_training = True
                        stop_reason = "early_stop"
                        print(f"[Train] Early stopping at epoch {epoch}: "
                              f"best epoch={best_epoch}, best val MAE={best_val_mae:.4f}")

            if stop_reason != "nan_val_mae":
                save_resume_state(
                    resume_path,
                    epoch,
                    model,
                    optimizer,
                    scaler,
                    best_val_mae,
                    best_epoch,
                    epochs_without_improvement,
                    args,
                )

        if distributed:
            stop_tensor = torch.tensor([int(stop_training)], device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
        if stop_training:
            break

    if main_process:
        torch.save(unwrap_model(model).state_dict(), ckpt_dir / "last_model.pth")
        print(f"\n[Train] Done. Best val MAE={best_val_mae:.4f}  ->  {out_dir}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
