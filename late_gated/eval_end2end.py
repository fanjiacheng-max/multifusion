"""
用法:
  python -m multi_fusion.cmr_irene_v7.late_gated.eval_end2end \
      --ckpt outputs/cmr_irene_v7_end2end/checkpoints/best_model.pth \
      --split test
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multi_fusion.cmr_irene_v7.late_gated.dataset import MultiModalRawDataset, MODALITIES
from multi_fusion.cmr_irene_v7.late_gated.model_end2end import EndToEndMultiModalAgeRegressor
from multimode_train.train_four_modal_mult import compute_metrics

BASE        = Path("/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp")
OUTCOME_CSV = str(BASE / "UKB_processed/age_at_scan_wide.csv")

SA_CKPT = str(BASE / "outputs/clean/sax/age_at_scan/checkpoints/best_model.pth")
LA_CKPT = str(BASE / "outputs/clean/lax4ch/age_at_scan/checkpoints/best_model.pth")
T1_CKPT = str(BASE / "outputs/clean_dinov2/shmolli/checkpoints/best_model.pth")
AO_CKPT = str(BASE / "outputs/clean_dinov2/aortic/checkpoints/best_model.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--split",      default="test")
    parser.add_argument("--out_dir",    default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--ao_random_crop", dest="ao_random_crop", action="store_true")
    parser.add_argument("--no_ao_random_crop", "--no-ao_random_crop",
                        dest="ao_random_crop", action="store_false")
    parser.add_argument("--sa_target_z", type=int, default=None)
    parser.add_argument("--sa_target_t", type=int, default=None)
    parser.add_argument("--la_target_z", type=int, default=None)
    parser.add_argument("--la_target_t", type=int, default=None)
    parser.set_defaults(ao_random_crop=None)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir   = Path(args.out_dir) if args.out_dir else ckpt_path.parents[1] / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    startup = json.load(open(ckpt_path.parents[1] / "logs" / "startup_metrics.json"))
    target_mean, target_std = startup["target_mean"], startup["target_std"]
    ao_random_crop = startup.get("ao_random_crop", True) if args.ao_random_crop is None else args.ao_random_crop
    sa_target_z = startup.get("sa_target_z", 6) if args.sa_target_z is None else args.sa_target_z
    sa_target_t = startup.get("sa_target_t", 50) if args.sa_target_t is None else args.sa_target_t
    la_target_z = startup.get("la_target_z", 1) if args.la_target_z is None else args.la_target_z
    la_target_t = startup.get("la_target_t", 50) if args.la_target_t is None else args.la_target_t
    use_aux = startup.get("use_deep_supervision", True)
    use_lia = startup.get("use_lia", True)
    lambda_lia = startup.get("lambda_lia", 0.1)
    lia_temperature = startup.get("lia_temperature", 0.1)
    sa_img_shape = (sa_target_z, sa_target_t, 128, 128)
    la_img_shape = (la_target_z, la_target_t, 128, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MultiModalRawDataset(
        OUTCOME_CSV, split=args.split, augment=False,
        ao_random_crop=ao_random_crop,
        sa_target_z=sa_target_z,
        sa_target_t=sa_target_t,
        la_target_z=la_target_z,
        la_target_t=la_target_t,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = EndToEndMultiModalAgeRegressor(
        sa_ckpt=SA_CKPT, la_ckpt=LA_CKPT, t1_ckpt=T1_CKPT, ao_ckpt=AO_CKPT,
        modality_dropout_p=0.0, use_deep_supervision=use_aux,
        use_lia=use_lia,
        lambda_lia=lambda_lia,
        lia_temperature=lia_temperature,
        sa_img_shape=sa_img_shape,
        la_img_shape=la_img_shape,
    ).to(device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_t, all_p, all_eids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            inputs = {m: batch[m].to(device) for m in MODALITIES}
            out = model(inputs)
            pred = out["age_pred"].squeeze(-1).float().cpu() * target_std + target_mean
            all_t.extend(batch["age"].tolist())
            all_p.extend(pred.tolist())
            all_eids.extend(batch["eid"])

    metrics = compute_metrics(all_t, all_p)
    metrics["ao_random_crop"] = ao_random_crop
    metrics["sa_target_z"] = sa_target_z
    metrics["sa_target_t"] = sa_target_t
    metrics["la_target_z"] = la_target_z
    metrics["la_target_t"] = la_target_t
    metrics["use_deep_supervision"] = use_aux
    metrics["use_lia"] = use_lia
    metrics["lambda_lia"] = lambda_lia
    metrics["lia_temperature"] = lia_temperature
    print(f"\n{'='*55}")
    print(f"  {args.split.upper()}  N={metrics['n']}  MAE={metrics['mae']:.4f}  "
          f"R²={metrics['r2']:.4f}")
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
