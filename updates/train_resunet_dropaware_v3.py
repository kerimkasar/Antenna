from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from train_resunet_dual import DualResUNet1D
from train_resunet_dual import compute_loss as compute_base_loss
from train_resunet_dual import effective_hilbert_weight, load_lhs_data, load_old_excel_data


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def split_train_val_test(
    n: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_test + n_val >= n:
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


def apply_gaussian_filter_channels(
    pred: torch.Tensor,
    sigma: float,
    kernel_size: int,
) -> torch.Tensor:
    if sigma <= 0:
        return pred
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1

    device = pred.device
    dtype = pred.dtype
    radius = (k - 1) / 2
    x = torch.linspace(-radius, radius, steps=k, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel_1d = kernel_1d / torch.sum(kernel_1d)
    kernel = kernel_1d.view(1, 1, k).repeat(pred.shape[1], 1, 1)

    padded = F.pad(pred, (k // 2, k // 2), mode="reflect")
    return F.conv1d(padded, kernel, groups=pred.shape[1])


def compute_global_rse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    real_p, imag_p = pred[:, 0, :], pred[:, 1, :]
    real_t, imag_t = target[:, 0, :], target[:, 1, :]
    mag_p = torch.sqrt(real_p**2 + imag_p**2 + 1e-12)
    mag_t = torch.sqrt(real_t**2 + imag_t**2 + 1e-12)
    return torch.mean(((mag_p - mag_t) ** 2) / (mag_t**2 + eps))


def compute_event_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    event_threshold_db: float,
    event_quantile: float,
    rebound_weight: float,
    event_gain: float,
    event_weight_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    real_p, imag_p = pred[:, 0, :], pred[:, 1, :]
    real_t, imag_t = target[:, 0, :], target[:, 1, :]

    mag_p = torch.sqrt(real_p**2 + imag_p**2 + 1e-12)
    mag_t = torch.sqrt(real_t**2 + imag_t**2 + 1e-12)

    db_p = 20.0 * torch.log10(torch.clamp(mag_p, min=1e-8))
    db_t = 20.0 * torch.log10(torch.clamp(mag_t, min=1e-8))

    slope_p = db_p[:, 1:] - db_p[:, :-1]
    slope_t = db_t[:, 1:] - db_t[:, :-1]

    # Sparse event gate: only sharp drop/rebound regions are emphasized.
    abs_slope_t = torch.abs(slope_t)
    adaptive_thr = torch.quantile(abs_slope_t.detach(), q=event_quantile, dim=1, keepdim=True)
    adaptive_thr = torch.maximum(adaptive_thr, torch.full_like(adaptive_thr, event_threshold_db))

    drop_excess = torch.relu(-slope_t - adaptive_thr)
    rebound_excess = torch.relu(slope_t - adaptive_thr)
    event_excess = drop_excess + rebound_weight * rebound_excess

    sparse_gate = (event_excess > 0).float()
    event_weight_slope = sparse_gate * (1.0 + event_gain * torch.tanh(event_excess))
    event_weight_slope = torch.clamp(event_weight_slope, max=event_weight_cap)

    event_slope_num = torch.sum(event_weight_slope * (slope_p - slope_t) ** 2)
    event_slope_den = torch.sum(event_weight_slope) + 1e-8
    event_slope_loss = event_slope_num / event_slope_den

    point_weight = torch.zeros_like(db_t)
    point_weight[:, :-1] += event_weight_slope
    point_weight[:, 1:] += event_weight_slope
    point_weight = point_weight / 2.0

    event_db_num = torch.sum(point_weight * (db_p - db_t) ** 2)
    event_db_den = torch.sum(point_weight) + 1e-8
    event_db_loss = event_db_num / event_db_den

    event_density = torch.mean(sparse_gate)
    return event_db_loss, event_slope_loss, event_density, torch.mean(adaptive_thr)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    w_hilbert_effective: float,
) -> tuple[float, float, float]:
    model.eval()
    losses = []
    event_db_losses = []
    event_slope_losses = []
    rse_losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            pred = apply_gaussian_filter_channels(pred, sigma=args.gauss_sigma, kernel_size=args.gauss_kernel_size)
            base = compute_base_loss(
                pred,
                yb,
                w_ri=args.w_ri,
                w_mag_db=args.w_mag_db,
                db_loss_mode=args.db_loss_mode,
                db_weight_alpha=args.db_weight_alpha,
                db_weight_max=args.db_weight_max,
                db_weight_eps=args.db_weight_eps,
                w_slope=args.w_slope,
                w_curv=args.w_curv,
                w_passivity=args.w_passivity,
                w_hilbert=w_hilbert_effective,
                hilbert_edge_frac=args.hilbert_edge_frac,
            )
            global_rse = compute_global_rse_loss(pred, yb, eps=args.rse_eps)
            event_db_loss, event_slope_loss, _, _ = compute_event_losses(
                pred,
                yb,
                event_threshold_db=args.event_threshold_db,
                event_quantile=args.event_quantile,
                rebound_weight=args.rebound_weight,
                event_gain=args.event_gain,
                event_weight_cap=args.event_weight_cap,
            )
            loss = (
                base
                + args.w_rse * global_rse
                + args.event_scale_eval * (args.w_event_db * event_db_loss + args.w_event_slope * event_slope_loss)
            )
            losses.append(loss.item())
            event_db_losses.append(event_db_loss.item())
            event_slope_losses.append(event_slope_loss.item())
            rse_losses.append(global_rse.item())

    return float(np.mean(losses)), float(np.mean(event_db_losses)), float(np.mean(event_slope_losses)), float(np.mean(rse_losses))


def train(args: argparse.Namespace) -> None:
    root = Path(args.project_root).resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "old_excel":
        x_raw, y = load_old_excel_data(root)
    else:
        x_raw, y = load_lhs_data(root)

    if args.init_scaler:
        scaler = joblib.load((root / args.init_scaler).resolve())
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(x_raw)
    x_scaled = scaler.transform(x_raw).astype(np.float32)

    tr_idx, va_idx, te_idx = split_train_val_test(
        len(x_scaled), val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    x_train = torch.tensor(x_scaled[tr_idx], dtype=torch.float32)
    y_train = torch.tensor(y[tr_idx], dtype=torch.float32)
    x_val = torch.tensor(x_scaled[va_idx], dtype=torch.float32)
    y_val = torch.tensor(y[va_idx], dtype=torch.float32)
    x_test = torch.tensor(x_scaled[te_idx], dtype=torch.float32)
    y_test = torch.tensor(y[te_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = DualResUNet1D(input_dim=x_train.shape[1], target_len=y_train.shape[-1], base_ch=args.base_channels).to(device)
    if args.init_model:
        init_model_path = (root / args.init_model).resolve()
        model.load_state_dict(torch.load(init_model_path, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_event_db = []
        train_event_slope = []
        train_event_density = []
        train_event_thr = []
        train_rse = []

        w_h_epoch = effective_hilbert_weight(epoch, args)
        event_scale = 1.0 if args.event_warmup_epochs <= 0 else min(1.0, float(epoch) / float(args.event_warmup_epochs))
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            pred = apply_gaussian_filter_channels(pred, sigma=args.gauss_sigma, kernel_size=args.gauss_kernel_size)
            base = compute_base_loss(
                pred,
                yb,
                w_ri=args.w_ri,
                w_mag_db=args.w_mag_db,
                db_loss_mode=args.db_loss_mode,
                db_weight_alpha=args.db_weight_alpha,
                db_weight_max=args.db_weight_max,
                db_weight_eps=args.db_weight_eps,
                w_slope=args.w_slope,
                w_curv=args.w_curv,
                w_passivity=args.w_passivity,
                w_hilbert=w_h_epoch,
                hilbert_edge_frac=args.hilbert_edge_frac,
            )
            global_rse = compute_global_rse_loss(pred, yb, eps=args.rse_eps)
            event_db_loss, event_slope_loss, event_density, event_thr = compute_event_losses(
                pred,
                yb,
                event_threshold_db=args.event_threshold_db,
                event_quantile=args.event_quantile,
                rebound_weight=args.rebound_weight,
                event_gain=args.event_gain,
                event_weight_cap=args.event_weight_cap,
            )
            loss = base + args.w_rse * global_rse + event_scale * (args.w_event_db * event_db_loss + args.w_event_slope * event_slope_loss)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_event_db.append(event_db_loss.item())
            train_event_slope.append(event_slope_loss.item())
            train_event_density.append(event_density.item())
            train_event_thr.append(event_thr.item())
            train_rse.append(global_rse.item())

        scheduler.step()
        train_loss = float(np.mean(train_losses))
        val_loss, val_event_db, val_event_slope, val_rse = evaluate(model, val_loader, device, args, w_h_epoch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rse_loss": float(np.mean(train_rse)),
                "val_rse_loss": val_rse,
                "train_event_db_loss": float(np.mean(train_event_db)),
                "train_event_slope_loss": float(np.mean(train_event_slope)),
                "train_event_density": float(np.mean(train_event_density)),
                "train_event_threshold_db": float(np.mean(train_event_thr)),
                "val_event_db_loss": val_event_db,
                "val_event_slope_loss": val_event_slope,
                "w_hilbert": w_h_epoch,
                "event_scale": float(event_scale),
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"epoch={epoch:4d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"train_rse={float(np.mean(train_rse)):.6f} val_rse={val_rse:.6f} "
                f"train_event_db={float(np.mean(train_event_db)):.6f} val_event_db={val_event_db:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_event_db, test_event_slope, test_rse = evaluate(
        model,
        test_loader,
        device,
        args,
        w_hilbert_effective=effective_hilbert_weight(args.epochs, args),
    )

    model_path = out_dir / args.model_name
    scaler_path = out_dir / args.scaler_name
    history_path = out_dir / args.history_name
    meta_path = out_dir / args.meta_name

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    pd.DataFrame(history).to_csv(history_path, index=False)

    meta = {
        "model": "DualResUNet1D_DropAware",
        "dataset": args.dataset,
        "base_channels": args.base_channels,
        "initialized_from": args.init_model,
        "weights": {
            "w_ri": args.w_ri,
            "w_mag_db": args.w_mag_db,
            "w_slope": args.w_slope,
            "w_curv": args.w_curv,
            "w_passivity": args.w_passivity,
            "w_hilbert": args.w_hilbert,
            "w_rse": args.w_rse,
            "w_event_db": args.w_event_db,
            "w_event_slope": args.w_event_slope,
        },
        "event": {
            "event_threshold_db": args.event_threshold_db,
            "event_quantile": args.event_quantile,
            "rebound_weight": args.rebound_weight,
            "event_gain": args.event_gain,
            "event_weight_cap": args.event_weight_cap,
            "event_warmup_epochs": args.event_warmup_epochs,
        },
        "gaussian_filter": {
            "sigma": args.gauss_sigma,
            "kernel_size": args.gauss_kernel_size,
        },
        "db_loss_mode": args.db_loss_mode,
        "db_weight_alpha": args.db_weight_alpha,
        "db_weight_max": args.db_weight_max,
        "db_weight_eps": args.db_weight_eps,
        "hilbert_edge_frac": args.hilbert_edge_frac,
        "hilbert_warmup_epochs": args.hilbert_warmup_epochs,
        "n_samples": int(len(x_scaled)),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "best_val_loss": float(best_val),
        "test_loss": float(test_loss),
        "test_rse_loss": float(test_rse),
        "test_event_db_loss": float(test_event_db),
        "test_event_slope_loss": float(test_event_slope),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Test loss: {test_loss:.6f} | Test RSE loss: {test_rse:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune ResUNet with RSE-first + sparse event penalty (v3)"
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="old_excel", choices=["old_excel", "lhs"])
    parser.add_argument("--output-dir", type=str, default="NNModel")
    parser.add_argument("--model-name", type=str, default="trained_model_resunet_dropaware_v3rse.pt")
    parser.add_argument("--scaler-name", type=str, default="scaler_resunet_dropaware_v3rse.gz")
    parser.add_argument("--history-name", type=str, default="history_resunet_dropaware_v3rse.csv")
    parser.add_argument("--meta-name", type=str, default="meta_resunet_dropaware_v3rse.json")

    parser.add_argument("--init-model", type=str, default="NNModel/trained_model_resunet_dual.pt")
    parser.add_argument("--init-scaler", type=str, default="NNModel/scaler_resunet_dual.gz")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)

    parser.add_argument("--w-ri", type=float, default=0.20)
    parser.add_argument("--w-mag-db", type=float, default=1.00)
    parser.add_argument("--db-loss-mode", type=str, default="rse", choices=["rse", "weighted"])
    parser.add_argument("--db-weight-alpha", type=float, default=1.0)
    parser.add_argument("--db-weight-max", type=float, default=10.0)
    parser.add_argument("--db-weight-eps", type=float, default=1e-4)
    parser.add_argument("--w-slope", type=float, default=0.10)
    parser.add_argument("--w-curv", type=float, default=0.05)
    parser.add_argument("--w-passivity", type=float, default=0.05)
    parser.add_argument("--w-hilbert", type=float, default=0.02)
    parser.add_argument("--hilbert-edge-frac", type=float, default=0.12)
    parser.add_argument("--hilbert-warmup-epochs", type=int, default=60)

    parser.add_argument("--w-rse", type=float, default=0.35)
    parser.add_argument("--rse-eps", type=float, default=1e-6)
    parser.add_argument("--w-event-db", type=float, default=0.02)
    parser.add_argument("--w-event-slope", type=float, default=0.015)
    parser.add_argument("--event-threshold-db", type=float, default=0.70)
    parser.add_argument("--event-quantile", type=float, default=0.93)
    parser.add_argument("--rebound-weight", type=float, default=0.60)
    parser.add_argument("--event-gain", type=float, default=0.80)
    parser.add_argument("--event-weight-cap", type=float, default=2.0)
    parser.add_argument("--event-warmup-epochs", type=int, default=20)
    parser.add_argument("--gauss-sigma", type=float, default=1.1)
    parser.add_argument("--gauss-kernel-size", type=int, default=11)

    parser.add_argument("--event-scale-eval", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    args = parse_args()
    train(args)
