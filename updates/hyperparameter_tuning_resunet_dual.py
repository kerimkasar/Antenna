from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "optuna is required for tree-based tuning. Install with: pip install optuna"
    ) from exc

from train_resunet_dual import (
    DualResUNet1D,
    compute_loss,
    effective_hilbert_weight,
    load_lhs_data,
    load_old_excel_data,
)


def split_train_val(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    return idx[n_val:], idx[:n_val]


def overall_rse_mag_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    real_p, imag_p = pred[:, 0, :], pred[:, 1, :]
    real_t, imag_t = target[:, 0, :], target[:, 1, :]
    mag_p = torch.sqrt(real_p**2 + imag_p**2 + 1e-12)
    mag_t = torch.sqrt(real_t**2 + imag_t**2 + 1e-12)
    db_p = 20.0 * torch.log10(torch.clamp(mag_p, min=1e-8))
    db_t = 20.0 * torch.log10(torch.clamp(mag_t, min=1e-8))
    return float(torch.mean(((db_p - db_t) ** 2) / (db_t**2 + eps)).detach().cpu().item())


def predict_all(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out.append(model(xb).detach().cpu())
    return torch.cat(out, dim=0)


def default_params() -> dict[str, Any]:
    return {
        "base_channels": 64,
        "batch_size": 16,
        "lr": 0.0004302517396016556,
        "weight_decay": 0.0007114476009343421,
        "w_ri": 0.13899863008405067,
        "w_mag_db": 0.7638919733850194,
        "db_loss_mode": "rse",
        "db_weight_alpha": 0.7727374508106509,
        "db_weight_max": 6.934472157654941,
        "db_weight_eps": 4.5897367362261086e-05,
        "w_slope": 0.2192205135282351,
        "w_curv": 0.0801393764679011,
        "w_passivity": 0.08642834644654523,
        "w_hilbert": 0.0016467595436641957,
        "hilbert_edge_frac": 0.19548647782429918,
        "hilbert_warmup_epochs": 104,
    }


def suggest_params(trial: optuna.trial.Trial, defaults: dict[str, Any]) -> dict[str, Any]:
    p = dict(defaults)
    p["db_loss_mode"] = "rse"  # fixed as requested
    p["lr"] = trial.suggest_float("lr", 1.5e-4, 2.5e-3, log=True)
    p["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    p["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
    p["w_ri"] = trial.suggest_float("w_ri", 0.10, 0.35)
    p["w_mag_db"] = trial.suggest_float("w_mag_db", 0.70, 1.80)
    p["w_slope"] = trial.suggest_float("w_slope", 0.02, 0.25)
    p["w_curv"] = trial.suggest_float("w_curv", 0.005, 0.13)
    p["w_passivity"] = trial.suggest_float("w_passivity", 0.005, 0.12)
    p["w_hilbert"] = trial.suggest_float("w_hilbert", 0.0, 0.08)
    p["hilbert_edge_frac"] = trial.suggest_float("hilbert_edge_frac", 0.05, 0.20)
    p["hilbert_warmup_epochs"] = trial.suggest_int("hilbert_warmup_epochs", 20, 120)
    p["db_weight_eps"] = trial.suggest_float("db_weight_eps", 2e-5, 1e-3, log=True)
    p["db_weight_alpha"] = trial.suggest_float("db_weight_alpha", 0.5, 2.0)
    p["db_weight_max"] = trial.suggest_float("db_weight_max", 4.0, 20.0)
    return p


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def train_and_eval(
    trial_name: str,
    params: dict[str, Any],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    epochs: int,
    early_stop_patience: int,
    use_amp: bool,
    num_workers: int,
    pin_memory: bool,
    trial: optuna.trial.Trial | None = None,
    save_state: bool = False,
) -> dict[str, Any]:
    train_loader = make_loader(
        x_train, y_train, int(params["batch_size"]), True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = make_loader(
        x_val, y_val, int(params["batch_size"]), False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = make_loader(
        x_test, y_test, max(64, int(params["batch_size"])), False, num_workers=num_workers, pin_memory=pin_memory
    )

    model = DualResUNet1D(
        input_dim=x_train.shape[1],
        target_len=y_train.shape[-1],
        base_ch=int(params["base_channels"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None if save_state else None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        w_h = effective_hilbert_weight(epoch, argparse.Namespace(**params))
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(xb)
                loss = compute_loss(
                    pred,
                    yb,
                    w_ri=params["w_ri"],
                    w_mag_db=params["w_mag_db"],
                    db_loss_mode="rse",
                    db_weight_alpha=params["db_weight_alpha"],
                    db_weight_max=params["db_weight_max"],
                    db_weight_eps=params["db_weight_eps"],
                    w_slope=params["w_slope"],
                    w_curv=params["w_curv"],
                    w_passivity=params["w_passivity"],
                    w_hilbert=w_h,
                    hilbert_edge_frac=params["hilbert_edge_frac"],
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                pred = model(xb)
                vloss = compute_loss(
                    pred,
                    yb,
                    w_ri=params["w_ri"],
                    w_mag_db=params["w_mag_db"],
                    db_loss_mode="rse",
                    db_weight_alpha=params["db_weight_alpha"],
                    db_weight_max=params["db_weight_max"],
                    db_weight_eps=params["db_weight_eps"],
                    w_slope=params["w_slope"],
                    w_curv=params["w_curv"],
                    w_passivity=params["w_passivity"],
                    w_hilbert=w_h,
                    hilbert_edge_frac=params["hilbert_edge_frac"],
                )
                val_losses.append(float(vloss.item()))
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            if save_state:
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"[{trial_name}] epoch={epoch:3d}/{epochs} "
                f"train={float(np.mean(train_losses)):.6f} val={val_loss:.6f} best={best_val:.6f}"
            )

        if trial is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if bad_epochs >= early_stop_patience:
            print(f"[{trial_name}] early stop at epoch {epoch}")
            break

    if save_state and best_state is not None:
        model.load_state_dict(best_state)

    pred_test = predict_all(model, test_loader, device)
    rse_db = overall_rse_mag_db(pred_test, y_test)
    return {
        "lhs_overall_rse_db": float(rse_db),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "params": dict(params),
        "state_dict": best_state,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tree-based (TPE) hyperparameter tuning for ResUNet dual; train=old_excel, test=LHS overall RSE"
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="NNModel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-rse-db", type=float, default=-1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--study-name", type=str, default="resunet_dual_tpe")

    parser.add_argument("--best-model-name", type=str, default="trained_model_resunet_dual_tuned.pt")
    parser.add_argument("--best-scaler-name", type=str, default="scaler_resunet_dual_tuned.gz")
    parser.add_argument("--result-json-name", type=str, default="resunet_dual_tuning_result.json")
    parser.add_argument("--trials-csv-name", type=str, default="resunet_dual_tuning_trials.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.project_root).resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    pin_memory = bool(device.type == "cuda")
    print(f"Device: {device} | AMP: {use_amp}")
    if device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    x_old, y_old = load_old_excel_data(root)
    x_lhs, y_lhs = load_lhs_data(root)
    tr_idx, va_idx = split_train_val(len(x_old), args.val_ratio, args.seed)

    x_old_train = x_old[tr_idx]
    x_old_val = x_old[va_idx]
    y_old_train = y_old[tr_idx]
    y_old_val = y_old[va_idx]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train_scaled = scaler.fit_transform(x_old_train).astype(np.float32)
    x_val_scaled = scaler.transform(x_old_val).astype(np.float32)
    x_lhs_scaled = scaler.transform(x_lhs).astype(np.float32)

    x_train = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_old_train, dtype=torch.float32)
    x_val = torch.tensor(x_val_scaled, dtype=torch.float32)
    y_val = torch.tensor(y_old_val, dtype=torch.float32)
    x_test = torch.tensor(x_lhs_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_lhs, dtype=torch.float32)

    defaults = default_params()
    print("\n[BASELINE] Running default parameters...")
    baseline = train_and_eval(
        trial_name="default",
        params=defaults,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        device=device,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        use_amp=use_amp,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        trial=None,
        save_state=False,
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=30, interval_steps=10)
    storage = args.storage.strip() or f"sqlite:///{(out_dir / 'resunet_dual_tpe.db').as_posix()}"
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, defaults)
        out = train_and_eval(
            trial_name=f"trial_{trial.number:03d}",
            params=params,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            device=device,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
            use_amp=use_amp,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            trial=trial,
            save_state=False,
        )
        trial.set_user_attr("lhs_overall_rse_db", out["lhs_overall_rse_db"])
        trial.set_user_attr("best_val_loss", out["best_val_loss"])
        trial.set_user_attr("best_epoch", out["best_epoch"])
        print(
            f"[TUNING] trial_{trial.number:03d} RSE(db)={out['lhs_overall_rse_db']:.6f} "
            f"| default={baseline['lhs_overall_rse_db']:.6f}"
        )
        return out["lhs_overall_rse_db"]

    stop_state = {"reached": False}

    def stop_on_target(study_obj: optuna.Study, trial_obj: optuna.trial.FrozenTrial) -> None:
        if args.target_rse_db > 0 and study_obj.best_value <= args.target_rse_db:
            stop_state["reached"] = True
            study_obj.stop()

    print(
        f"\n[TUNING] Tree-based search: trials={args.max_trials}, epochs={args.epochs}, "
        f"db_loss_mode=rse, n_jobs={args.n_jobs}, num_workers={args.num_workers}"
    )
    study.optimize(objective, n_trials=args.max_trials, callbacks=[stop_on_target], n_jobs=args.n_jobs)

    if study.best_trial is None:
        raise RuntimeError("No completed Optuna trial found.")

    best_params = dict(defaults)
    best_params.update(study.best_trial.params)
    best_params["db_loss_mode"] = "rse"
    print("\n[FINAL] Re-training best trial once to save best state_dict...")
    best_bundle = train_and_eval(
        trial_name="best_retrain",
        params=best_params,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        device=device,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        use_amp=use_amp,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        trial=None,
        save_state=True,
    )

    best_model_path = out_dir / args.best_model_name
    best_scaler_path = out_dir / args.best_scaler_name
    result_json_path = out_dir / args.result_json_name
    trials_csv_path = out_dir / args.trials_csv_name

    if best_bundle["state_dict"] is not None:
        torch.save(best_bundle["state_dict"], best_model_path)
    joblib.dump(scaler, best_scaler_path)

    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "trial_name": "default",
            "lhs_overall_rse_db": baseline["lhs_overall_rse_db"],
            "best_val_loss": baseline["best_val_loss"],
            "best_epoch": baseline["best_epoch"],
            **{f"hp_{k}": v for k, v in baseline["params"].items()},
        }
    )
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append(
            {
                "trial_name": f"trial_{t.number:03d}",
                "lhs_overall_rse_db": t.user_attrs.get("lhs_overall_rse_db", float("nan")),
                "best_val_loss": t.user_attrs.get("best_val_loss", float("nan")),
                "best_epoch": t.user_attrs.get("best_epoch", -1),
                **{f"hp_{k}": v for k, v in t.params.items()},
                "hp_db_loss_mode": "rse",
            }
        )
    pd.DataFrame(rows).sort_values("lhs_overall_rse_db", ascending=True).to_csv(trials_csv_path, index=False)

    improvement = baseline["lhs_overall_rse_db"] - best_bundle["lhs_overall_rse_db"]
    improvement_pct = (improvement / baseline["lhs_overall_rse_db"] * 100.0) if baseline["lhs_overall_rse_db"] > 0 else 0.0

    summary = {
        "search_type": "tree_based_tpe",
        "train_dataset": "old_excel",
        "test_dataset": "lhs_all_seeds",
        "db_loss_mode": "rse",
        "device": str(device),
        "amp": use_amp,
        "epochs": int(args.epochs),
        "max_trials": int(args.max_trials),
        "n_jobs": int(args.n_jobs),
        "num_workers": int(args.num_workers),
        "optuna_storage": storage,
        "study_name": args.study_name,
        "target_rse_db": float(args.target_rse_db),
        "stopped_on_target": bool(stop_state["reached"]),
        "baseline": {
            "lhs_overall_rse_db": baseline["lhs_overall_rse_db"],
            "best_val_loss": baseline["best_val_loss"],
            "best_epoch": baseline["best_epoch"],
            "params": baseline["params"],
        },
        "best": {
            "lhs_overall_rse_db": best_bundle["lhs_overall_rse_db"],
            "best_val_loss": best_bundle["best_val_loss"],
            "best_epoch": best_bundle["best_epoch"],
            "params": best_bundle["params"],
        },
        "improvement_db_rse_abs": float(improvement),
        "improvement_db_rse_pct": float(improvement_pct),
        "n_optuna_trials_total": int(len(study.trials)),
        "n_optuna_trials_complete": int(len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])),
        "saved_best_model": str(best_model_path),
        "saved_best_scaler": str(best_scaler_path),
        "trials_csv": str(trials_csv_path),
    }
    result_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== TUNING SUMMARY =====")
    print(f"Search type: tree-based TPE")
    print(f"db_loss_mode: rse")
    print(f"Default LHS overall RSE (dB): {baseline['lhs_overall_rse_db']:.6f}")
    print(f"Best    LHS overall RSE (dB): {best_bundle['lhs_overall_rse_db']:.6f}")
    print(f"Improvement (abs): {improvement:.6f}")
    print(f"Improvement (%):   {improvement_pct:.2f}%")
    print(f"Saved summary: {result_json_path}")


if __name__ == "__main__":
    main()
