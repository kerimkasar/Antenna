from __future__ import annotations

import argparse
import numpy as np
import torch

from train_resunet_dual import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResUNet dual with tuned defaults (separate artifacts)"
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="old_excel", choices=["old_excel", "lhs"])
    parser.add_argument("--output-dir", type=str, default="NNModel")

    parser.add_argument("--model-name", type=str, default="trained_model_resunet_dual_tuned.pt")
    parser.add_argument("--scaler-name", type=str, default="scaler_resunet_dual_tuned.gz")
    parser.add_argument("--history-name", type=str, default="history_resunet_dual_tuned.csv")
    parser.add_argument("--meta-name", type=str, default="meta_resunet_dual_tuned.json")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0004302517396016556)
    parser.add_argument("--weight-decay", type=float, default=0.0007114476009343421)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)

    parser.add_argument("--w-ri", type=float, default=0.13899863008405067)
    parser.add_argument("--w-mag-db", type=float, default=0.7638919733850194)
    parser.add_argument("--db-loss-mode", type=str, default="rse", choices=["rse", "weighted"])
    parser.add_argument("--db-weight-alpha", type=float, default=0.7727374508106509)
    parser.add_argument("--db-weight-max", type=float, default=6.934472157654941)
    parser.add_argument("--db-weight-eps", type=float, default=4.5897367362261086e-05)
    parser.add_argument("--w-slope", type=float, default=0.2192205135282351)
    parser.add_argument("--w-curv", type=float, default=0.0801393764679011)
    parser.add_argument("--w-passivity", type=float, default=0.08642834644654523)
    parser.add_argument("--w-hilbert", type=float, default=0.0016467595436641957)
    parser.add_argument("--hilbert-edge-frac", type=float, default=0.19548647782429918)
    parser.add_argument("--hilbert-warmup-epochs", type=int, default=104)

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
