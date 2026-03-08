from __future__ import annotations

from pathlib import Path
import warnings
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.exceptions import InconsistentVersionWarning

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from compare_antenna_vs_tcnn_sdd11 import AntennaNeuralNet
from train_resunet_dual import INPUT_COLUMNS, DualResUNet1D


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


GAUSSIAN_WINDOW_LENGTH = 51


def to_mag(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return np.sqrt(real**2 + imag**2)


def to_db(mag: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.clip(mag, 1e-12, None))


def rse(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(((y_pred - y_true) ** 2) / (y_true**2 + eps)))


def gaussian_kernel(window_length: int, sigma: float) -> np.ndarray:
    window = max(3, int(window_length))
    if window % 2 == 0:
        window += 1
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    sigma_safe = max(0.1, float(sigma))
    kernel = np.exp(-0.5 * (x / sigma_safe) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def gaussian_smooth_1d(y: np.ndarray, sigma: float, window_length: int = GAUSSIAN_WINDOW_LENGTH) -> np.ndarray:
    if sigma <= 0:
        return y
    kernel = gaussian_kernel(window_length, sigma)
    pad = len(kernel) // 2
    y_pad = np.pad(y, (pad, pad), mode="reflect")
    return np.convolve(y_pad, kernel, mode="valid")


def gaussian_smooth_2d(y: np.ndarray, sigma: float, window_length: int = GAUSSIAN_WINDOW_LENGTH) -> np.ndarray:
    if sigma <= 0:
        return y
    kernel = gaussian_kernel(window_length, sigma)
    pad = len(kernel) // 2
    y_pad = np.pad(y, ((0, 0), (pad, pad)), mode="reflect")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=y_pad)


@st.cache_data
def list_lhs_files(project_root: str) -> list[Path]:
    lhs = Path(project_root) / "data" / "LHS"
    return sorted(lhs.glob("input_trials_done_LHS_n20_rounded_seed*.csv"))


@st.cache_data
def load_lhs_seed(input_file: str):
    input_path = Path(input_file)
    seed = input_path.stem.split("seed")[-1]
    lhs = input_path.parent
    real_path = lhs / f"real_initial_LHS_n20_rounded_seed{seed}.csv"
    imag_path = lhs / f"imag_initial_LHS_n20_rounded_seed{seed}.csv"
    x_df = pd.read_csv(input_path)[INPUT_COLUMNS]
    real = pd.read_csv(real_path).values.astype(np.float32)
    imag = pd.read_csv(imag_path).values.astype(np.float32)
    return x_df, real, imag, seed


@st.cache_data
def load_old_excel(project_root: str):
    old = Path(project_root) / "old" / "data"
    x_df = pd.read_excel(old / "input_parameters.xlsx")[INPUT_COLUMNS]
    real = pd.read_excel(old / "reel.xlsx").values.astype(np.float32)
    imag = pd.read_excel(old / "imaginary.xlsx").values.astype(np.float32)
    n = min(len(x_df), len(real), len(imag))
    return x_df.iloc[:n].copy(), real[:n], imag[:n]


@st.cache_resource
def load_antenna(project_root: str, model_path: str, scaler_path: str):
    root = Path(project_root)
    scaler = joblib.load(root / scaler_path)
    model = AntennaNeuralNet()
    model.load_state_dict(torch.load(root / model_path, map_location="cpu"))
    model.eval()
    return model, scaler


@st.cache_resource
def load_resunet(project_root: str, model_path: str, scaler_path: str, target_len: int):
    root = Path(project_root)
    scaler = joblib.load(root / scaler_path)
    model = DualResUNet1D(input_dim=len(INPUT_COLUMNS), target_len=target_len)
    model.load_state_dict(torch.load(root / model_path, map_location="cpu"))
    model.eval()
    return model, scaler


def run_model_batch(model: torch.nn.Module, x_scaled: np.ndarray, batch_size: int = 512) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_scaled), batch_size):
            xb = torch.tensor(x_scaled[i : i + batch_size], dtype=torch.float32)
            preds.append(model(xb).numpy())
    return np.concatenate(preds, axis=0)


@st.cache_data
def compute_dataset_metrics(
    project_root: str,
    dataset: str,
    lhs_selected_name: str,
    magnitude_db: bool,
    apply_gaussian: bool,
    gauss_sigma: float,
    gauss_window: int,
    antenna_model_path: str,
    antenna_scaler_path: str,
    base_resunet_model_path: str,
    base_resunet_scaler_path: str,
    dropaware_v3_model_path: str,
    dropaware_v3_scaler_path: str,
) -> pd.DataFrame:
    root = Path(project_root)

    if dataset == "lhs":
        lhs_file = root / "data" / "LHS" / lhs_selected_name
        x_df, real_all, imag_all, _ = load_lhs_seed(str(lhs_file))
    else:
        x_df, real_all, imag_all = load_old_excel(project_root)

    target_len = real_all.shape[1]
    x_all = x_df.values.astype(np.float32)

    antenna_model, antenna_scaler = load_antenna(project_root, antenna_model_path, antenna_scaler_path)
    base_model, base_scaler = load_resunet(project_root, base_resunet_model_path, base_resunet_scaler_path, target_len)
    dropaware_model, dropaware_scaler = load_resunet(
        project_root, dropaware_v3_model_path, dropaware_v3_scaler_path, target_len
    )

    x_ant = antenna_scaler.transform(x_all).astype(np.float32)
    x_base = base_scaler.transform(x_all).astype(np.float32)
    x_drop = dropaware_scaler.transform(x_all).astype(np.float32)

    pred_ant = run_model_batch(antenna_model, x_ant)
    pred_base = run_model_batch(base_model, x_base)
    pred_drop = run_model_batch(dropaware_model, x_drop)

    mag_true = to_mag(real_all, imag_all)
    mag_ant = to_mag(pred_ant[:, 0, :], pred_ant[:, 1, :])
    mag_base = to_mag(pred_base[:, 0, :], pred_base[:, 1, :])
    mag_drop = to_mag(pred_drop[:, 0, :], pred_drop[:, 1, :])

    if apply_gaussian:
        mag_base = gaussian_smooth_2d(mag_base, gauss_sigma, gauss_window)
        mag_drop = gaussian_smooth_2d(mag_drop, gauss_sigma, gauss_window)

    if magnitude_db:
        y_true = to_db(mag_true)
        y_ant = to_db(mag_ant)
        y_base = to_db(mag_base)
        y_drop = to_db(mag_drop)
    else:
        y_true = mag_true
        y_ant = mag_ant
        y_base = mag_base
        y_drop = mag_drop

    rows = [
        {"Model": "Antenna NN", "MSE": float(np.mean((y_ant - y_true) ** 2)), "RSE": rse(y_ant, y_true)},
        {"Model": "Base ResUNet", "MSE": float(np.mean((y_base - y_true) ** 2)), "RSE": rse(y_base, y_true)},
        {"Model": "DropAware v3", "MSE": float(np.mean((y_drop - y_true) ** 2)), "RSE": rse(y_drop, y_true)},
    ]
    return pd.DataFrame(rows).sort_values("RSE", ascending=True).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="ResUNet Dual + DropAware v3 Comparator", layout="wide")
    st.title("ResUNet Dual Web App + DropAware v3")

    project_root = Path(__file__).resolve().parents[1]

    with st.sidebar:
        st.header("Controls")
        dataset = st.selectbox("Dataset", ["lhs", "old_excel"], index=0)
        trace_label = st.selectbox("Trace label", ["S11", "Sdd11"], index=0)
        magnitude_db = st.checkbox("Show magnitude in dB", value=True)

        st.subheader("Gaussian Filter (Base ResUNet + DropAware v3)")
        apply_gaussian = st.checkbox("Apply Gaussian filter", value=False)
        gauss_sigma = st.slider("Gaussian sigma", 0.1, 3.0, 1.1, 0.1, disabled=not apply_gaussian)
        gauss_window = GAUSSIAN_WINDOW_LENGTH
        st.caption(f"Gaussian window length: {gauss_window} (fixed)")

        antenna_model_path = st.text_input("Antenna model", value="NNModel/trained_model.pt")
        antenna_scaler_path = st.text_input("Antenna scaler", value="NNModel/scaler.gz")

        base_resunet_model_path = st.text_input("Base ResUNet model", value="NNModel/trained_model_resunet_dual.pt")
        base_resunet_scaler_path = st.text_input("Base ResUNet scaler", value="NNModel/scaler_resunet_dual.gz")

        dropaware_v3_model_path = st.text_input(
            "DropAware v3 model", value="NNModel/trained_model_resunet_dropaware_v3rse.pt"
        )
        dropaware_v3_scaler_path = st.text_input("DropAware v3 scaler", value="NNModel/scaler_resunet_dropaware_v3rse.gz")

    if dataset == "lhs":
        files = list_lhs_files(str(project_root))
        if not files:
            st.error("No LHS files found in data/LHS")
            return
        labels = [f.name for f in files]
        selected = st.selectbox("LHS file", labels)
        selected_file = next(f for f in files if f.name == selected)
        x_df, real_all, imag_all, seed = load_lhs_seed(str(selected_file))
        dataset_label = f"seed{seed}"
        lhs_selected_name = selected_file.name
    else:
        x_df, real_all, imag_all = load_old_excel(str(project_root))
        dataset_label = "old_excel"
        lhs_selected_name = ""

    idx = st.slider("Sample index", min_value=0, max_value=len(x_df) - 1, value=0, step=1)

    real_true = real_all[idx]
    imag_true = imag_all[idx]
    mag_true = to_mag(real_true, imag_true)

    antenna_model, antenna_scaler = load_antenna(str(project_root), antenna_model_path, antenna_scaler_path)
    base_model, base_scaler = load_resunet(
        str(project_root), base_resunet_model_path, base_resunet_scaler_path, target_len=real_true.shape[0]
    )
    dropaware_model, dropaware_scaler = load_resunet(
        str(project_root), dropaware_v3_model_path, dropaware_v3_scaler_path, target_len=real_true.shape[0]
    )

    x_one = x_df.iloc[[idx]].values.astype(np.float32)
    x_ant = antenna_scaler.transform(x_one).astype(np.float32)
    x_base = base_scaler.transform(x_one).astype(np.float32)
    x_drop = dropaware_scaler.transform(x_one).astype(np.float32)

    with torch.no_grad():
        pred_ant = antenna_model(torch.tensor(x_ant))
        pred_base = base_model(torch.tensor(x_base))
        pred_drop = dropaware_model(torch.tensor(x_drop))

    mag_ant = to_mag(pred_ant[0, 0, :].numpy(), pred_ant[0, 1, :].numpy())
    mag_base = to_mag(pred_base[0, 0, :].numpy(), pred_base[0, 1, :].numpy())
    mag_drop = to_mag(pred_drop[0, 0, :].numpy(), pred_drop[0, 1, :].numpy())

    if apply_gaussian:
        mag_base = gaussian_smooth_1d(mag_base, gauss_sigma, gauss_window)
        mag_drop = gaussian_smooth_1d(mag_drop, gauss_sigma, gauss_window)

    if magnitude_db:
        y_true = to_db(mag_true)
        y_ant = to_db(mag_ant)
        y_base = to_db(mag_base)
        y_drop = to_db(mag_drop)
        y_label = f"|{trace_label}| (dB)"
    else:
        y_true = mag_true
        y_ant = mag_ant
        y_base = mag_base
        y_drop = mag_drop
        y_label = f"|{trace_label}|"

    sample_metrics = pd.DataFrame(
        [
            {"Model": "Antenna NN", "MSE": float(np.mean((y_ant - y_true) ** 2)), "RSE": rse(y_ant, y_true)},
            {"Model": "Base ResUNet", "MSE": float(np.mean((y_base - y_true) ** 2)), "RSE": rse(y_base, y_true)},
            {"Model": "DropAware v3", "MSE": float(np.mean((y_drop - y_true) ** 2)), "RSE": rse(y_drop, y_true)},
        ]
    ).sort_values("RSE", ascending=True).reset_index(drop=True)

    st.subheader("Sample Metrics (Current Sample)")
    st.caption(f"Dataset: {dataset_label} | Sample index: {idx}")
    st.dataframe(sample_metrics, use_container_width=True)

    x_axis = np.arange(len(y_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=y_true, mode="lines", name=f"Real {trace_label}", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=x_axis, y=y_ant, mode="lines", name="Antenna NN", line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x_axis, y=y_base, mode="lines", name="Base ResUNet", line=dict(width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x_axis, y=y_drop, mode="lines", name="DropAware v3", line=dict(width=2, dash="dashdot")))
    fig.update_layout(
        title=f"{trace_label} comparison ({dataset_label}, sample={idx})",
        xaxis_title="Point index",
        yaxis_title=y_label,
        template="plotly_white",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Selected geometry input"):
        st.dataframe(x_df.iloc[[idx]], use_container_width=True)

    st.subheader("Dataset-Level Metrics (MSE / RSE)")
    metrics_df = compute_dataset_metrics(
        project_root=str(project_root),
        dataset=dataset,
        lhs_selected_name=lhs_selected_name,
        magnitude_db=magnitude_db,
        apply_gaussian=apply_gaussian,
        gauss_sigma=gauss_sigma,
        gauss_window=gauss_window,
        antenna_model_path=antenna_model_path,
        antenna_scaler_path=antenna_scaler_path,
        base_resunet_model_path=base_resunet_model_path,
        base_resunet_scaler_path=base_resunet_scaler_path,
        dropaware_v3_model_path=dropaware_v3_model_path,
        dropaware_v3_scaler_path=dropaware_v3_scaler_path,
    )
    st.dataframe(metrics_df, use_container_width=True)
    st.caption("RSE proje icin daha kritik oldugu icin tablo RSE'ye gore artan siralandi.")


if __name__ == "__main__":
    main()
