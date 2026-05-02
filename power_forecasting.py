from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import numpy as np
    import pandas as pd
    from pandas.tseries.frequencies import to_offset
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing required core packages. "
        "Run `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt` first."
    ) from exc


DEFAULT_DATA_PATH = Path("household_power_consumption.txt")
DEFAULT_TARGET = "Global_active_power"
DEFAULT_FREQ = "1h"
DEFAULT_AGG = "mean"
DEFAULT_FORECAST_HORIZON = "24h"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_LAG_SPECS = ("1step", "2step", "3step", "6step", "12step", "1D", "2D", "7D")
DEFAULT_WINDOW_SPECS = ("3step", "6step", "12step", "1D", "7D")
UCI_DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
UCI_DATA_FILE = "household_power_consumption.txt"


@dataclass(frozen=True)
class ModelSpec:
    family: str
    description: str
    factory: Callable[[], object] | None = None


def build_hgbt() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
    )


def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )


def build_extra_trees() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )


def build_mlp() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )


def build_sgd() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SGDRegressor(
                    loss="huber",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=3000,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )


def build_lightgbm() -> object:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "lightgbm is not installed. Run `.\\.venv\\Scripts\\python.exe -m pip install lightgbm`."
        ) from exc

    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )


def build_xgboost() -> object:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Run `.\\.venv\\Scripts\\python.exe -m pip install xgboost`."
        ) from exc

    return XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=42,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "hgbt": ModelSpec("tree", "HistGradientBoostingRegressor", build_hgbt),
    "random_forest": ModelSpec("tree", "RandomForestRegressor", build_random_forest),
    "extra_trees": ModelSpec("tree", "ExtraTreesRegressor", build_extra_trees),
    "mlp": ModelSpec("nn", "MLPRegressor + StandardScaler", build_mlp),
    "sgd": ModelSpec("large_scale", "SGDRegressor + StandardScaler", build_sgd),
    "lightgbm": ModelSpec("large_scale", "LightGBM", build_lightgbm),
    "xgboost": ModelSpec("large_scale", "XGBoost", build_xgboost),
    "lstm": ModelSpec("nn", "PyTorch LSTMRegressor"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast Global_active_power with classical ML models, including XGBoost.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to household_power_consumption.txt")
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download the UCI open dataset when --data-path does not exist.",
    )
    parser.add_argument("--freq", default=DEFAULT_FREQ, help="Resample interval, for example 30min, 1h, or 1D.")
    parser.add_argument("--agg", choices=("mean", "sum", "median"), default=DEFAULT_AGG, help="Resample aggregation.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column to forecast.")
    parser.add_argument("--model", choices=tuple(MODEL_SPECS.keys()), default="hgbt", help="Forecast model.")
    parser.add_argument(
        "--forecast-horizon",
        default=DEFAULT_FORECAST_HORIZON,
        help="Future forecast horizon. Converted into steps from --freq, for example 24h or 7D.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Holdout ratio. Must satisfy 0 < test_size < 1.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory where outputs will be saved.")
    parser.add_argument("--lstm-lookback", default="7D", help="LSTM lookback window, for example 24h or 7D.")
    parser.add_argument("--lstm-epochs", type=int, default=12, help="Training epochs for --model lstm.")
    parser.add_argument("--lstm-batch-size", type=int, default=256, help="Batch size for --model lstm.")
    parser.add_argument("--lstm-hidden-size", type=int, default=64, help="Hidden size for --model lstm.")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of recurrent layers for --model lstm.")
    parser.add_argument("--lstm-learning-rate", type=float, default=1e-3, help="Learning rate for --model lstm.")
    parser.add_argument("--list-models", action="store_true", help="Print available models and exit.")
    return parser.parse_args()


def download_household_power_data(data_path: Path, url: str = UCI_DATA_URL) -> Path:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = data_path.with_suffix(".zip")

    print(f"Downloading open dataset from UCI: {url}")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        if UCI_DATA_FILE not in members:
            raise FileNotFoundError(f"{UCI_DATA_FILE} was not found in {zip_path}.")

        with archive.open(members[UCI_DATA_FILE]) as source, data_path.open("wb") as target:
            target.write(source.read())

    print(f"Saved dataset to {data_path}")
    return data_path


def sanitize_token(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


def get_fixed_freq_delta(freq: str) -> pd.Timedelta:
    offset = to_offset(freq)
    try:
        nanos = offset.nanos
    except ValueError as exc:
        raise ValueError(
            f"`{freq}` is not a fixed-width frequency. Use values like `30min`, `1h`, or `1D`."
        ) from exc
    return pd.Timedelta(nanos, unit="ns")


def resolve_step_spec(spec: str, freq_delta: pd.Timedelta) -> tuple[str, int] | None:
    if spec.endswith("step"):
        return spec, int(spec.removesuffix("step"))

    duration = pd.Timedelta(spec)
    steps = duration / freq_delta
    if steps < 1:
        return None
    if not float(steps).is_integer():
        return None
    return spec, int(steps)


def resolve_specs(
    specs: tuple[str, ...],
    freq_delta: pd.Timedelta,
) -> tuple[list[tuple[str, int]], list[str]]:
    resolved: list[tuple[str, int]] = []
    skipped: list[str] = []

    for spec in specs:
        result = resolve_step_spec(spec, freq_delta)
        if result is None:
            skipped.append(spec)
        else:
            resolved.append(result)

    return resolved, skipped


def horizon_to_steps(horizon: str, freq_delta: pd.Timedelta) -> int:
    duration = pd.Timedelta(horizon)
    steps = duration / freq_delta
    if steps < 1:
        raise ValueError(f"Forecast horizon `{horizon}` is shorter than a single `{freq_delta}` step.")
    if not float(steps).is_integer():
        raise ValueError(f"Forecast horizon `{horizon}` must be an integer multiple of `{freq_delta}`.")
    return int(steps)


def print_models() -> None:
    print("Available models:")
    for name, spec in MODEL_SPECS.items():
        print(f"  - {name:<14} [{spec.family:<16}] {spec.description}")


def load_household_power_data(data_path: Path, download_if_missing: bool = False) -> pd.DataFrame:
    if not data_path.exists():
        if download_if_missing:
            download_household_power_data(data_path)
        else:
            raise FileNotFoundError(
                f"{data_path} was not found. Run `python fetch_open_data.py` or pass `--download-if-missing`."
            )

    df = pd.read_csv(data_path, sep=";", na_values=["?"], low_memory=False)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.drop(columns=["Date", "Time"])
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resample_data(df: pd.DataFrame, freq: str, agg: str) -> pd.DataFrame:
    if agg == "mean":
        resampled = df.resample(freq).mean()
    elif agg == "sum":
        resampled = df.resample(freq).sum()
    elif agg == "median":
        resampled = df.resample(freq).median()
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    return resampled.interpolate(method="time").ffill().bfill()


def make_features(
    data: pd.DataFrame,
    target_col: str,
    lag_specs: list[tuple[str, int]],
    window_specs: list[tuple[str, int]],
) -> pd.DataFrame:
    out = data.copy()

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["day"] = out.index.day
    out["month"] = out.index.month
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    for label, steps in lag_specs:
        out[f"{target_col}_lag_{label}"] = out[target_col].shift(steps)

    for label, steps in window_specs:
        out[f"{target_col}_roll_mean_{label}"] = out[target_col].shift(1).rolling(steps).mean()
        out[f"{target_col}_roll_std_{label}"] = out[target_col].shift(1).rolling(steps).std()

    exog_cols = [col for col in data.columns if col != target_col]
    for col in exog_cols:
        out[f"{col}_lag_1step"] = out[col].shift(1)

    return out


def build_model(model_name: str) -> object:
    factory = MODEL_SPECS[model_name].factory
    if factory is None:
        raise ValueError(f"{model_name} is not a scikit-learn style model.")
    return factory()


def split_train_test(
    feat_df: pd.DataFrame,
    target_col: str,
    test_size: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_idx = int(len(feat_df) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(feat_df):
        raise ValueError("Train or test split became empty. Adjust test_size.")

    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]

    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return x_train, y_train, x_test, y_test


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": float(mae), "rmse": float(rmse)}


def import_torch() -> object:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is not installed. Run `.\\.venv\\Scripts\\python.exe -m pip install torch`."
        ) from exc
    return torch


def build_lstm_module(torch: object, input_size: int, hidden_size: int, num_layers: int) -> object:
    class LSTMRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dropout = 0.1 if num_layers > 1 else 0.0
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            head_hidden = max(16, hidden_size // 2)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, head_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(head_hidden, 1),
            )

        def forward(self, x: object) -> object:
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return LSTMRegressor()


def make_lstm_sequences(
    data: pd.DataFrame,
    target_col: str,
    lookback_steps: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    feature_cols = list(data.columns)
    values = data[feature_cols].to_numpy(dtype=np.float32)
    target_values = data[target_col].to_numpy(dtype=np.float32)

    x = np.stack(
        [values[idx - lookback_steps : idx] for idx in range(lookback_steps, len(data))],
        axis=0,
    )
    y = target_values[lookback_steps:]
    index = data.index[lookback_steps:]
    return x, y, index, feature_cols


def predict_lstm_batches(
    torch: object,
    model: object,
    x: np.ndarray,
    batch_size: int,
    device: object,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            pred = model(batch).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.concatenate(preds)


def iterative_forecast_lstm(
    torch: object,
    model: object,
    history: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    lookback_steps: int,
    forecast_steps: int,
    freq: str,
    device: object,
) -> pd.DataFrame:
    work = history.copy()
    future_preds: list[tuple[pd.Timestamp, float]] = []

    model.eval()
    for _ in range(forecast_steps):
        next_time = work.index[-1] + to_offset(freq)
        seq_values = work[feature_cols].iloc[-lookback_steps:].to_numpy(dtype=np.float32)
        seq_scaled = feature_scaler.transform(seq_values).reshape(1, lookback_steps, -1).astype(np.float32)

        with torch.no_grad():
            pred_scaled = float(model(torch.from_numpy(seq_scaled).to(device)).detach().cpu().numpy()[0, 0])

        pred = float(target_scaler.inverse_transform(np.array([[pred_scaled]], dtype=np.float32))[0, 0])
        next_row = work.iloc[-1:].copy()
        next_row.index = [next_time]
        next_row[target_col] = pred
        work = pd.concat([work, next_row], axis=0)
        future_preds.append((next_time, pred))

    return pd.DataFrame(future_preds, columns=["datetime", "forecast"]).set_index("datetime")


def iterative_forecast_sklearn(
    model: object,
    history: pd.DataFrame,
    target_col: str,
    lag_specs: list[tuple[str, int]],
    window_specs: list[tuple[str, int]],
    forecast_steps: int,
    freq: str,
) -> pd.DataFrame:
    work = history.copy()
    future_preds: list[tuple[pd.Timestamp, float]] = []

    for _ in range(forecast_steps):
        next_time = work.index[-1] + to_offset(freq)
        next_row = work.iloc[-1:].copy()
        next_row.index = [next_time]
        next_row[target_col] = np.nan

        temp = pd.concat([work, next_row], axis=0)
        temp_feat = make_features(temp, target_col, lag_specs, window_specs)
        x_next = temp_feat.drop(columns=[target_col]).iloc[[-1]]

        y_next = float(model.predict(x_next)[0])
        temp.loc[next_time, target_col] = y_next
        work = temp
        future_preds.append((next_time, y_next))

    return pd.DataFrame(future_preds, columns=["datetime", "forecast"]).set_index("datetime")


def run_sklearn_pipeline(
    args: argparse.Namespace,
    resampled: pd.DataFrame,
    lag_specs: list[tuple[str, int]],
    window_specs: list[tuple[str, int]],
    skipped_lags: list[str],
    skipped_windows: list[str],
    forecast_steps: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    if not lag_specs:
        raise ValueError("No valid lag features were created. Check --freq.")
    if not window_specs:
        raise ValueError("No valid rolling features were created. Check --freq.")

    print("Resolved lags:", ", ".join(f"{label}={steps}" for label, steps in lag_specs))
    print("Resolved windows:", ", ".join(f"{label}={steps}" for label, steps in window_specs))
    if skipped_lags:
        print("Skipped lag specs:", ", ".join(skipped_lags))
    if skipped_windows:
        print("Skipped window specs:", ", ".join(skipped_windows))

    feat_df = make_features(resampled, args.target, lag_specs, window_specs).dropna()
    print("Feature shape:", feat_df.shape)

    x_train, y_train, x_test, y_test = split_train_test(feat_df, args.target, args.test_size)
    print("Train:", x_train.shape, y_train.shape)
    print("Test :", x_test.shape, y_test.shape)

    model = build_model(args.model)
    print(f"Training model: {args.model} ({MODEL_SPECS[args.model].description})")
    model.fit(x_train, y_train)

    pred_test = model.predict(x_test)
    metrics = evaluate_predictions(y_test, pred_test)

    test_result = pd.DataFrame({"actual": y_test, "pred": pred_test}, index=y_test.index)
    future_forecast = iterative_forecast_sklearn(
        model=model,
        history=resampled.copy(),
        target_col=args.target,
        lag_specs=lag_specs,
        window_specs=window_specs,
        forecast_steps=forecast_steps,
        freq=args.freq,
    )
    return metrics, test_result, future_forecast


def run_lstm_pipeline(
    args: argparse.Namespace,
    resampled: pd.DataFrame,
    forecast_steps: int,
    freq_delta: pd.Timedelta,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    torch = import_torch()
    torch.manual_seed(42)

    lookback = resolve_step_spec(args.lstm_lookback, freq_delta)
    if lookback is None:
        raise ValueError(f"LSTM lookback `{args.lstm_lookback}` must be a multiple of --freq.")

    lookback_label, lookback_steps = lookback
    if lookback_steps >= len(resampled):
        raise ValueError("LSTM lookback window is longer than the available data.")

    print(f"Resolved LSTM lookback: {lookback_label}={lookback_steps} steps")
    x_raw, y_raw, index, feature_cols = make_lstm_sequences(resampled, args.target, lookback_steps)
    split_idx = int(len(x_raw) * (1 - args.test_size))
    if split_idx <= 0 or split_idx >= len(x_raw):
        raise ValueError("Train or test split became empty. Adjust test_size.")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    n_features = x_raw.shape[-1]
    feature_scaler.fit(x_raw[:split_idx].reshape(-1, n_features))
    target_scaler.fit(y_raw[:split_idx].reshape(-1, 1))

    x_scaled = feature_scaler.transform(x_raw.reshape(-1, n_features)).reshape(x_raw.shape).astype(np.float32)
    y_scaled = target_scaler.transform(y_raw.reshape(-1, 1)).astype(np.float32)

    x_train_all = x_scaled[:split_idx]
    y_train_all = y_scaled[:split_idx]
    x_test = x_scaled[split_idx:]
    y_test = y_raw[split_idx:]
    test_index = index[split_idx:]

    val_size = max(1, int(len(x_train_all) * 0.1))
    x_train = x_train_all[:-val_size]
    y_train = y_train_all[:-val_size]
    x_val = x_train_all[-val_size:]
    y_val = y_train_all[-val_size:]

    print("LSTM feature shape:", x_scaled.shape)
    print("Train:", x_train.shape, y_train.shape)
    print("Validation:", x_val.shape, y_val.shape)
    print("Test :", x_test.shape, y_test.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lstm_module(
        torch=torch,
        input_size=n_features,
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_layers,
    ).to(device)

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.lstm_batch_size,
        shuffle=True,
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lstm_learning_rate)
    best_state = None
    best_val_loss = float("inf")
    patience = 3
    bad_epochs = 0

    print(f"Training model: lstm ({MODEL_SPECS['lstm'].description}) on {device}")
    for epoch in range(1, args.lstm_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(x_val).to(device))
            val_loss = float(criterion(val_pred, torch.from_numpy(y_val).to(device)).detach().cpu())

        train_loss = float(np.mean(train_losses))
        print(f"Epoch {epoch:02d}/{args.lstm_epochs} - train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    pred_scaled = predict_lstm_batches(torch, model, x_test, args.lstm_batch_size, device)
    pred_test = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    y_test_series = pd.Series(y_test, index=test_index, name=args.target)
    metrics = evaluate_predictions(y_test_series, pred_test)
    test_result = pd.DataFrame({"actual": y_test_series, "pred": pred_test}, index=test_index)

    future_forecast = iterative_forecast_lstm(
        torch=torch,
        model=model,
        history=resampled.copy(),
        target_col=args.target,
        feature_cols=feature_cols,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        lookback_steps=lookback_steps,
        forecast_steps=forecast_steps,
        freq=args.freq,
        device=device,
    )
    return metrics, test_result, future_forecast


def save_outputs(
    output_dir: Path,
    model_name: str,
    freq: str,
    metrics: dict[str, float],
    test_result: pd.DataFrame,
    future_forecast: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{sanitize_token(model_name)}_{sanitize_token(freq)}"

    pred_path = output_dir / f"test_predictions_{suffix}.csv"
    forecast_path = output_dir / f"future_forecast_{suffix}.csv"
    metrics_path = output_dir / f"metrics_{suffix}.json"

    test_result.to_csv(pred_path)
    future_forecast.to_csv(forecast_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved files:")
    print(f"- {pred_path}")
    print(f"- {forecast_path}")
    print(f"- {metrics_path}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        print_models()
        return

    freq_delta = get_fixed_freq_delta(args.freq)
    forecast_steps = horizon_to_steps(args.forecast_horizon, freq_delta)
    lag_specs, skipped_lags = resolve_specs(DEFAULT_LAG_SPECS, freq_delta)
    window_specs, skipped_windows = resolve_specs(DEFAULT_WINDOW_SPECS, freq_delta)

    print("Loading data...")
    df = load_household_power_data(args.data_path, download_if_missing=args.download_if_missing)
    print("Raw shape:", df.shape)

    if args.target not in df.columns:
        raise ValueError(f"Target column `{args.target}` was not found in the dataset.")

    resampled = resample_data(df, args.freq, args.agg)
    print(f"Resampled shape ({args.freq}, {args.agg}):", resampled.shape)

    if args.model == "lstm":
        metrics, test_result, future_forecast = run_lstm_pipeline(
            args=args,
            resampled=resampled,
            forecast_steps=forecast_steps,
            freq_delta=freq_delta,
        )
    else:
        metrics, test_result, future_forecast = run_sklearn_pipeline(
            args=args,
            resampled=resampled,
            lag_specs=lag_specs,
            window_specs=window_specs,
            skipped_lags=skipped_lags,
            skipped_windows=skipped_windows,
            forecast_steps=forecast_steps,
        )

    print(f"MAE : {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")

    print("\nTest prediction sample:")
    print(test_result.head(20))

    print(f"\n=== Next {args.forecast_horizon} forecast ({forecast_steps} steps) ===")
    print(future_forecast)

    save_outputs(
        output_dir=args.output_dir,
        model_name=args.model,
        freq=args.freq,
        metrics=metrics,
        test_result=test_result,
        future_forecast=future_forecast,
    )


if __name__ == "__main__":
    main()
