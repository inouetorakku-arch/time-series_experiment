from __future__ import annotations

import argparse
import json
import os
import re
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast Global_active_power with classical ML models, including XGBoost.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to household_power_consumption.txt")
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
    parser.add_argument("--list-models", action="store_true", help="Print available models and exit.")
    return parser.parse_args()


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


def load_household_power_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} was not found. Download household_power_consumption.txt from UCI and place it here."
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
    return MODEL_SPECS[model_name].factory()


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
    df = load_household_power_data(args.data_path)
    print("Raw shape:", df.shape)

    if args.target not in df.columns:
        raise ValueError(f"Target column `{args.target}` was not found in the dataset.")

    resampled = resample_data(df, args.freq, args.agg)
    print(f"Resampled shape ({args.freq}, {args.agg}):", resampled.shape)

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
