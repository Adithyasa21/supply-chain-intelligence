from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump

from .utils import load_config, safe_mape


warnings.filterwarnings("ignore", category=FutureWarning)


def _make_weekly_demand(root: Path) -> pd.DataFrame:
    orders = pd.read_csv(root / "data" / "raw" / "orders.csv", parse_dates=["week_start", "order_date"])
    products = pd.read_csv(root / "data" / "raw" / "products.csv")
    regions = pd.read_csv(root / "data" / "raw" / "regions.csv")

    weekly = (
        orders.groupby(["week_start", "sku_id", "region_id"], as_index=False)
        .agg(
            actual_demand=("order_units", "sum"),
            promotion_rate=("promotion_flag", "mean"),
            observed_weather_delay_risk=("weather_delay_risk", "mean"),
        )
    )

    all_weeks = pd.DataFrame({"week_start": pd.date_range(weekly["week_start"].min(), weekly["week_start"].max(), freq="W-MON")})
    base = (
        all_weeks.assign(key=1)
        .merge(products[["sku_id", "category", "velocity_class"]].assign(key=1), on="key")
        .merge(regions[["region_id", "market_size_index", "avg_weather_delay_risk"]].assign(key=1), on="key")
        .drop(columns="key")
    )

    weekly = base.merge(weekly, on=["week_start", "sku_id", "region_id"], how="left")
    weekly["actual_demand"] = weekly["actual_demand"].fillna(0)
    weekly["promotion_rate"] = weekly["promotion_rate"].fillna(0)
    weekly["observed_weather_delay_risk"] = weekly["observed_weather_delay_risk"].fillna(weekly["avg_weather_delay_risk"])

    weekly = weekly.sort_values(["sku_id", "region_id", "week_start"]).reset_index(drop=True)
    group_cols = ["sku_id", "region_id"]

    weekly["lag_1"] = weekly.groupby(group_cols)["actual_demand"].shift(1)
    weekly["lag_4"] = weekly.groupby(group_cols)["actual_demand"].shift(4)
    weekly["rolling_4"] = weekly.groupby(group_cols)["actual_demand"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    weekly["rolling_8"] = weekly.groupby(group_cols)["actual_demand"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week_start"].dt.month

    for col in ["lag_1", "lag_4", "rolling_4", "rolling_8"]:
        weekly[col] = weekly[col].fillna(weekly.groupby(group_cols)["actual_demand"].transform("mean")).fillna(0)

    return weekly


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "sku_id", "region_id", "category", "velocity_class", "market_size_index",
        "promotion_rate", "observed_weather_delay_risk", "lag_1", "lag_4",
        "rolling_4", "rolling_8", "week_of_year", "month"
    ]
    encoded = df[feature_cols + ["actual_demand", "week_start"]].copy()
    encoded = pd.get_dummies(encoded, columns=["sku_id", "region_id", "category", "velocity_class"], drop_first=False)
    return encoded


def run_forecasting_pipeline(root: Path) -> None:
    cfg = load_config(root)
    horizon = int(cfg["forecast_horizon_weeks"])
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    weekly = _make_weekly_demand(root)
    weekly.to_csv(processed_dir / "weekly_demand_features.csv", index=False)

    frame = _encode_features(weekly)
    cutoff = weekly["week_start"].max() - pd.Timedelta(weeks=horizon)
    train = frame[frame["week_start"] <= cutoff].copy()
    test = frame[frame["week_start"] > cutoff].copy()

    X_train = train.drop(columns=["actual_demand", "week_start"])
    y_train = train["actual_demand"]
    X_test = test.drop(columns=["actual_demand", "week_start"])
    y_test = test["actual_demand"]

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=3,
        random_state=int(cfg["random_seed"]),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred_test = np.maximum(0, model.predict(X_test))

    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mape = safe_mape(y_test, pred_test)
    bias = float((pred_test.sum() - y_test.sum()) / max(1, y_test.sum()) * 100)

    # Feature importance for explainability.
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(processed_dir / "forecast_feature_importance.csv", index=False)

    # Future recursive forecast.
    history = weekly.copy()
    model_columns = X_train.columns
    future_rows = []
    products = weekly[["sku_id", "category", "velocity_class"]].drop_duplicates()
    regions = weekly[["region_id", "market_size_index", "avg_weather_delay_risk"]].drop_duplicates()
    last_week = weekly["week_start"].max()

    for step in range(1, horizon + 1):
        week = last_week + pd.Timedelta(weeks=step)
        base = (
            products.assign(key=1)
            .merge(regions.assign(key=1), on="key")
            .drop(columns="key")
        )
        base["week_start"] = week
        base["promotion_rate"] = 0.04
        base["observed_weather_delay_risk"] = base["avg_weather_delay_risk"]
        base["week_of_year"] = int(week.isocalendar().week)
        base["month"] = int(week.month)

        features_for_step = []
        for _, row in base.iterrows():
            hist = history[
                (history["sku_id"] == row["sku_id"]) & (history["region_id"] == row["region_id"])
            ].sort_values("week_start")
            vals = hist["actual_demand"].tail(8).to_numpy()
            lag_1 = vals[-1] if len(vals) >= 1 else 0
            lag_4 = vals[-4] if len(vals) >= 4 else lag_1
            rolling_4 = float(np.mean(vals[-4:])) if len(vals) >= 1 else 0
            rolling_8 = float(np.mean(vals[-8:])) if len(vals) >= 1 else 0

            features_for_step.append({
                "week_start": week,
                "sku_id": row["sku_id"],
                "region_id": row["region_id"],
                "category": row["category"],
                "velocity_class": row["velocity_class"],
                "market_size_index": row["market_size_index"],
                "promotion_rate": row["promotion_rate"],
                "observed_weather_delay_risk": row["observed_weather_delay_risk"],
                "avg_weather_delay_risk": row["avg_weather_delay_risk"],
                "lag_1": lag_1,
                "lag_4": lag_4,
                "rolling_4": rolling_4,
                "rolling_8": rolling_8,
                "week_of_year": row["week_of_year"],
                "month": row["month"],
            })

        step_df = pd.DataFrame(features_for_step)
        model_frame = pd.get_dummies(
            step_df.drop(columns=["week_start", "avg_weather_delay_risk"]),
            columns=["sku_id", "region_id", "category", "velocity_class"],
            drop_first=False,
        )
        model_frame = model_frame.reindex(columns=model_columns, fill_value=0)
        predictions = np.maximum(0, model.predict(model_frame))
        step_df["forecasted_demand"] = predictions
        future_rows.append(step_df)

        recursive_append = step_df.rename(columns={"forecasted_demand": "actual_demand"})
        # Match columns needed by feature engineering for next steps.
        history = pd.concat([
            history,
            recursive_append[[
                "week_start", "sku_id", "region_id", "category", "velocity_class",
                "market_size_index", "avg_weather_delay_risk", "actual_demand",
                "promotion_rate", "observed_weather_delay_risk", "lag_1", "lag_4",
                "rolling_4", "rolling_8", "week_of_year", "month"
            ]]
        ], ignore_index=True)

    future = pd.concat(future_rows, ignore_index=True)
    forecast_output = future[["week_start", "sku_id", "region_id", "forecasted_demand"]].copy()
    forecast_output["forecasted_demand"] = forecast_output["forecasted_demand"].round(2)
    forecast_output["model_name"] = "RandomForestRegressor"
    forecast_output.to_csv(processed_dir / "forecast_output.csv", index=False)

    original_test = weekly.loc[test.index, ["week_start", "sku_id", "region_id", "actual_demand"]].copy()
    original_test["forecasted_demand"] = pred_test.round(2)
    original_test["absolute_error"] = (original_test["actual_demand"] - original_test["forecasted_demand"]).abs()
    original_test["ape"] = original_test["absolute_error"] / original_test["actual_demand"].replace(0, 1)
    original_test.to_csv(processed_dir / "forecast_backtest.csv", index=False)

    metrics = pd.DataFrame([{
        "model_name": "RandomForestRegressor",
        "backtest_weeks": horizon,
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "forecast_bias_pct": round(bias, 2),
    }])
    metrics.to_csv(processed_dir / "forecast_metrics.csv", index=False)
    dump(model, models_dir / "demand_forecast_random_forest.pkl")

    print(f"  Forecasting completed. Backtest MAPE={mape:.2f}%, RMSE={rmse:.2f}, Bias={bias:.2f}%.")
