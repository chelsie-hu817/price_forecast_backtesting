from requests.auth import HTTPBasicAuth

import caiso_lmp_forecast
from config import *
import pandas as pd
import datetime
import requests
from io import StringIO
import time
from functools import reduce
import numpy as np
from caiso_lmp_forecast import caiso_energy_da_forecast

from caiso_lmp_forecast import caiso_lmp_da_forecast
import os
from typing import Optional, Tuple

def prepare_data(df, start, end, target, drop_cols):
    y = df.loc[start:end][target]
    X = df.loc[start:end].drop(columns=drop_cols)
    return X, y


def define_drop_cols(forecast_model, df):
    return forecast_model.TARGETS

"""
def get_feature_data(forecast_model):
    # Load yes energy data - dict {yes_energy_item:data}
    yes_energy_data_dict = {}

    for item in forecast_model.YES_ENERGY_ITEMS:
        datatype = forecast_model.YES_ENERGY_ITEMS[item][0]
        object_id = forecast_model.YES_ENERGY_ITEMS[item][1]
        url = f"{base_url}{datatype}/{object_id}"
        response = requests.get(url, params=params, auth=HTTPBasicAuth(username, password))

        if response.status_code == 200 and response.headers.get("Content-Type", ""):
            tables = pd.read_html(StringIO(response.text))
            tables[0]['DATETIME'] = pd.to_datetime(tables[0]['DATETIME'])
            print(item, tables[0].head())
            yes_energy_data_dict[item] = tables[0][['DATETIME', 'TIMEZONE' , 'AVGVALUE']]

        time.sleep(10)

    data = forecast_model.define_data(yes_energy_data_dict)
    dfs = list(data.values())
    df = reduce(lambda left, right: pd.merge(left, right, on=['DATETIME'], how='outer'), dfs)

    return df"""

def get_feature_data(forecast_model):
    """
    Pull Yes Energy series for every item in forecast_model.YES_ENERGY_ITEMS,
    then hand the raw tables to forecast_model.define_data(...) to engineer features.
    Ensures each intermediate frame has a real 'DATETIME' column (not an index).
    """
    from requests.auth import HTTPBasicAuth
    import pandas as pd, time
    from io import StringIO
    import requests
    from functools import reduce

    yes_energy_data_dict = {}

    # 1) Download raw series
    for item, (datatype, object_id) in forecast_model.YES_ENERGY_ITEMS.items():
        url = f"{base_url}{datatype}/{object_id}"
        response = requests.get(url, params=params, auth=HTTPBasicAuth(username, password))
        if response.status_code == 200 and response.headers.get("Content-Type", ""):
            tables = pd.read_html(StringIO(response.text))
            t = tables[0].copy()
            t["DATETIME"] = pd.to_datetime(t["DATETIME"])
            yes_energy_data_dict[item] = t[["DATETIME", "TIMEZONE", "AVGVALUE"]]
            print(item, t.head())
        time.sleep(5)

    # 2) Engineer features via model (calendar, DART, lags, etc.)
    data_dict = forecast_model.define_data(yes_energy_data_dict)  # uses model’s pipeline
    # (Your pipeline constructs features/targets here. :contentReference[oaicite:1]{index=1})

    # 3) Normalize: guarantee DATETIME is a column for every frame
    for k, v in list(data_dict.items()):
        df = v
        if "DATETIME" not in df.columns:
            if df.index.name == "DATETIME":
                df = df.reset_index()
            else:
                # last resort: if index is datetime-like with no name
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    df = df.reset_index().rename(columns={"index": "DATETIME"})
                else:
                    raise ValueError(f"{k} is missing a DATETIME column.")
        # De-duplicate DATETIME just in case
        df = df.sort_values("DATETIME").drop_duplicates(subset=["DATETIME"], keep="last")
        data_dict[k] = df

    # 4) Merge on DATETIME (outer)
    keys = list(data_dict.keys())
    base = data_dict[keys[0]].copy()
    for k in keys[1:]:
        cols = [c for c in data_dict[k].columns if c != "DATETIME"]
        base = base.merge(data_dict[k][["DATETIME"] + cols], on="DATETIME", how="outer")

    base = base.sort_values("DATETIME").reset_index(drop=True)
    return base


def rolling_backtesting(
    forecast_model,
    target_name: str,
    target_datatype: str,
    df: pd.DataFrame,
    train_start,
    train_end,
    backtesting_window: int = 365,
    cutoff_he: int = 10,     # include HE <= cutoff_he on day t; HE=0 means 24
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    For each step i:
      - Training rows:
          * all rows with date in [train_start+i, train_end+i - 1]
          * plus rows on date == (train_end+i) with HE <= cutoff_he = 10
      - Test rows:
          * all rows with date == (train_end+i+1)  (HE 1..24)

    Assumes:
      - df.index is datetime-like.
      - df['Hour of Day'] exists.
    """
    dates = df.index.normalize()
    he = df["Hour Of Day"].astype(int)

    results = []
    rolling_importance = []
    drop_cols = define_drop_cols(forecast_model, df)

    base_train_start = pd.Timestamp(train_start).normalize()
    base_train_end   = pd.Timestamp(train_end).normalize()

    for i in range(0, backtesting_window + 1):
        start_i = base_train_start + pd.Timedelta(days=i)   # first train date
        t_date  = base_train_end   + pd.Timedelta(days=i)   # last train day (with cutoff)
        test_d  = base_train_end   + pd.Timedelta(days=i+1) # next day

        # Training mask = full prior days + partial t_date up to cutoff HE
        prior_days_mask = (dates >= start_i) & (dates < t_date)
        t_cut_mask      = (dates == t_date) & (he <= cutoff_he) & (he != 0)  # 0 (=24) is > any cutoff
        train_mask      = prior_days_mask | t_cut_mask

        # Test mask = all rows on test day (HE 1..24; includes HE==0)
        test_mask = (dates == test_d)

        if not test_mask.any():
            continue

        print("train: ", df.loc[train_mask].index)
        print("test: ", df.loc[test_mask].index)

        X_train = df.loc[train_mask].drop(columns=drop_cols)
        y_train = df.loc[train_mask, target_datatype]
        print("X_train:", X_train.columns)
        print("y_train:", y_train.name)

        X_test  = df.loc[test_mask].drop(columns=drop_cols)
        y_test  = df.loc[test_mask, target_datatype]


        model = forecast_model.MODELS[target_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append(pd.DataFrame({"y_test": y_test, "y_pred_test": y_pred}, index=y_test.index))
        importance = model.booster_.feature_importance(importance_type="gain")
        rolling_importance.append(pd.DataFrame({
            "feature": X_train.columns,
            "importance": importance,
            "date": test_d,
        }))

    result_df = pd.concat(results) if results else pd.DataFrame(columns=["y_test", "y_pred_test"])
    final_imp  = pd.concat(rolling_importance) if rolling_importance else pd.DataFrame(columns=["feature","importance","date"])
    agg_imp    = (final_imp.groupby("feature")["importance"].mean().sort_values(ascending=False)
                  if not final_imp.empty else pd.Series(dtype=float))
    return result_df, agg_imp


def derive_dart_soft_clip(
    df: pd.DataFrame,
    new_clip: float,
    source_col: str = "DART_SOFT_CLIPPED_270",
    base_clip: float = 270.0,
    out_col: str = None,
) -> pd.DataFrame:
    """
    From a soft-clipped DART with base_clip (270), reconstruct the
    un-clipped DART and re-apply soft-clip using new_clip.
    Only use it when getting data directly from existing old feature file
    No need to use it if rerun the feature fetching process

    y_base = base_clip * tanh(x / base_clip)
    => x = base_clip * atanh(y_base / base_clip)

    y_new  = new_clip * tanh(x / new_clip)

    Writes result to out_col (defaults to DART_SOFT_CLIPPED_{new_clip}), drops the old feature (DART_SOFT_CLIPPED_270),
    and returns the modified df.
    """
    if out_col is None:
        out_col = f"DART_SOFT_CLIPPED_{int(new_clip)}"

    y_base = df[source_col].astype(float).to_numpy()

    z = y_base / float(base_clip)

    # recover un-clipped DART
    x = float(base_clip) * np.arctanh(z)

    # re-clip with new_clip
    yc = float(new_clip) * np.tanh(x / float(new_clip))

    df.drop(columns=[source_col], inplace=True)
    df[out_col] = yc

    return df


def main(price, target_name, train_start, train_end, backtesting_window, dart_clip: float = 270):
    """
    price: 'LMP' or 'ENERGY'
    target_name: 'DA' or 'DART' (DART naming is reused for ENERGY)
    dart_clip: used for setting DART Energy clip
    """
    price = price.upper()
    tgt = target_name.upper()

    if price == "ENERGY":
        forecast_model = caiso_energy_da_forecast(name="backtesting_energy",
                                                  price_node=PRICE_NODE_SP15, dart_clip=dart_clip)
        data_path = "energy/feature_data_ENERGY.xlsx"
    elif price == "LMP":
        forecast_model = caiso_lmp_da_forecast(name="backtesting_lmp",
                                               price_node=PRICE_NODE_SP15)
        data_path = "lmp/feature_data_LMP.xlsx"
    else:
        raise ValueError("price must be 'LMP' or 'ENERGY'.")

    # Cache engineered frame per price
    if os.path.exists(data_path):
        data = pd.read_excel(data_path)
    else:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data = get_feature_data(forecast_model)
        data.to_excel(data_path, index=False)

    # Index & clean
    data["DATETIME"] = pd.to_datetime(data["DATETIME"])
    feature = data.copy()
    feature.index = feature["DATETIME"]
    feature = feature.drop(columns=["DATETIME"], axis=1).dropna()

    # Replace the old clipped feature with the new clip
    if dart_clip != 270 and "DART_SOFT_CLIPPED_270" in feature.columns:
        feature = derive_dart_soft_clip(feature, new_clip= clip, source_col= "DART_SOFT_CLIPPED_270", base_clip=270)

    # Map target_name to actual column & model key
    if price == "LMP":
        if tgt == "DA":
            target_col = "DALMP"
            model_key = "DA"
        elif tgt == "DART":
            target_col = forecast_model.CLIPPED_DART
            model_key = "DART"
        else:
            raise ValueError("For LMP, target_name must be 'DA' or 'DART'.")
    else:  # ENERGY (DART naming reused)
        if tgt == "DA":
            target_col = "DAENERGY"
            model_key = "DA"
        elif tgt == "DART":
            target_col = forecast_model.CLIPPED_DART   # ENERGY-based DART name kept
            model_key = "DART"
        else:
            raise ValueError("For ENERGY, target_name must be 'DA' or 'DART'.")

    # Run rolling backtest
    result, importance = rolling_backtesting(
        forecast_model, model_key, target_col, feature,
        train_start, train_end, backtesting_window
    )


    return result, importance


if __name__ == "__main__":
    # Use 3 yr historical data to forecast 1 day
    train_start = datetime.datetime(2022, 1, 1, 0, 0)
    train_end = datetime.datetime(2024, 12, 31, 23, 0)

    """
    # Compare LMP and Energy backtesting
    # LMP backtests
    for tgt in ["DA", "DART"]:
        res, imp = main("LMP", tgt, train_start, train_end, backtesting_window=240)
        res.to_excel(f"lmp/{tgt}_LMP.xlsx", index=True)
        imp.to_excel(f"lmp/{tgt}_LMP_importance.xlsx", index=True)
        print(f"[OK] Saved {tgt} LMP results")

    # ENERGY backtests (DART kept as name)
    for tgt in ["DA", "DART"]:
        res, imp = main("ENERGY", tgt, train_start, train_end, backtesting_window=240)
        res.to_excel(f"energy/{tgt}_ENERGY.xlsx", index=True)
        imp.to_excel(f"energy/{tgt}_ENERGY_importance.xlsx", index=True)
        print(f"[OK] Saved {tgt} ENERGY results")"""

    # Test clip parameter for DART Energy
    clip_grid = [30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

    rows = []
    for clip in clip_grid:
        res, imp = main(
            price="ENERGY",
            target_name="DART",
            train_start=train_start,
            train_end=train_end,
            backtesting_window=240,
            dart_clip=clip,
        )

        res.to_excel(f"energy/clip_compare/DART_ENERGY_clip{clip}.xlsx")

        mae = float(np.mean(np.abs(res["y_test"] - res["y_pred_test"])))
        rmse = float(np.sqrt(np.mean((res["y_test"] - res["y_pred_test"]) ** 2)))
        rows.append({"clip": clip, "MAE": mae, "RMSE": rmse})
        print(f"[OK] clip={clip:>3}  MAE={mae:.3f}  RMSE={rmse:.3f}")

    # rank the clips
    summary = pd.DataFrame(rows).sort_values(["MAE", "RMSE"])
    print("\nBest (by MAE then RMSE):")
    print(summary.head(5))




