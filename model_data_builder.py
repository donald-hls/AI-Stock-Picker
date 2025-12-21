import numpy as np
import pandas as pd

from utility import rolling_total_return, infer_months_per_period

# To take away the outliers -> 98%
TARGET_CLIP_BOUNDS = (-2.0, 2.0)

def _forward_total_return(ret_series, window):
    """
    Compute forward total return over the next `window` periods for a single firm.

    For row t, this returns:
        prod_{k=1..window} (1 + r_{t+k}) - 1

    If any of the required forward returns are missing/non-finite, output is NaN for that row.
    """
    if window <= 0:
        window = 1
    r = pd.to_numeric(ret_series, errors="coerce").to_numpy(dtype="float64", copy=False)
    n = r.shape[0]
    out = np.full(n, np.nan, dtype="float64")
    if n <= window:
        return pd.Series(out, index=ret_series.index)

    finite = np.isfinite(r)
    valid = finite.astype(np.int64)
    cvalid = np.concatenate([[0], np.cumsum(valid)])

    # Use log space for numerical stability: exp(sum(log(1+r))) - 1
    logr = np.where(finite, np.log1p(r), 0.0)
    clog = np.concatenate([[0.0], np.cumsum(logr)])

    # For row i, we need returns r[i+1 : i+1+window]
    # Vectorized window sums using cumulative sums.
    # Length = n - window
    valid_window_sum = cvalid[1 + window : n + 1] - cvalid[1 : n + 1 - window]
    log_window_sum = clog[1 + window : n + 1] - clog[1 : n + 1 - window]
    good = valid_window_sum == window
    out[: n - window] = np.where(good, np.expm1(log_window_sum), np.nan)
    return pd.Series(out, index=ret_series.index)


def add_factor_betas(panel):
    """
    Runs a Time Series Regression to estimate the factor betas for specific stocks
    """
    # X
    factor_cols = ("Mkt_RF_ann", "SMB_ann", "HML_ann")
    windows = 5
    # Minimum number of periods to run the regression
    min_periods = 3
    # Y
    target_col = "excess_ret"
    # Sorting by Firms and Periods to order them chronologically
    df = panel.sort_values(["SP Identifier", "period_end"]).copy()
    beta_cols = [f"beta_{col.lower()}" for col in factor_cols]
    betas = pd.DataFrame(beta_cols, dtype="float64")

    for firm_id, firm_panel in df.groupby("SP Identifier"):
        firm_panel = firm_panel.sort_values("period_end")
        firm_index = firm_panel.index.to_list() # tracking the index of the firm

        for pos, idx in enumerate(firm_index):
            end = pos
            if end == 0:
                continue
            start = max(0, end - windows)
            # to avoid look ahead bias. 
            window_idx = firm_index[start:end]
            # Selecting the target and factor columns for the window
            window_frame = firm_panel.loc[window_idx, [target_col, * factor_cols]].dropna()
            if window_frame.shape[0] < min_periods:
                continue
            # The Stock's excess return (target)
            y_window = window_frame[target_col].astype(float).to_numpy()
            # The Factor returns (X)
            X_window = window_frame[list(factor_cols)].astype(float).to_numpy()
            # Design Matrix, adding a column of ones for the intercept
            X_design = np.column_stack([np.ones(len(y_window)), X_window])
            # Running the OLS regression to find the line of best fit. 
            beta_vector, *_ = np.linalg.lstsq(X_design, y_window, rcond=None)

            for j, factor in enumerate(factor_cols, start=1):
                col_name = f"beta_{factor.lower().replace('-', '_')}"
                betas.at[idx, col_name] = beta_vector[j]

    for col in beta_cols:
        df[col] = betas[col]

    df.rename(columns={"beta_mkt_rf_ann": "beta_mkt_rf", "beta_smb_ann": "beta_smb", "beta_hml_ann": "beta_hml"}, inplace=True)
    return df


def build_model_data(windsorized_path = "windsorized_data.csv", ff_factors_path="fama_french_factors.csv"):
    """
    Build the model data by merging the windsorized data and fama-french factors.
    Add the factor betas to the model data.
    """
    windsorized = pd.read_csv(windsorized_path, parse_dates=["period_end"]).sort_values(["SP Identifier", "period_end"])
    fama_french = pd.read_csv(ff_factors_path, parse_dates=["period_end"])
    fama_french = fama_french.drop(columns=["Unnamed: 0"], errors="ignore")
    merged = pd.merge(windsorized, fama_french, on="period_end", how="left")
    merged = merged.sort_values(["SP Identifier", "period_end"])

    # Infer the panel frequency to correctly define "forward ~12m" in *periods*.
    # - monthly panel => months_per_period ~= 1  => forward_window ~= 12
    # - annual-ish panel => months_per_period ~= 12 => forward_window ~= 1
    months_per_period = infer_months_per_period(merged, id_col="SP Identifier", date_col="period_end")
    forward_window = max(1, int(round(12 / months_per_period)))

    # True forward total return (compounded) over the next ~12 months.
    merged["next_ret_12m"] = merged.groupby("SP Identifier")["ret_1m"].transform(
        lambda s: _forward_total_return(s, forward_window)
    )
    merged["next_ret_12m"] = merged["next_ret_12m"].clip(*TARGET_CLIP_BOUNDS)
    merged["next_excess_ret"] = merged["next_ret_12m"] - merged["next_rf_12m"]
    merged["excess_ret"] = merged["ret_12m"] - merged["Risk Free"]
    merged = add_factor_betas(merged)
    model_data = merged.dropna(subset=["next_excess_ret"]).copy()
    model_data.to_csv("model_data.csv", index=False)

    return model_data


