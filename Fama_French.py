# Risk-free forward curve builder and exporter (annualised workflow)
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from data_selection import computed_data
from utility import rolling_total_return


# Fama-French three-factor
"""
Risk is a function of market risk, firm size, and book-to-market ratio.
Fama-French three-factor model:
    R_i - R_f = alpha_i + beta_i * (R_m - R_f) + beta_j * SMB + beta_k * HML + epsilon_i

The goal here is to build the period-by-period factor returns (Mkt_RF, SMB, HML)
using the annualised panel. The output feeds both the factor-pricing analysis
and the excess-return target (next_rf_12m) used downstream.
"""


def value_weighted_return(df):
    """
    Value-weighted return helper used inside the SMB/HML portfolio grids.
    value_weighted_return(df) -> float
    e.g: (mcap_1 * ret_1 + mcap_2 * ret_2 + ... + mcap_n * ret_n) / (mcap_1 + mcap_2 + ... + mcap_n)
    """
    data = df.dropna(subset=["mcap", "Monthly Total Return"]).copy()
    data = data[data["mcap"] > 0]
    if data.empty:
        return np.nan
    
    total_mcap = data["mcap"].sum()
    if total_mcap <= 0:
        return np.nan

    weights = data["mcap"] / total_mcap
    return np.dot(data["Monthly Total Return"], weights)


def calculate_factors(df, period_col="period_end"):
    """
    calculate_factors constructs SMB and HML factor returns for each period.
    The function returns a DataFrame with the SMB and HML factor returns.
    
    """
    # setting some default thresholds for the SMB and HML factors
    smb_quantiles = (0.5, 0.5) # median split
    hml_quantiles = (0.3, 0.7) # 30% and 70% quantiles
    min_stocks_smb = 10
    min_stocks_hml = 12
    min_bucket_size = 3
    
    df = df.copy()
    ce_mask = (
        df["PB"].isna()
        & df["mcap"].notna()
        & (df["mcap"] > 0)
        & df["Common Equity"].notna()
        & (df["Common Equity"] > 0)
    )
    # Calculate the price-to-book ratio (fill in the NaN ones if possible)
    df.loc[ce_mask, "PB"] = df.loc[ce_mask, "mcap"] / df.loc[ce_mask, "Common Equity"]
    # True if the price-to-book ratio is not 0, False otherwise
    df["Book to Market"] = np.where( df["PB"].replace(0.0, np.nan).notna(), 1.0 / df["PB"].replace(0.0, np.nan), np.nan)
    # storing factor results for each period
    results = []
    for period, period_data in df.groupby(period_col):
        period_slice = period_data.copy()
        # Skip if no stock is available in the period.
        if len(period_slice) == 0:
            continue
        # check if enough stocks are available for SMB
        enough_smb = len(period_slice) >= min_stocks_smb
        valid_bm_mask = period_slice["Book to Market"].notna()
        bm_series = period_slice.loc[valid_bm_mask, "Book to Market"]
        # check if enough stocks are available for HML
        enough_hml = len(bm_series) >= min_stocks_hml
        SMB = np.nan
        HML = np.nan

        if enough_smb and enough_hml:
            # Find the size that splits the SIZE factor 
            size_split = period_slice["mcap"].quantile(smb_quantiles[0])
            if pd.isna(size_split):
                # Append this period as NaN values for SMB and HML
                results.append({period_col: period, "SMB": SMB, "HML": HML})
                continue
            # Small size stocks 
            small_mask = period_slice["mcap"] <= size_split
            # Big size stocks 
            big_mask = period_slice["mcap"] > size_split
            # Require at least 3 stocks in each bucket
            if (small_mask.sum() < min_bucket_size) or (big_mask.sum() < min_bucket_size):
                results.append({period_col: period, "SMB": SMB, "HML": HML})
                continue
            low_bm_quantile = bm_series.quantile(hml_quantiles[0]) # 30% quantile
            high_bm_quantile = bm_series.quantile(hml_quantiles[1]) # 70% quantile
            # Create a Series of labels for the Book to Market ratio
            bm_labels = pd.Series(index = period_slice.index, dtype="object")
            # Stocks with Valid Book to Market ratio and <= 30% quantile -> Low B/M (Growth Stocks)
            low_cond = valid_bm_mask & (period_slice["Book to Market"] <= low_bm_quantile)
            # Stocks with Valid Book to Market ratio and >= 70% quantile -> High B/M (Value Stocks)
            high_cond = valid_bm_mask & (period_slice["Book to Market"] >= high_bm_quantile)
            # Stocks with Valid Book to Market ratio and not in the 30% or 70% quantile -> Mid B/M (Neutral)
            mid_cond = valid_bm_mask & ~(low_cond | high_cond)
            # Assign the labels to the Series
            bm_labels.loc[low_cond] = "L"
            bm_labels.loc[mid_cond] = "M"
            bm_labels.loc[high_cond] = "H"
            # Tracking count of each label 
            bm_counts = bm_labels.value_counts(dropna=True)
            # Check if there are enough stocks in each label
            has_low = bm_counts.get("L", 0) >= min_bucket_size
            has_mid = bm_counts.get("M", 0) >= min_bucket_size
            has_high = bm_counts.get("H", 0) >= min_bucket_size

            if not (has_low and has_mid and has_high):
                results.append({period_col: period, "SMB": SMB, "HML": HML})
                continue
            # 3 x 2 grid of portfolios
            bm_masks = {label: (bm_labels == label) for label in ["L", "M", "H"]}
            size_masks = {"S": small_mask, "B": big_mask}

            cells = {}
            # Iterate over S & M
            for size_label, size_mask in size_masks.items():
                # Iterate over L, M, H
                for bm_label, bm_mask in bm_masks.items():
                    mask = size_mask & bm_mask # Combine both 
                    if mask.sum() >= min_bucket_size:
                        # calculate the value-weighted return for the portfolio
                        cells[f"{size_label}{bm_label}"] = value_weighted_return(period_slice.loc[mask])
                    else:
                        cells[f"{size_label}{bm_label}"] = np.nan
            small_portfolio = [cells.get("SL", np.nan), cells.get("SM", np.nan), cells.get("SH", np.nan)]
            big_portfolio = [cells.get("BL", np.nan), cells.get("BM", np.nan), cells.get("BH", np.nan)]
            # Only if both sides have at least one valid portfolio
            if not (np.all(np.isnan(small_portfolio)) or np.all(np.isnan(big_portfolio))):
                SMB = np.nanmean(small_portfolio) - np.nanmean(big_portfolio)

            high_portfolio = [cells.get("SH", np.nan), cells.get("BH", np.nan)]
            low_portfolio = [cells.get("SL", np.nan), cells.get("BL", np.nan)]
             # Only if both sides have at least one valid portfolio
            if not (np.all(np.isnan(high_portfolio)) or np.all(np.isnan(low_portfolio))):
                HML = np.nanmean(high_portfolio) - np.nanmean(low_portfolio)

        results.append({period_col: period, "SMB": SMB, "HML": HML})
    # print(results)
    
    if not results:
        return pd.DataFrame(columns=[period_col, "SMB", "HML"]).set_index(period_col)

    final_df = pd.DataFrame(results).set_index(period_col).sort_index()
    return final_df


def build_annualized_risk_free_returns(start_date, end_date):
    """
    Build trailing and forward 12-month risk-free returns from 3M T-bills.

    """
    # Extend the window so forward calculations have enough observations
    buffer_start = start_date - pd.offsets.MonthEnd(12)
    buffer_end = end_date + pd.offsets.MonthEnd(12)
    rf_raw = pdr.get_data_fred("TB3MS", start=buffer_start, end=buffer_end)
    rf = rf_raw.resample("ME").last()
    # monthly risk-free return
    rf["rf_1m"] = (rf["TB3MS"] / 100.0) / 12.0
    rf["rf_12m_trailing"] = rolling_total_return(rf["rf_1m"], 12, 12)
    forward_col = []
    # calculate the forward 12-month risk-free return
    rf_values = rf["rf_1m"].to_numpy()
    for idx in range(len(rf)):
        # Take a forward 12-month window of monthly RF rates starting at this month.
        window = rf_values[idx : idx + 12]
        # Only if the window has 12 months and all the values are finite
        if len(window) == 12 and np.all(np.isfinite(window)):
            # Calculate the annualized return
            annualized_return = float(np.prod(1.0 + window) - 1.0)
            forward_col.append(annualized_return)
        else:
            forward_col.append(np.nan)
    rf["next_rf_12m"] = forward_col
    # Extract Monthly Risk-Free Return, Trailing 12-Month Risk-Free Return, and Forward 12-Month Risk-Free Return
    rf_map = rf.loc[:, ["rf_1m", "rf_12m_trailing", "next_rf_12m"]]
    rf_map = rf_map.loc[(rf_map.index >= start_date) & (rf_map.index <= end_date)].copy()
    rf_map.rename(columns={"rf_12m_trailing": "Risk Free", "rf_1m": "rf_1m"}, inplace=True)
    rf_map.index.name = "period_end"
    return rf_map.reset_index()


def three_factor_extraction(df, forward_fill_missing = False, max_forward_fill = 1):
    """
    Calculate Mkt_RF, SMB, HML, and attach the forward risk-free curve.
    """
    df = df.dropna(subset=["mcap", "Monthly Total Return"]).copy()
    df = df[df["mcap"] > 0]
    df["weighted_ret"] = df["mcap"] * df["Monthly Total Return"]
    
    market = df.groupby("period_end").agg(
        weighted_ret=("weighted_ret", "sum"),
        total_mcap=("mcap", "sum"),
    )
    market = market[market["total_mcap"] > 0].copy()
    market["Market"] = market["weighted_ret"] / market["total_mcap"]
    market = market[["Market"]]
    # Get the Risk Free Returns 
    start_date = df["period_end"].min()
    end_date = df["period_end"].max()
    
    rf_table = build_annualized_risk_free_returns(start_date, end_date).set_index("period_end")
    # print(f"Risk Free Returns: {rf_table}")
    factors = market.join(rf_table, how="left")
    # print(f"Joined Factors dataframe: {factors}")
    # factors["Mkt_RF"] = factors["Market"] - factors["Risk Free"]
    factors["Mkt_RF"] = factors["Market"] - factors["rf_1m"]


    smb_hml = calculate_factors(df, period_col="period_end")
    factors = factors.join(smb_hml, how="left")

    # trailing 12-month compounded factor returns for annual alignment
    for col in ["Mkt_RF", "SMB", "HML"]:
        factors[f"{col}_ann"] = rolling_total_return(factors[col], 12, 12)
    # print(f"Annualized Factors: {factors.head()}")
    if forward_fill_missing:
        limit = None if max_forward_fill is None else max_forward_fill
        for col in ["SMB", "HML"]:
            factors[col] = factors[col].ffill(limit=limit)

    factors = factors.reset_index().rename(columns={"period_end": "period_end"})
    return factors


def main():
    factors = three_factor_extraction(computed_data, forward_fill_missing=True)
    print(f"Factors dataframe: {factors.head()}")
    factors.to_csv("fama_french_factors.csv", index = False)

if __name__ == "__main__":
    main()
 