# this file contains the Fama-French
import numpy as np
import pandas as pd
from data_selection import windsorized_data
from pandas_datareader import data as pdr

# Fama-French three-factor
"""
Risk is a function of market risk, size of the firms, and total book to market ratio.
Fama-French three-factor model:
    Three factors: size of the firms, book to market ratio, and excess return of the market (momentum)
    R_i - R_f = alpha_i + beta_i * (R_m - Rf) + beta_j * (SMB) + beta_k * (HML) + epsilon_i
    where: 
        R_i - Rf = expected excess return of the stock i, assume at time t. 
        R_M = total market portfolio, assume at time t. 
        SMB = Small Minus Big (for size), assume at time t. 
        HML = High Minus Low (for book to market ratio), assume at time t. 
        epsilon_i = error term, assume at time t. 
        
# the data is monthly factors only, collapses the cross-section into a single observation per month.
        
"""
# purpose: to capture linear systematic exposures to size and book to market ratio.
# uses monthly cross-sectional regression / time series per stock.


# Helper function in calculate_factors
def determine_cutoff(df):
    """
    determine_cutoff is a helper function to determine the cutoff for the SMB and HML factors. 
    The helper uses value weighted return using marketcap as the weights.
    """
    # drop na values in market cap and monthly total return
    data = df.dropna(subset=["mcap", "Monthly Total Return"])
    data = data[data["mcap"] > 0]
    if data.empty or data["mcap"].sum() <= 0:
        return np.nan
    weights = data["mcap"] / data["mcap"].sum()
    # value weighted return
    vw_return = (data["Monthly Total Return"] * weights).sum()
    return vw_return

def calculate_factors(df, smb_quantiles=(0.5, 0.5), hml_quantiles=(0.3, 0.7), 
                      min_stocks_smb=10, min_stocks_hml=12, min_bucket_size=3):
    """
    calculate_factors calculates the SMB and HML factors. 
    calculate_factors(df): DataFrame -> DataFrame. 
    
    smb_quantiles : tuple, default (0.5, 0.5)
        Lower and upper quantiles for SMB (Small Minus Big) calculation
    hml_quantiles : tuple, default (0.3, 0.7)
        Lower and upper quantiles for HML (High Minus Low) calculation
    min_stocks_smb : int, default 10
        Minimum number of stocks required to calculate SMB
    min_stocks_hml : int, default 12
        Minimum number of stocks with Book-to-Market required to calculate HML
    min_bucket_size : int, default 3
        Minimum number of stocks required in each bucket (small/big, high/low)
    """
    ce_mask = (df["PB"].isna() # rows where PB is na 
               & df["mcap"].notna() # rows where mcap is available
               & (df["mcap"] > 0) # rows where mcap is positive
               & df["Common Equity"].notna() # rows where common equity is available
               & (df["Common Equity"] > 0) # rows where common equity is positive
              )
    # derive PB from mcap and common equity
    df.loc[ce_mask, "PB"] = df.loc[ce_mask, "mcap"] / df.loc[ce_mask, "Common Equity"]
    # compute Book-to-Market
    df["Book to Market"] = 1.0 / df["PB"]
    # to store monthly results
    results = []
    for mon, month_data in df.groupby("month"):
        month_data = month_data.copy()
        total_obs = month_data.shape[0]
        smb_ready = total_obs >= min_stocks_smb
        bm_valid_mask = month_data["Book to Market"].notna()
        bm_series = month_data.loc[bm_valid_mask, "Book to Market"]
        hml_ready = bm_series.shape[0] >= min_stocks_hml

        SMB = np.nan
        HML = np.nan
        # if enough data to calculate SMB and HML:
        if smb_ready and hml_ready:
            size_split = month_data["mcap"].quantile(smb_quantiles[0])
            small_mask = month_data["mcap"] <= size_split
            big_mask = month_data["mcap"] > size_split

            if (small_mask.sum() < min_bucket_size) or (big_mask.sum() < min_bucket_size):
                results.append({"month": mon, "SMB": SMB, "HML": HML})
                continue

            q_low_bm = bm_series.quantile(hml_quantiles[0])
            q_high_bm = bm_series.quantile(hml_quantiles[1])

            bm_labels = pd.Series(index=month_data.index, dtype="object")
            low_cond = bm_valid_mask & (month_data["Book to Market"] <= q_low_bm)
            high_cond = bm_valid_mask & (month_data["Book to Market"] >= q_high_bm)
            mid_cond = bm_valid_mask & ~(low_cond | high_cond)

            bm_labels.loc[low_cond] = "L"
            bm_labels.loc[mid_cond] = "M"
            bm_labels.loc[high_cond] = "H"

            bm_counts = bm_labels.value_counts(dropna=True)
            has_low = bm_counts.get("L", 0) >= min_bucket_size
            has_mid = bm_counts.get("M", 0) >= min_bucket_size
            has_high = bm_counts.get("H", 0) >= min_bucket_size

            if not (has_low and has_mid and has_high):
                results.append({"month": mon, "SMB": SMB, "HML": HML})
                continue

            bm_masks = {label: (bm_labels == label) for label in ["L", "M", "H"]}
            size_masks = {"S": small_mask, "B": big_mask}
            cells = {}

            for size_label, size_mask in size_masks.items():
                for bm_label, bm_mask in bm_masks.items():
                    mask = size_mask & bm_mask
                    cells[f"{size_label}{bm_label}"] = (
                        determine_cutoff(month_data.loc[mask]) if mask.sum() >= min_bucket_size else np.nan
                    )

            small_ports = [cells.get("SL", np.nan), cells.get("SM", np.nan), cells.get("SH", np.nan)]
            big_ports = [cells.get("BL", np.nan), cells.get("BM", np.nan), cells.get("BH", np.nan)]
            if not (np.all(np.isnan(small_ports)) or np.all(np.isnan(big_ports))):
                SMB = np.nanmean(small_ports) - np.nanmean(big_ports)

            high_ports = [cells.get("SH", np.nan), cells.get("BH", np.nan)]
            low_ports = [cells.get("SL", np.nan), cells.get("BL", np.nan)]
            if not (np.all(np.isnan(high_ports)) or np.all(np.isnan(low_ports))):
                HML = np.nanmean(high_ports) - np.nanmean(low_ports)

        results.append({"month": mon, "SMB": SMB, "HML": HML})

    final_df = pd.DataFrame(results).set_index("month").sort_index()
    return final_df
    

def three_factor_extraction(df, forward_fill_missing=False, max_forward_fill=1):
    """
    three_factor_extraction calculates the three Fama-French factors.
    namely, R_m - R_f, SMB, and HML. forward_fill_missing is False by default.
    If True, forward-fills missing SMB/HML values from last available month.
    max_forward_fill is 1 by default. Maximum number of consecutive months to forward fill 
    when forward_fill_missing is True.
    - for fama-french, we use short-term rates as risk-free rate.
        - don't use 10Y US treasury due to duration / interest rate risk.
    
    three_factor_extraction(df): DataFrame -> DataFrame. 
    """

    panel = df.dropna(subset=["mcap", "Monthly Total Return"]).copy()
    panel = panel[panel["mcap"] > 0]
    # monthly value-weighted return: sum(mcap * ret) / sum(mcap) without groupby.apply deprecation
    weighted_value = (panel["mcap"] * panel["Monthly Total Return"]).groupby(panel["month"]).sum()
    total_mcap = panel.groupby("month")["mcap"].sum()
    mkt_series = (weighted_value / total_mcap) # value-weighted return
    spx_m = mkt_series.to_frame("Market")
    # 3M T-bill (Considered as risk-free rate)
    risk_free_rate = pdr.get_data_fred("DGS3MO", start=df["month"].min(), end=df["month"].max()).ffill()
    risk_free_rate_m = risk_free_rate.resample("M").last()  
    # Spx500 used as the benchmark index for the market return.
    spx_m.index = spx_m.index.to_period("M")
    risk_free_rate_m.index = risk_free_rate_m.index.to_period("M")
    # DGS3MO is in annual percentage rate, convert to monthly decimal.
    risk_free_rate_m["Risk Free"] = risk_free_rate_m["DGS3MO"] / 100 / 12
    factors = spx_m.join(risk_free_rate_m[["Risk Free"]], how="inner")
    factors["Mkt_RF"] = factors["Market"] - factors["Risk Free"]
    factors.index = factors.index.to_timestamp("M")
    
    # calculate the SMB and HML factors
    factors_2 = calculate_factors(df)
    factors = factors.join(factors_2, how="left")
    factors = factors.dropna(subset=["Mkt_RF"])
    # factors = factors.drop(columns=[col for col in ("Market", "Risk Free") if col in factors.columns])
    # forward fill missing SMB/HML values if requested
    if forward_fill_missing:
        limit = None if max_forward_fill is None else max_forward_fill
        factors["SMB"] = factors["SMB"].ffill(limit=limit)
        factors["HML"] = factors["HML"].ffill(limit=limit)
    
    # set the index to be the month column
    factors.index.name = "month"
    factors = factors.reset_index()
    return factors
    
fama_french_factors = three_factor_extraction(windsorized_data, True)
print(fama_french_factors)
fama_french_factors.to_csv("fama_french_factors.csv")
 